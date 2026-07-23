from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RouteDecision:
    expert_index: int
    created: bool
    similarity: float
    pending_shift_count: int
    quarantined: bool


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, dropout=0.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(self.feature_dim, int(bottleneck_dim), bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(int(bottleneck_dim), self.feature_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.ones_(self.scale)

    def forward(self, features):
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"adapter 特征维度应为 {self.feature_dim}，实际为 {features.size(-1)}"
            )
        delta = self.up(self.dropout(self.activation(self.down(self.norm(features)))))
        return features + self.scale * delta


class FeatureWiseAffineAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale_delta = nn.Parameter(torch.zeros(self.feature_dim))
        self.shift = nn.Parameter(torch.zeros(self.feature_dim))

    def forward(self, features):
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"adapter 特征维度应为 {self.feature_dim}，实际为 {features.size(-1)}"
            )
        return features * (1.0 + self.scale_delta) + self.shift


def sequence_signature(features, motion_order=0):
    if features.ndim != 3 or features.size(0) != 1:
        raise ValueError("路由签名当前只支持形状为 [1, T, D] 的单条序列")
    if features.size(1) == 0:
        raise ValueError("路由签名的时间维不能为空")
    motion_order = int(motion_order)
    if not 0 <= motion_order <= 2:
        raise ValueError("运动签名阶数必须介于 0 和 2 之间")

    features = features.detach().float()
    statistics = [
        features.mean(dim=1),
        features.std(dim=1, unbiased=False),
    ]
    motion = features
    for _ in range(motion_order):
        motion = (
            torch.diff(motion, dim=1)
            if motion.size(1) > 1
            else torch.zeros_like(motion[:, :1])
        )
        statistics.append(motion.abs().mean(dim=1))
    return F.normalize(torch.cat(statistics, dim=-1), dim=-1).squeeze(0)


class ExpertBank(nn.Module):
    def __init__(
        self,
        feature_dim,
        bottleneck_dim,
        max_experts=8,
        route_threshold=0.9,
        growth_patience=6,
        pending_similarity=0.95,
        prototype_momentum=0.98,
        signature_motion_order=0,
        dropout=0.0,
        allow_growth=True,
        adapter_type="bottleneck",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.signature_motion_order = int(signature_motion_order)
        if not 0 <= self.signature_motion_order <= 2:
            raise ValueError("运动签名阶数必须介于 0 和 2 之间")
        self.prototype_dim = self.feature_dim * (2 + self.signature_motion_order)
        self.bottleneck_dim = int(bottleneck_dim)
        self.max_experts = int(max_experts)
        self.route_threshold = float(route_threshold)
        self.growth_patience = int(growth_patience)
        self.pending_similarity = float(pending_similarity)
        self.prototype_momentum = float(prototype_momentum)
        self.dropout = float(dropout)
        self.allow_growth = bool(allow_growth)
        self.adapter_type = str(adapter_type)
        if self.adapter_type not in {"bottleneck", "feature_film"}:
            raise ValueError("adapter_type 仅支持 bottleneck 或 feature_film")
        if self.max_experts < 1:
            raise ValueError("专家数量上限必须大于 0")
        if self.growth_patience < 1:
            raise ValueError("域偏移确认次数必须大于 0")
        self.experts = nn.ModuleList()
        self.register_buffer(
            "prototypes", torch.zeros(self.max_experts, self.prototype_dim)
        )
        self.register_buffer(
            "prototype_counts", torch.zeros(self.max_experts, dtype=torch.long)
        )
        self.register_buffer(
            "route_counts", torch.zeros(self.max_experts, dtype=torch.long)
        )
        self.register_buffer(
            "accepted_updates", torch.zeros(self.max_experts, dtype=torch.long)
        )
        self.register_buffer("pending_prototype", torch.zeros(self.prototype_dim))
        self.register_buffer("pending_count", torch.zeros((), dtype=torch.long))

    @property
    def expert_count(self):
        return len(self.experts)

    def _new_adapter(self):
        if self.adapter_type == "feature_film":
            adapter = FeatureWiseAffineAdapter(self.feature_dim)
        else:
            adapter = ResidualBottleneckAdapter(
                self.feature_dim, self.bottleneck_dim, self.dropout
            )
        return adapter.to(self.prototypes.device)

    @torch.no_grad()
    def add_expert(self, signature):
        if self.expert_count >= self.max_experts:
            raise RuntimeError("专家数量已达到上限")
        index = self.expert_count
        self.experts.append(self._new_adapter())
        normalized = F.normalize(signature.float(), dim=-1)
        self.prototypes[index].copy_(normalized)
        self.prototype_counts[index] = 1
        return index

    def ensure_experts(self, count):
        if count > self.max_experts:
            raise ValueError("待加载专家数量超过当前配置上限")
        while self.expert_count < count:
            self.add_expert(self.pending_prototype)

    @torch.no_grad()
    def _reset_pending(self):
        self.pending_prototype.zero_()
        self.pending_count.zero_()

    @torch.no_grad()
    def _observe_shift(self, signature):
        signature = F.normalize(signature.float(), dim=-1)
        if self.pending_count.item() == 0:
            self.pending_prototype.copy_(signature)
            self.pending_count.fill_(1)
            return
        similarity = F.cosine_similarity(
            self.pending_prototype.unsqueeze(0), signature.unsqueeze(0)
        ).item()
        if similarity < self.pending_similarity:
            self.pending_prototype.copy_(signature)
            self.pending_count.fill_(1)
            return
        count = int(self.pending_count.item())
        updated = (self.pending_prototype * count + signature) / (count + 1)
        self.pending_prototype.copy_(F.normalize(updated, dim=-1))
        self.pending_count.add_(1)

    @torch.no_grad()
    def route(self, signature, confirm_shift=False):
        signature = F.normalize(signature.float(), dim=-1)
        if signature.numel() != self.prototype_dim:
            raise ValueError(
                f"路由签名维度应为 {self.prototype_dim}，实际为 {signature.numel()}"
            )
        if self.expert_count == 0:
            index = self.add_expert(signature)
            self.route_counts[index] += 1
            return RouteDecision(index, True, 1.0, 0, False)

        similarities = torch.mv(self.prototypes[: self.expert_count], signature)
        similarity, best_index = similarities.max(dim=0)
        best_index = int(best_index.item())
        similarity_value = float(similarity.item())

        if similarity_value >= self.route_threshold or not self.allow_growth:
            self._reset_pending()
            self.route_counts[best_index] += 1
            return RouteDecision(best_index, False, similarity_value, 0, False)

        if self.expert_count >= self.max_experts:
            self._reset_pending()
            self.route_counts[best_index] += 1
            return RouteDecision(best_index, False, similarity_value, 0, True)

        self._observe_shift(signature)
        pending_count = int(self.pending_count.item())
        if bool(confirm_shift) or pending_count >= self.growth_patience:
            new_index = self.add_expert(self.pending_prototype)
            self._reset_pending()
            self.route_counts[new_index] += 1
            return RouteDecision(new_index, True, similarity_value, pending_count, False)

        self.route_counts[best_index] += 1
        return RouteDecision(best_index, False, similarity_value, pending_count, True)

    @torch.no_grad()
    def confirm_pending_shift(self, route):
        if not isinstance(route, RouteDecision):
            raise TypeError("route 必须为 RouteDecision")
        if (
            not route.quarantined
            or not self.allow_growth
            or self.expert_count >= self.max_experts
            or self.pending_count.item() == 0
        ):
            return int(route.expert_index)
        new_index = self.add_expert(self.pending_prototype)
        self._reset_pending()
        return new_index

    @torch.no_grad()
    def mark_accepted(self, expert_index, signature):
        count = int(self.prototype_counts[expert_index].item())
        signature = F.normalize(signature.float(), dim=-1)
        if count == 0:
            updated = signature
        else:
            updated = (
                self.prototype_momentum * self.prototypes[expert_index]
                + (1.0 - self.prototype_momentum) * signature
            )
        self.prototypes[expert_index].copy_(F.normalize(updated, dim=-1))
        self.prototype_counts[expert_index] += 1
        self.accepted_updates[expert_index] += 1

    def forward(self, features, expert_index):
        return self.experts[int(expert_index)](features)

    def summary(self):
        return {
            "expert_count": self.expert_count,
            "adapter_type": self.adapter_type,
            "signature_motion_order": self.signature_motion_order,
            "route_counts": self.route_counts[: self.expert_count].cpu().tolist(),
            "accepted_updates": self.accepted_updates[: self.expert_count]
            .cpu()
            .tolist(),
            "prototype_counts": self.prototype_counts[: self.expert_count]
            .cpu()
            .tolist(),
            "pending_shift_count": int(self.pending_count.item()),
        }
