import copy
import math
from dataclasses import asdict, dataclass

import torch

from .adapters import ExpertBank, RouteDecision, sequence_signature
from .corrections import CorrectionDiagnostics, localize_feedback_correction
from .objectives import (
    ctc_posterior_entropy,
    ctc_sequence_loss,
    ctc_target_error,
    feature_anchor_loss,
    posterior_entropy,
    posterior_kl,
)
from .reliability import ReliabilityDecision, ReliabilityGate


@dataclass(frozen=True)
class UpdateOutcome:
    status: str
    supervision: str
    loss_before: float | None
    loss_after: float | None
    anchor_kl: float | None
    reasons: tuple[str, ...]
    correction: CorrectionDiagnostics | None = None

    def to_dict(self):
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class ProcessOutcome:
    transcript: str
    decoder_tokens: tuple[int, ...]
    ctc_tokens: tuple[int, ...]
    route: RouteDecision
    reliability: ReliabilityDecision
    update: UpdateOutcome

    def to_dict(self):
        return {
            "transcript": self.transcript,
            "decoder_tokens": list(self.decoder_tokens),
            "ctc_tokens": list(self.ctc_tokens),
            "route": asdict(self.route),
            "reliability": self.reliability.to_dict(),
            "update": self.update.to_dict(),
        }


class ContinualAdaptationEngine:
    def __init__(
        self,
        base_model,
        expert_bank: ExpertBank,
        reliability_gate: ReliabilityGate,
        decoder=None,
        device="cuda:0",
        enabled=True,
        reliability_enabled=True,
        rollback_enabled=True,
        learning_rate=5e-4,
        weight_decay=1e-4,
        adaptation_steps=1,
        gradient_clip=1.0,
        pseudo_ctc_weight=1.0,
        consistency_weight=0.5,
        entropy_weight=0.01,
        feature_anchor_weight=0.1,
        max_anchor_kl=0.08,
        max_reliability_drop=0.03,
        max_target_loss_increase=0.02,
        view_noise_std=0.01,
        temporal_mask_ratio=0.05,
        blank_id=0,
        feedback_confirms_growth=True,
        feedback_correct_span_kl_enabled=False,
        adaptation_objective="rsp",
        entropy_frame_selection="all",
    ):
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.expert_bank = expert_bank.to(self.device)
        self.reliability_gate = reliability_gate
        self.decoder = decoder
        self.enabled = bool(enabled)
        self.reliability_enabled = bool(reliability_enabled)
        self.rollback_enabled = bool(rollback_enabled)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adaptation_steps = int(adaptation_steps)
        self.gradient_clip = float(gradient_clip)
        self.loss_weights = {
            "pseudo_ctc": float(pseudo_ctc_weight),
            "consistency": float(consistency_weight),
            "entropy": float(entropy_weight),
            "feature_anchor": float(feature_anchor_weight),
        }
        self.max_anchor_kl = float(max_anchor_kl)
        self.max_reliability_drop = float(max_reliability_drop)
        self.max_target_loss_increase = float(max_target_loss_increase)
        self.view_noise_std = float(view_noise_std)
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        self.blank_id = int(blank_id)
        self.feedback_confirms_growth = bool(feedback_confirms_growth)
        self.feedback_correct_span_kl_enabled = bool(
            feedback_correct_span_kl_enabled
        )
        self.adaptation_objective = str(adaptation_objective)
        if self.adaptation_objective not in {"rsp", "tent_adapter"}:
            raise ValueError("适应目标仅支持 rsp 或 tent_adapter")
        self.entropy_frame_selection = str(entropy_frame_selection)
        if self.entropy_frame_selection not in {"all", "nonblank"}:
            raise ValueError("entropy_frame_selection 仅支持 all 或 nonblank")
        if self.adaptation_steps < 1:
            raise ValueError("每次适应的更新步数必须大于 0")
        if self.gradient_clip <= 0:
            raise ValueError("梯度裁剪阈值必须大于 0")
        self._optimizers = {}

    def _weak_view(self, video):
        augmented = video.detach().clone()
        if self.temporal_mask_ratio > 0 and augmented.size(0) > 2:
            span = max(1, round(augmented.size(0) * self.temporal_mask_ratio))
            span = min(span, augmented.size(0) - 1)
            start = int(
                torch.randint(
                    0, augmented.size(0) - span + 1, (1,), device=augmented.device
                ).item()
            )
            replacement = augmented.mean(dim=0, keepdim=True)
            augmented[start : start + span] = replacement
        if self.view_noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.view_noise_std
        return augmented

    @torch.no_grad()
    def _encode(self, video):
        features, _ = self.base_model.encoder(video.unsqueeze(0), None)
        return features.detach()

    @torch.no_grad()
    def extract_route_signature(self, video):
        if not isinstance(video, torch.Tensor):
            raise TypeError("视频输入必须为 torch.Tensor")
        if video.ndim != 4:
            raise ValueError("视频输入必须为 [T, C, H, W]")
        if video.size(0) == 0:
            raise ValueError("视频输入的时间维不能为空")
        features = self._encode(video.to(self.device))
        return sequence_signature(
            features, self.expert_bank.signature_motion_order
        ).detach()

    def _adapted_features(self, frozen_features, expert_index):
        return self.expert_bank(frozen_features, expert_index)

    def _log_probs(self, adapted_features):
        return self.base_model.ctc.log_softmax(adapted_features)

    def _optimizer_for(self, expert_index):
        if expert_index not in self._optimizers:
            self._optimizers[expert_index] = torch.optim.AdamW(
                self.expert_bank.experts[expert_index].parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return self._optimizers[expert_index]

    def optimizer_state_dict(self):
        return {
            str(index): optimizer.state_dict()
            for index, optimizer in self._optimizers.items()
        }

    def load_optimizer_state_dict(self, states):
        for raw_index, state in states.items():
            index = int(raw_index)
            if not 0 <= index < self.expert_bank.expert_count:
                raise ValueError(f"优化器状态引用了不存在的专家：{index}")
            self._optimizer_for(index).load_state_dict(state)

    def _decode(self, adapted_features, ctc_tokens):
        if self.decoder is None:
            return " ".join(map(str, ctc_tokens)), list(ctc_tokens)
        return self.decoder(adapted_features.squeeze(0))

    def _skip_update(self, supervision, reasons):
        return UpdateOutcome(
            status="skipped",
            supervision=supervision,
            loss_before=None,
            loss_after=None,
            anchor_kl=None,
            reasons=tuple(reasons),
        )

    def _transactional_update(
        self,
        expert_index,
        clean_features,
        augmented_features,
        base_log_probs,
        teacher_log_probs,
        target_tokens,
        pre_reliability,
        supervision,
    ):
        adapter = self.expert_bank.experts[expert_index]
        optimizer = self._optimizer_for(expert_index)
        adapter_snapshot = {
            name: value.detach().clone() for name, value in adapter.state_dict().items()
        }
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        correction = None
        consistency_mask = None
        if supervision == "feedback":
            correction, correct_frame_mask = localize_feedback_correction(
                teacher_log_probs, target_tokens, self.blank_id
            )
            if self.feedback_correct_span_kl_enabled:
                consistency_mask = correct_frame_mask

        def restore_snapshot():
            adapter.load_state_dict(adapter_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
            adapter.eval()

        with torch.no_grad():
            loss_before = float(
                ctc_sequence_loss(
                    teacher_log_probs, target_tokens, self.blank_id
                ).item()
            )

        failure_reason = None
        adapter.train()
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad(set_to_none=True)
            clean_adapted = self._adapted_features(clean_features, expert_index)
            augmented_adapted = self._adapted_features(
                augmented_features, expert_index
            )
            clean_log_probs = self._log_probs(clean_adapted)
            augmented_log_probs = self._log_probs(augmented_adapted)
            pseudo_ctc = 0.5 * (
                ctc_sequence_loss(clean_log_probs, target_tokens, self.blank_id)
                + ctc_sequence_loss(
                    augmented_log_probs, target_tokens, self.blank_id
                )
            )
            consistency = posterior_kl(
                teacher_log_probs,
                augmented_log_probs,
                frame_mask=consistency_mask,
            )
            entropy = posterior_entropy(clean_log_probs)
            feature_anchor = feature_anchor_loss(clean_adapted, clean_features)
            total_loss = (
                self.loss_weights["pseudo_ctc"] * pseudo_ctc
                + self.loss_weights["consistency"] * consistency
                + self.loss_weights["entropy"] * entropy
                + self.loss_weights["feature_anchor"] * feature_anchor
            )
            if not torch.isfinite(total_loss):
                failure_reason = "non_finite_loss"
                break
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), self.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                failure_reason = "non_finite_gradient"
                break
            optimizer.step()

        if failure_reason:
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=None,
                anchor_kl=None,
                reasons=(failure_reason,),
                correction=correction,
            )

        adapter.eval()
        with torch.no_grad():
            post_clean = self._adapted_features(clean_features, expert_index)
            post_augmented = self._adapted_features(
                augmented_features, expert_index
            )
            post_clean_log_probs = self._log_probs(post_clean)
            post_augmented_log_probs = self._log_probs(post_augmented)
            loss_after = float(
                ctc_sequence_loss(
                    post_clean_log_probs, target_tokens, self.blank_id
                ).item()
            )
            anchor_kl = float(
                posterior_kl(base_log_probs, post_clean_log_probs).item()
            )
            post_ctc_tokens, _ = self.reliability_gate.evaluate(
                post_clean_log_probs, post_augmented_log_probs
            )
            _, post_decoder_tokens = self._decode(
                post_clean.detach(), post_ctc_tokens
            )
            _, post_reliability = self.reliability_gate.evaluate(
                post_clean_log_probs,
                post_augmented_log_probs,
                post_decoder_tokens,
            )

        if not all(
            math.isfinite(value)
            for value in (loss_before, loss_after, anchor_kl, post_reliability.score)
        ):
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=("non_finite_post_update_metric",),
                correction=correction,
            )

        rejection_reasons = []
        if loss_after > loss_before + self.max_target_loss_increase:
            rejection_reasons.append("target_loss_regressed")
        if anchor_kl > self.max_anchor_kl:
            rejection_reasons.append("anchor_divergence_exceeded")
        if (
            self.reliability_enabled
            and pre_reliability.accepted
            and not post_reliability.accepted
        ):
            rejection_reasons.append("post_update_unreliable")
        if (
            self.reliability_enabled
            and supervision == "pseudo"
            and post_reliability.score
            < pre_reliability.score - self.max_reliability_drop
        ):
            rejection_reasons.append("reliability_regressed")

        if rejection_reasons and self.rollback_enabled:
            restore_snapshot()
            return UpdateOutcome(
                status="rolled_back",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=tuple(rejection_reasons),
                correction=correction,
            )

        return UpdateOutcome(
            status="accepted",
            supervision=supervision,
            loss_before=loss_before,
            loss_after=loss_after,
            anchor_kl=anchor_kl,
            reasons=tuple(rejection_reasons),
            correction=correction,
        )

    def _transactional_entropy_update(
        self,
        expert_index,
        clean_features,
        base_log_probs,
    ):
        adapter = self.expert_bank.experts[expert_index]
        optimizer = self._optimizer_for(expert_index)
        adapter_snapshot = {
            name: value.detach().clone() for name, value in adapter.state_dict().items()
        }
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        def restore_snapshot():
            adapter.load_state_dict(adapter_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
            adapter.eval()

        adapter.eval()
        with torch.no_grad():
            pre_adapted = self._adapted_features(clean_features, expert_index)
            pre_log_probs = self._log_probs(pre_adapted)
            if (
                self.entropy_frame_selection == "nonblank"
                and not pre_log_probs.argmax(dim=-1).ne(self.blank_id).any()
            ):
                return self._skip_update("entropy", ["no_nonblank_frames"])
            loss_before = float(
                ctc_posterior_entropy(
                    pre_log_probs,
                    blank_id=self.blank_id,
                    frame_selection=self.entropy_frame_selection,
                ).item()
            )

        failure_reason = None
        adapter.train()
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad(set_to_none=True)
            adapted = self._adapted_features(clean_features, expert_index)
            log_probs = self._log_probs(adapted)
            entropy = ctc_posterior_entropy(
                log_probs,
                blank_id=self.blank_id,
                frame_selection=self.entropy_frame_selection,
            )
            if not torch.isfinite(entropy):
                failure_reason = "non_finite_loss"
                break
            entropy.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), self.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                failure_reason = "non_finite_gradient"
                break
            optimizer.step()

        if failure_reason:
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision="entropy",
                loss_before=loss_before,
                loss_after=None,
                anchor_kl=None,
                reasons=(failure_reason,),
            )

        adapter.eval()
        with torch.no_grad():
            post_adapted = self._adapted_features(clean_features, expert_index)
            post_log_probs = self._log_probs(post_adapted)
            loss_after = float(
                ctc_posterior_entropy(
                    post_log_probs,
                    blank_id=self.blank_id,
                    frame_selection=self.entropy_frame_selection,
                ).item()
            )
            anchor_kl = float(posterior_kl(base_log_probs, post_log_probs).item())

        if not all(
            math.isfinite(value) for value in (loss_before, loss_after, anchor_kl)
        ):
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision="entropy",
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=("non_finite_post_update_metric",),
            )

        rejection_reasons = []
        if loss_after > loss_before + self.max_target_loss_increase:
            rejection_reasons.append("entropy_regressed")
        if anchor_kl > self.max_anchor_kl:
            rejection_reasons.append("anchor_divergence_exceeded")
        if rejection_reasons and self.rollback_enabled:
            restore_snapshot()
            return UpdateOutcome(
                status="rolled_back",
                supervision="entropy",
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=tuple(rejection_reasons),
            )
        return UpdateOutcome(
            status="accepted",
            supervision="entropy",
            loss_before=loss_before,
            loss_after=loss_after,
            anchor_kl=anchor_kl,
            reasons=tuple(rejection_reasons),
        )

    def process(self, video, feedback_tokens=None):
        if video.ndim != 4:
            raise ValueError("视频输入必须为 [T, C, H, W]")
        video = video.to(self.device)
        clean_features = self._encode(video)
        augmented_features = (
            self._encode(self._weak_view(video))
            if self.enabled and self.adaptation_objective == "rsp"
            else clean_features
        )
        signature = sequence_signature(
            clean_features, self.expert_bank.signature_motion_order
        )
        has_feedback = feedback_tokens is not None
        route = self.expert_bank.route(
            signature,
            confirm_shift=has_feedback and self.feedback_confirms_growth,
        )
        adapter = self.expert_bank.experts[route.expert_index]
        adapter.eval()

        with torch.no_grad():
            clean_adapted = self._adapted_features(
                clean_features, route.expert_index
            )
            augmented_adapted = self._adapted_features(
                augmented_features, route.expert_index
            )
            clean_log_probs = self._log_probs(clean_adapted)
            augmented_log_probs = self._log_probs(augmented_adapted)
            base_log_probs = self._log_probs(clean_features)

        provisional_tokens, _ = self.reliability_gate.evaluate(
            clean_log_probs, augmented_log_probs
        )
        transcript, decoder_tokens = self._decode(
            clean_adapted.detach(), provisional_tokens
        )
        pseudo_tokens, reliability = self.reliability_gate.evaluate(
            clean_log_probs, augmented_log_probs, decoder_tokens
        )

        supervision = "feedback" if has_feedback else "pseudo"
        target_tokens = list(feedback_tokens) if has_feedback else pseudo_tokens
        target_error = (
            ctc_target_error(clean_log_probs, target_tokens, self.blank_id)
            if target_tokens
            else None
        )
        if not self.enabled:
            update = self._skip_update(supervision, ["adaptation_disabled"])
        elif route.quarantined:
            update = self._skip_update(supervision, ["shift_quarantine"])
        elif self.adaptation_objective == "tent_adapter":
            update = self._transactional_entropy_update(
                route.expert_index,
                clean_features,
                base_log_probs,
            )
        elif (
            not has_feedback
            and self.reliability_enabled
            and not reliability.accepted
        ):
            update = self._skip_update(supervision, reliability.reasons)
        elif not target_tokens:
            update = self._skip_update(supervision, ["empty_target"])
        elif target_error is not None:
            update = self._skip_update(supervision, [target_error])
        else:
            update = self._transactional_update(
                route.expert_index,
                clean_features,
                augmented_features,
                base_log_probs,
                clean_log_probs.detach(),
                target_tokens,
                reliability,
                supervision,
            )
            if update.status == "accepted":
                self.expert_bank.mark_accepted(route.expert_index, signature)

        return ProcessOutcome(
            transcript=transcript,
            decoder_tokens=tuple(decoder_tokens),
            ctc_tokens=tuple(pseudo_tokens),
            route=route,
            reliability=reliability,
            update=update,
        )
