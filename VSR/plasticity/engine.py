import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .adapters import ExpertBank, RouteDecision, sequence_signature
from .corrections import (
    CorrectionDiagnostics,
    localize_feedback_correction,
    localize_target_conditioned_occupancy,
    randomized_error_support,
)
from .objectives import (
    ctc_posterior_entropy,
    ctc_sequence_loss,
    ctc_target_error,
    feature_anchor_loss,
    occupancy_weighted_blank_loss,
    occupancy_weighted_posterior_kl,
    occupancy_weighted_token_loss,
    posterior_entropy,
    posterior_kl,
)
from .parameter_adaptation import (
    configure_batch_norm_adaptation,
    load_parameter_state,
    named_lora_parameters,
    parameter_state,
    validate_named_parameters,
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
    objective_before: float | None = None
    objective_after: float | None = None
    localization: dict | None = None

    def to_dict(self):
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class ProcessOutcome:
    transcript: str
    decoder_tokens: tuple[int, ...]
    decoder_nbest: tuple[dict, ...]
    ctc_tokens: tuple[int, ...]
    route: RouteDecision
    adaptation_expert_index: int
    reliability: ReliabilityDecision
    update: UpdateOutcome

    def to_dict(self):
        value = {
            "transcript": self.transcript,
            "decoder_tokens": list(self.decoder_tokens),
            "ctc_tokens": list(self.ctc_tokens),
            "route": asdict(self.route),
            "adaptation_expert_index": self.adaptation_expert_index,
            "reliability": self.reliability.to_dict(),
            "update": self.update.to_dict(),
        }
        if self.decoder_nbest:
            value["decoder_nbest"] = copy.deepcopy(list(self.decoder_nbest))
        return value


@dataclass(frozen=True)
class PredictionState:
    transcript: str
    decoder_tokens: tuple[int, ...]
    decoder_nbest: tuple[dict, ...]
    ctc_tokens: tuple[int, ...]
    route: RouteDecision
    reliability: ReliabilityDecision
    signature: torch.Tensor
    clean_features: torch.Tensor
    augmented_features: torch.Tensor
    base_log_probs: torch.Tensor
    clean_log_probs: torch.Tensor
    video: torch.Tensor


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
        feedback_update_strategy="full_sequence",
        feedback_local_error_target_weight=1.0,
        feedback_local_insertion_blank_weight=1.0,
        feedback_local_matched_kl_weight=0.5,
        feedback_random_control_seed=0,
        non_feedback_updates_enabled=True,
        adaptation_objective="rsp",
        entropy_frame_selection="all",
        parameter_update_mode="adapter",
        parameter_module_names=(),
        eta_entropy_margin=None,
        eta_redundancy_margin=0.05,
        eta_momentum=0.9,
    ):
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.expert_bank = expert_bank.to(self.device)
        self.parameter_update_mode = str(parameter_update_mode)
        if self.parameter_update_mode not in {"adapter", "batch_norm", "lora"}:
            raise ValueError("参数更新模式仅支持 adapter、batch_norm 或 lora")
        configured_parameter_module_names = tuple(parameter_module_names)
        if self.parameter_update_mode == "batch_norm":
            self._named_adaptation_parameters = validate_named_parameters(
                configure_batch_norm_adaptation(self.base_model)
            )
        elif self.parameter_update_mode == "lora":
            self.base_model.eval()
            self.base_model.requires_grad_(False)
            self._named_adaptation_parameters = validate_named_parameters(
                named_lora_parameters(self.base_model)
            )
            if not self._named_adaptation_parameters:
                raise ValueError("LoRA 模式没有找到可更新参数")
            for _, parameter in self._named_adaptation_parameters:
                parameter.requires_grad_(True)
        else:
            self._named_adaptation_parameters = ()
        self.parameter_module_names = configured_parameter_module_names or tuple(
            sorted(
                {
                    name.rsplit(".", 1)[0]
                    for name, _ in self._named_adaptation_parameters
                }
            )
        )
        if self.parameter_update_mode != "adapter":
            self.expert_bank.requires_grad_(False)
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
        feedback_update_strategy = str(feedback_update_strategy)
        if feedback_correct_span_kl_enabled:
            if feedback_update_strategy != "full_sequence":
                raise ValueError(
                    "旧 correct-span 开关不能与新 feedback_update_strategy 同时启用"
                )
            feedback_update_strategy = "greedy_correct_span"
        if feedback_update_strategy not in {
            "full_sequence",
            "greedy_correct_span",
            "ctc_error_local",
            "ctc_error_local_random",
            "ctc_error_hybrid",
            "ctc_error_hybrid_random",
        }:
            raise ValueError("未知 feedback update strategy")
        self.feedback_update_strategy = feedback_update_strategy
        self.feedback_correct_span_kl_enabled = (
            feedback_update_strategy == "greedy_correct_span"
        )
        self.feedback_local_weights = {
            "error_target": float(feedback_local_error_target_weight),
            "insertion_blank": float(feedback_local_insertion_blank_weight),
            "matched_kl": float(feedback_local_matched_kl_weight),
        }
        if any(value < 0 for value in self.feedback_local_weights.values()):
            raise ValueError("局部反馈损失权重不能为负数")
        if not any(self.feedback_local_weights.values()):
            raise ValueError("局部反馈损失至少需要一个非零权重")
        self.feedback_random_control_seed = int(feedback_random_control_seed)
        self.non_feedback_updates_enabled = bool(non_feedback_updates_enabled)
        self.adaptation_objective = str(adaptation_objective)
        if self.adaptation_objective not in {
            "rsp",
            "tent_adapter",
            "bn_tent",
            "eta",
            "online_lora",
        }:
            raise ValueError("未知适应目标")
        expected_parameter_mode = {
            "rsp": "adapter",
            "tent_adapter": "adapter",
            "bn_tent": "batch_norm",
            "eta": "batch_norm",
            "online_lora": "lora",
        }[self.adaptation_objective]
        if self.parameter_update_mode != expected_parameter_mode:
            raise ValueError(
                f"适应目标 {self.adaptation_objective} 要求参数模式 "
                f"{expected_parameter_mode}"
            )
        self.entropy_frame_selection = str(entropy_frame_selection)
        if self.entropy_frame_selection not in {"all", "nonblank"}:
            raise ValueError("entropy_frame_selection 仅支持 all 或 nonblank")
        if self.adaptation_steps < 1:
            raise ValueError("每次适应的更新步数必须大于 0")
        if self.gradient_clip <= 0:
            raise ValueError("梯度裁剪阈值必须大于 0")
        self.eta_entropy_margin = (
            None if eta_entropy_margin is None else float(eta_entropy_margin)
        )
        if self.eta_entropy_margin is not None and self.eta_entropy_margin <= 0:
            raise ValueError("ETA entropy margin 必须大于 0")
        self.eta_redundancy_margin = float(eta_redundancy_margin)
        if not 0.0 <= self.eta_redundancy_margin <= 1.0:
            raise ValueError("ETA redundancy margin 必须介于 0 和 1 之间")
        self.eta_momentum = float(eta_momentum)
        if not 0.0 <= self.eta_momentum < 1.0:
            raise ValueError("ETA momentum 必须介于 0（含）和 1（不含）之间")
        if (
            self.adaptation_objective in {"bn_tent", "eta"}
            and not self.non_feedback_updates_enabled
        ):
            raise ValueError("BN-TENT/ETA 必须启用无反馈更新")
        self._eta_reference = None
        self._eta_resolved_entropy_margin = None
        self._eta_reliable_samples = 0
        self._eta_nonredundant_samples = 0
        self._optimizers = {}
        self._parameter_optimizer = None

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
        if self.parameter_update_mode != "adapter":
            return frozen_features
        return self.expert_bank(frozen_features, expert_index)

    def _log_probs(self, adapted_features):
        return self.base_model.ctc.log_softmax(adapted_features)

    def _forward_parameter_model(self, video):
        features, _ = self.base_model.encoder(video.unsqueeze(0), None)
        return features, self._log_probs(features)

    def adaptation_parameters(self):
        if self.parameter_update_mode == "adapter":
            if not self.enabled:
                return ()
            return tuple(self.expert_bank.parameters())
        return tuple(parameter for _, parameter in self._named_adaptation_parameters)

    def adaptation_summary(self):
        return {
            "parameter_update_mode": self.parameter_update_mode,
            "parameter_count": sum(
                parameter.numel() for parameter in self.adaptation_parameters()
            ),
            "parameter_tensors": (
                len(tuple(self.expert_bank.parameters()))
                if self.parameter_update_mode == "adapter" and self.enabled
                else len(self._named_adaptation_parameters)
            ),
            "parameter_names": [
                name for name, _ in self._named_adaptation_parameters
            ],
            "parameter_modules": list(self.parameter_module_names),
            "eta_entropy_margin": self.eta_entropy_margin,
            "eta_resolved_entropy_margin": self._eta_resolved_entropy_margin,
            "eta_redundancy_margin": self.eta_redundancy_margin,
            "eta_momentum": self.eta_momentum,
            "eta_reference_initialized": self._eta_reference is not None,
            "eta_reliable_samples": self._eta_reliable_samples,
            "eta_nonredundant_samples": self._eta_nonredundant_samples,
        }

    def adaptation_state_dict(self):
        return {
            "parameter_update_mode": self.parameter_update_mode,
            "parameters": parameter_state(self._named_adaptation_parameters),
            "eta_reference": (
                self._eta_reference.detach().cpu().clone()
                if self._eta_reference is not None
                else None
            ),
            "eta_resolved_entropy_margin": self._eta_resolved_entropy_margin,
            "eta_reliable_samples": self._eta_reliable_samples,
            "eta_nonredundant_samples": self._eta_nonredundant_samples,
        }

    def load_adaptation_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("checkpoint 缺少参数适应状态")
        if state.get("parameter_update_mode") != self.parameter_update_mode:
            raise ValueError("checkpoint 参数更新模式与当前配置不一致")
        load_parameter_state(
            self._named_adaptation_parameters,
            state.get("parameters") or {},
        )
        eta_reference = state.get("eta_reference")
        if eta_reference is not None:
            if not isinstance(eta_reference, torch.Tensor) or eta_reference.ndim != 1:
                raise ValueError("checkpoint 的 ETA reference 无效")
            vocabulary_size = getattr(
                getattr(self.base_model.ctc, "ctc_lo", None),
                "out_features",
                None,
            )
            if vocabulary_size is not None and eta_reference.numel() != vocabulary_size:
                raise ValueError("checkpoint 的 ETA reference 词表维度不匹配")
            eta_reference = eta_reference.to(self.device)
            if not torch.isfinite(eta_reference).all():
                raise ValueError("checkpoint 的 ETA reference 包含非有限值")
        self._eta_reference = eta_reference
        resolved_margin = state.get("eta_resolved_entropy_margin")
        if resolved_margin is not None:
            resolved_margin = float(resolved_margin)
            if not math.isfinite(resolved_margin) or resolved_margin <= 0:
                raise ValueError("checkpoint 的 ETA entropy margin 无效")
        self._eta_resolved_entropy_margin = resolved_margin
        self._eta_reliable_samples = int(state.get("eta_reliable_samples", 0))
        self._eta_nonredundant_samples = int(
            state.get("eta_nonredundant_samples", 0)
        )
        if self._eta_reliable_samples < 0 or self._eta_nonredundant_samples < 0:
            raise ValueError("checkpoint 的 ETA 样本计数无效")
        if self._eta_nonredundant_samples > self._eta_reliable_samples:
            raise ValueError("checkpoint 的 ETA 样本计数不守恒")

    def _optimizer_for(self, expert_index):
        if expert_index not in self._optimizers:
            self._optimizers[expert_index] = torch.optim.AdamW(
                self.expert_bank.experts[expert_index].parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return self._optimizers[expert_index]

    def _parameter_optimizer_for(self):
        if self.parameter_update_mode == "adapter":
            raise RuntimeError("adapter 模式不能创建参数级优化器")
        if self._parameter_optimizer is None:
            self._parameter_optimizer = torch.optim.AdamW(
                self.adaptation_parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return self._parameter_optimizer

    def optimizer_state_dict(self):
        if self.parameter_update_mode != "adapter":
            return (
                {"parameters": self._parameter_optimizer.state_dict()}
                if self._parameter_optimizer is not None
                else {}
            )
        return {
            str(index): optimizer.state_dict()
            for index, optimizer in self._optimizers.items()
        }

    def load_optimizer_state_dict(self, states):
        if self.parameter_update_mode != "adapter":
            if not states:
                return
            if set(states) != {"parameters"}:
                raise ValueError("参数级优化器 checkpoint 字段无效")
            self._parameter_optimizer_for().load_state_dict(states["parameters"])
            return
        for raw_index, state in states.items():
            index = int(raw_index)
            if not 0 <= index < self.expert_bank.expert_count:
                raise ValueError(f"优化器状态引用了不存在的专家：{index}")
            self._optimizer_for(index).load_state_dict(state)

    @staticmethod
    def _state_is_finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(
                ContinualAdaptationEngine._state_is_finite(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return all(
                ContinualAdaptationEngine._state_is_finite(item) for item in value
            )
        return True

    def _parameter_loss(self, log_probs, target_tokens):
        if target_tokens is None:
            return ctc_posterior_entropy(
                log_probs,
                blank_id=self.blank_id,
                frame_selection=self.entropy_frame_selection,
            )
        return ctc_sequence_loss(log_probs, target_tokens, self.blank_id)

    def _transactional_parameter_update(
        self,
        video,
        base_log_probs,
        *,
        supervision,
        target_tokens=None,
        loss_scale=1.0,
    ):
        parameters = self.adaptation_parameters()
        if not parameters:
            raise RuntimeError("参数级适应没有可更新参数")
        optimizer = self._parameter_optimizer_for()
        parameter_snapshot = parameter_state(self._named_adaptation_parameters)
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        loss_scale = float(loss_scale)
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("参数级适应的 loss scale 必须为正有限值")

        def restore_snapshot():
            load_parameter_state(
                self._named_adaptation_parameters,
                parameter_snapshot,
            )
            optimizer.load_state_dict(optimizer_snapshot)
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_before = float(
                self._parameter_loss(base_log_probs, target_tokens).item()
            )

        failure_reason = None
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad(set_to_none=True)
            _, log_probs = self._forward_parameter_model(video)
            loss = self._parameter_loss(log_probs, target_tokens) * loss_scale
            if not torch.isfinite(loss):
                failure_reason = "non_finite_loss"
                break
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                failure_reason = "non_finite_gradient"
                break
            optimizer.step()
            if not self._state_is_finite(parameters) or not self._state_is_finite(
                optimizer.state_dict()
            ):
                failure_reason = "non_finite_parameter_state"
                break

        if failure_reason:
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=None,
                anchor_kl=None,
                reasons=(failure_reason,),
            )

        with torch.no_grad():
            _, post_log_probs = self._forward_parameter_model(video)
            loss_after = float(
                self._parameter_loss(post_log_probs, target_tokens).item()
            )
            anchor_kl = float(posterior_kl(base_log_probs, post_log_probs).item())

        if not all(
            math.isfinite(value) for value in (loss_before, loss_after, anchor_kl)
        ):
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=("non_finite_post_update_metric",),
            )

        rejection_reasons = []
        if loss_after > loss_before + self.max_target_loss_increase:
            rejection_reasons.append(
                "entropy_regressed"
                if target_tokens is None
                else "target_loss_regressed"
            )
        if anchor_kl > self.max_anchor_kl:
            rejection_reasons.append("anchor_divergence_exceeded")
        if rejection_reasons and self.rollback_enabled:
            restore_snapshot()
            return UpdateOutcome(
                status="rolled_back",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=tuple(rejection_reasons),
            )
        return UpdateOutcome(
            status="accepted",
            supervision=supervision,
            loss_before=loss_before,
            loss_after=loss_after,
            anchor_kl=anchor_kl,
            reasons=tuple(rejection_reasons),
        )

    def _eta_filter(self, log_probs):
        entropy = float(
            ctc_posterior_entropy(
                log_probs,
                blank_id=self.blank_id,
                frame_selection=self.entropy_frame_selection,
            ).item()
        )
        if not math.isfinite(entropy):
            return False, None, "non_finite_entropy"
        if self._eta_resolved_entropy_margin is None:
            self._eta_resolved_entropy_margin = (
                self.eta_entropy_margin
                if self.eta_entropy_margin is not None
                else max(1e-6, 0.5 * math.log(log_probs.size(-1)) - 1.0)
            )
        margin = self._eta_resolved_entropy_margin
        if entropy >= margin:
            return False, None, "eta_unreliable"
        self._eta_reliable_samples += 1

        probabilities = log_probs.detach().exp()
        nonblank_frames = log_probs.detach().argmax(dim=-1).ne(self.blank_id)
        if nonblank_frames.any():
            probability = probabilities[nonblank_frames].mean(dim=0)
        else:
            probability = probabilities.mean(dim=(0, 1))
        probability = probability.clone()
        probability[self.blank_id] = 0
        probability = probability / probability.sum().clamp_min(
            torch.finfo(probability.dtype).tiny
        )
        similarity = None
        if self._eta_reference is not None:
            similarity = float(
                F.cosine_similarity(
                    self._eta_reference.unsqueeze(0),
                    probability.unsqueeze(0),
                    dim=1,
                ).item()
            )
            if not math.isfinite(similarity):
                return False, None, "non_finite_eta_similarity"
            if abs(similarity) >= self.eta_redundancy_margin:
                return False, None, "eta_redundant"

        self._eta_nonredundant_samples += 1
        if self._eta_reference is None:
            self._eta_reference = probability.clone()
        else:
            self._eta_reference.mul_(self.eta_momentum).add_(
                probability, alpha=1.0 - self.eta_momentum
            )
        coefficient = math.exp(margin - entropy)
        return True, coefficient, None

    def _parameter_entropy_update(self, prediction):
        if (
            self.entropy_frame_selection == "nonblank"
            and not prediction.clean_log_probs.argmax(dim=-1).ne(self.blank_id).any()
        ):
            return self._skip_update("entropy", ["no_nonblank_frames"])
        loss_scale = 1.0
        if self.adaptation_objective == "eta":
            selected, loss_scale, reason = self._eta_filter(
                prediction.clean_log_probs
            )
            if not selected:
                return self._skip_update("entropy", [reason])
        return self._transactional_parameter_update(
            prediction.video,
            prediction.base_log_probs,
            supervision="entropy",
            target_tokens=None,
            loss_scale=loss_scale,
        )

    def _decode(self, adapted_features, ctc_tokens):
        if self.decoder is None:
            return " ".join(map(str, ctc_tokens)), list(ctc_tokens), ()
        features = adapted_features.squeeze(0)
        if hasattr(self.decoder, "decode_with_nbest"):
            return self.decoder.decode_with_nbest(features)
        transcript, tokens = self.decoder(features)
        return transcript, tokens, ()

    def _skip_update(
        self, supervision, reasons, *, correction=None, localization=None
    ):
        return UpdateOutcome(
            status="skipped",
            supervision=supervision,
            loss_before=None,
            loss_after=None,
            anchor_kl=None,
            reasons=tuple(reasons),
            correction=correction,
            localization=localization,
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
        localization=None,
    ):
        adapter = self.expert_bank.experts[expert_index]
        optimizer = self._optimizer_for(expert_index)
        adapter_snapshot = {
            name: value.detach().clone() for name, value in adapter.state_dict().items()
        }
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        correction = localization["correction"] if localization is not None else None
        localization_summary = (
            localization["localization_summary"]
            if localization is not None
            else None
        )
        consistency_mask = None
        if supervision == "feedback" and localization is None:
            correction, correct_frame_mask = localize_feedback_correction(
                teacher_log_probs, target_tokens, self.blank_id
            )
            if self.feedback_correct_span_kl_enabled:
                consistency_mask = correct_frame_mask

        def restore_snapshot():
            adapter.load_state_dict(adapter_snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
            adapter.eval()

        def localized_objective(log_probs):
            if localization is None:
                return None
            return self._localized_feedback_aux(
                log_probs,
                teacher_log_probs,
                target_tokens,
                localization,
            )

        with torch.no_grad():
            loss_before = float(
                ctc_sequence_loss(
                    teacher_log_probs, target_tokens, self.blank_id
                ).item()
            )
            objective_before_tensor = localized_objective(teacher_log_probs)
            objective_before = (
                float(objective_before_tensor.item())
                if objective_before_tensor is not None
                else None
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
            auxiliary_loss = localized_objective(clean_log_probs)
            if auxiliary_loss is not None:
                total_loss = total_loss + auxiliary_loss
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
                objective_before=objective_before,
                localization=localization_summary,
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
            objective_after_tensor = localized_objective(post_clean_log_probs)
            objective_after = (
                float(objective_after_tensor.item())
                if objective_after_tensor is not None
                else None
            )
            anchor_kl = float(
                posterior_kl(base_log_probs, post_clean_log_probs).item()
            )
            post_ctc_tokens, _ = self.reliability_gate.evaluate(
                post_clean_log_probs, post_augmented_log_probs
            )
            _, post_decoder_tokens, _ = self._decode(
                post_clean.detach(), post_ctc_tokens
            )
            _, post_reliability = self.reliability_gate.evaluate(
                post_clean_log_probs,
                post_augmented_log_probs,
                post_decoder_tokens,
            )

        finite_metrics = [loss_before, loss_after, anchor_kl, post_reliability.score]
        if objective_before is not None:
            finite_metrics.extend((objective_before, objective_after))
        if not all(math.isfinite(value) for value in finite_metrics):
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision=supervision,
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=("non_finite_post_update_metric",),
                correction=correction,
                objective_before=objective_before,
                objective_after=objective_after,
                localization=localization_summary,
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
                objective_before=objective_before,
                objective_after=objective_after,
                localization=localization_summary,
            )

        return UpdateOutcome(
            status="accepted",
            supervision=supervision,
            loss_before=loss_before,
            loss_after=loss_after,
            anchor_kl=anchor_kl,
            reasons=tuple(rejection_reasons),
            correction=correction,
            objective_before=objective_before,
            objective_after=objective_after,
            localization=localization_summary,
        )

    def _prepare_ctc_error_localization(
        self, teacher_log_probs, target_tokens, sample_key, randomized
    ):
        correction, _ = localize_feedback_correction(
            teacher_log_probs, target_tokens, self.blank_id
        )
        alignment = localize_target_conditioned_occupancy(
            teacher_log_probs, target_tokens, self.blank_id
        )
        error_target_indices = (
            alignment.substitution_target_indices
            + alignment.deletion_target_indices
        )
        token_occupancy = alignment.token_occupancy
        insertion_weights = alignment.extra_prediction_frame_mask
        if randomized:
            token_occupancy, insertion_weights = randomized_error_support(
                alignment,
                sample_key,
                seed=self.feedback_random_control_seed,
            )
        if error_target_indices:
            error_index_tensor = torch.tensor(
                error_target_indices,
                dtype=torch.long,
                device=token_occupancy.device,
            )
            error_frame_weights = token_occupancy.index_select(
                -1, error_index_tensor
            ).sum(dim=-1)
        else:
            error_frame_weights = torch.zeros_like(
                alignment.matched_target_occupancy
            )

        def effective_frames(weights):
            mass = float(weights.sum().item())
            squared_mass = float(weights.square().sum().item())
            return mass, mass * mass / squared_mass if squared_mass else 0.0

        matched_mass, matched_effective_frames = effective_frames(
            alignment.matched_target_occupancy
        )
        error_mass, error_effective_frames = effective_frames(error_frame_weights)
        localization_summary = {
            "strategy": self.feedback_update_strategy,
            "randomized_support": bool(randomized),
            "ctc_frames": int(teacher_log_probs.size(1)),
            "target_log_likelihood": float(alignment.log_likelihood.item()),
            "matched_target_tokens": len(alignment.matched_target_indices),
            "error_target_tokens": len(error_target_indices),
            "substitution_target_tokens": len(
                alignment.substitution_target_indices
            ),
            "deletion_target_tokens": len(alignment.deletion_target_indices),
            "insertion_frames": int(insertion_weights.sum().item()),
            "matched_occupancy_mass": matched_mass,
            "error_occupancy_mass": error_mass,
            "matched_effective_frames": matched_effective_frames,
            "error_effective_frames": error_effective_frames,
        }
        return {
            "correction": correction,
            "alignment": alignment,
            "error_target_indices": error_target_indices,
            "token_occupancy": token_occupancy,
            "insertion_weights": insertion_weights,
            "localization_summary": localization_summary,
        }

    def _localized_feedback_aux(
        self,
        log_probs,
        teacher_log_probs,
        target_tokens,
        localization,
    ):
        error_target = occupancy_weighted_token_loss(
            log_probs,
            target_tokens,
            localization["token_occupancy"],
            localization["error_target_indices"],
        )
        insertion_blank = occupancy_weighted_blank_loss(
            log_probs, localization["insertion_weights"], self.blank_id
        )
        matched_kl = occupancy_weighted_posterior_kl(
            teacher_log_probs,
            log_probs,
            localization["alignment"].matched_target_occupancy,
        )
        return (
            self.feedback_local_weights["error_target"] * error_target
            + self.feedback_local_weights["insertion_blank"] * insertion_blank
            + self.feedback_local_weights["matched_kl"] * matched_kl
        )

    def _transactional_local_feedback_update(
        self,
        expert_index,
        clean_features,
        augmented_features,
        base_log_probs,
        teacher_log_probs,
        target_tokens,
        pre_reliability,
        sample_key,
    ):
        randomized = self.feedback_update_strategy == "ctc_error_local_random"
        localization = self._prepare_ctc_error_localization(
            teacher_log_probs, target_tokens, sample_key, randomized
        )
        correction = localization["correction"]
        localization_summary = localization["localization_summary"]
        if (
            not localization["error_target_indices"]
            and not localization["insertion_weights"].any()
        ):
            return self._skip_update(
                "feedback",
                ["feedback_already_correct"],
                correction=correction,
                localization=localization_summary,
            )

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

        def local_objective(log_probs, adapted_features):
            return self._localized_feedback_aux(
                log_probs,
                teacher_log_probs,
                target_tokens,
                localization,
            ) + self.loss_weights["feature_anchor"] * feature_anchor_loss(
                adapted_features, clean_features
            )

        with torch.no_grad():
            loss_before = float(
                ctc_sequence_loss(
                    teacher_log_probs, target_tokens, self.blank_id
                ).item()
            )
            objective_before = float(
                local_objective(teacher_log_probs, clean_features).item()
            )

        failure_reason = None
        adapter.train()
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad(set_to_none=True)
            clean_adapted = self._adapted_features(clean_features, expert_index)
            clean_log_probs = self._log_probs(clean_adapted)
            objective = local_objective(clean_log_probs, clean_adapted)
            if not torch.isfinite(objective):
                failure_reason = "non_finite_loss"
                break
            objective.backward()
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
                supervision="feedback",
                loss_before=loss_before,
                loss_after=None,
                anchor_kl=None,
                reasons=(failure_reason,),
                correction=correction,
                objective_before=objective_before,
                localization=localization_summary,
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
            objective_after = float(
                local_objective(post_clean_log_probs, post_clean).item()
            )
            anchor_kl = float(
                posterior_kl(base_log_probs, post_clean_log_probs).item()
            )
            post_ctc_tokens, _ = self.reliability_gate.evaluate(
                post_clean_log_probs, post_augmented_log_probs
            )
            _, post_decoder_tokens, _ = self._decode(
                post_clean.detach(), post_ctc_tokens
            )
            _, post_reliability = self.reliability_gate.evaluate(
                post_clean_log_probs,
                post_augmented_log_probs,
                post_decoder_tokens,
            )

        if not all(
            math.isfinite(value)
            for value in (
                loss_before,
                loss_after,
                objective_before,
                objective_after,
                anchor_kl,
                post_reliability.score,
            )
        ):
            restore_snapshot()
            return UpdateOutcome(
                status="failed",
                supervision="feedback",
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=("non_finite_post_update_metric",),
                correction=correction,
                objective_before=objective_before,
                objective_after=objective_after,
                localization=localization_summary,
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
        if rejection_reasons and self.rollback_enabled:
            restore_snapshot()
            return UpdateOutcome(
                status="rolled_back",
                supervision="feedback",
                loss_before=loss_before,
                loss_after=loss_after,
                anchor_kl=anchor_kl,
                reasons=tuple(rejection_reasons),
                correction=correction,
                objective_before=objective_before,
                objective_after=objective_after,
                localization=localization_summary,
            )
        return UpdateOutcome(
            status="accepted",
            supervision="feedback",
            loss_before=loss_before,
            loss_after=loss_after,
            anchor_kl=anchor_kl,
            reasons=tuple(rejection_reasons),
            correction=correction,
            objective_before=objective_before,
            objective_after=objective_after,
            localization=localization_summary,
        )

    def _transactional_hybrid_feedback_update(
        self,
        expert_index,
        clean_features,
        augmented_features,
        base_log_probs,
        teacher_log_probs,
        target_tokens,
        pre_reliability,
        sample_key,
    ):
        """Full-sequence primary loss + CTC-error occupancy auxiliary terms."""
        randomized = self.feedback_update_strategy == "ctc_error_hybrid_random"
        localization = self._prepare_ctc_error_localization(
            teacher_log_probs, target_tokens, sample_key, randomized
        )
        return self._transactional_update(
            expert_index,
            clean_features,
            augmented_features,
            base_log_probs,
            teacher_log_probs,
            target_tokens,
            pre_reliability,
            "feedback",
            localization=localization,
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

    def predict(self, video):
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
        route = (
            self.expert_bank.route(signature, confirm_shift=False)
            if self.parameter_update_mode == "adapter"
            else RouteDecision(0, False, 1.0, 0, False)
        )
        if self.parameter_update_mode == "adapter":
            self.expert_bank.experts[route.expert_index].eval()

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
        transcript, decoder_tokens, decoder_nbest = self._decode(
            clean_adapted.detach(), provisional_tokens
        )
        pseudo_tokens, reliability = self.reliability_gate.evaluate(
            clean_log_probs, augmented_log_probs, decoder_tokens
        )
        return PredictionState(
            transcript=transcript,
            decoder_tokens=tuple(decoder_tokens),
            decoder_nbest=tuple(decoder_nbest),
            ctc_tokens=tuple(pseudo_tokens),
            route=route,
            reliability=reliability,
            signature=signature,
            clean_features=clean_features,
            augmented_features=augmented_features,
            base_log_probs=base_log_probs,
            clean_log_probs=clean_log_probs.detach(),
            video=video.detach(),
        )

    def adapt(self, prediction, feedback_tokens=None, sample_key=None):
        if not isinstance(prediction, PredictionState):
            raise TypeError("prediction 必须来自当前 engine.predict()")
        has_feedback = feedback_tokens is not None
        supervision = "feedback" if has_feedback else "pseudo"
        target_tokens = (
            list(feedback_tokens) if has_feedback else list(prediction.ctc_tokens)
        )
        target_error = (
            ctc_target_error(
                prediction.clean_log_probs, target_tokens, self.blank_id
            )
            if target_tokens
            else None
        )
        adaptation_expert_index = int(prediction.route.expert_index)
        if (
            self.enabled
            and has_feedback
            and target_tokens
            and target_error is None
            and self.feedback_confirms_growth
            and prediction.route.quarantined
        ):
            adaptation_expert_index = self.expert_bank.confirm_pending_shift(
                prediction.route
            )
        confirmed_shift = adaptation_expert_index != prediction.route.expert_index
        if not self.enabled:
            update = self._skip_update(supervision, ["adaptation_disabled"])
        elif not has_feedback and not self.non_feedback_updates_enabled:
            update = self._skip_update(
                supervision, ["non_feedback_updates_disabled"]
            )
        elif prediction.route.quarantined and not confirmed_shift:
            update = self._skip_update(supervision, ["shift_quarantine"])
        elif self.adaptation_objective == "tent_adapter":
            update = self._transactional_entropy_update(
                adaptation_expert_index,
                prediction.clean_features,
                prediction.base_log_probs,
            )
        elif self.adaptation_objective in {"bn_tent", "eta"}:
            update = self._parameter_entropy_update(prediction)
        elif self.adaptation_objective == "online_lora" and not has_feedback:
            update = self._skip_update(supervision, ["feedback_required"])
        elif (
            not has_feedback
            and self.reliability_enabled
            and not prediction.reliability.accepted
        ):
            update = self._skip_update(
                supervision, prediction.reliability.reasons
            )
        elif not target_tokens:
            update = self._skip_update(supervision, ["empty_target"])
        elif target_error is not None:
            update = self._skip_update(supervision, [target_error])
        elif self.adaptation_objective == "online_lora":
            update = self._transactional_parameter_update(
                prediction.video,
                prediction.base_log_probs,
                supervision="feedback",
                target_tokens=target_tokens,
            )
        elif has_feedback and self.feedback_update_strategy in {
            "ctc_error_local",
            "ctc_error_local_random",
        }:
            update = self._transactional_local_feedback_update(
                adaptation_expert_index,
                prediction.clean_features,
                prediction.augmented_features,
                prediction.base_log_probs,
                prediction.clean_log_probs,
                target_tokens,
                prediction.reliability,
                sample_key,
            )
            if update.status == "accepted":
                self.expert_bank.mark_accepted(
                    adaptation_expert_index, prediction.signature
                )
        elif has_feedback and self.feedback_update_strategy in {
            "ctc_error_hybrid",
            "ctc_error_hybrid_random",
        }:
            update = self._transactional_hybrid_feedback_update(
                adaptation_expert_index,
                prediction.clean_features,
                prediction.augmented_features,
                prediction.base_log_probs,
                prediction.clean_log_probs,
                target_tokens,
                prediction.reliability,
                sample_key,
            )
            if update.status == "accepted":
                self.expert_bank.mark_accepted(
                    adaptation_expert_index, prediction.signature
                )
        else:
            update = self._transactional_update(
                adaptation_expert_index,
                prediction.clean_features,
                prediction.augmented_features,
                prediction.base_log_probs,
                prediction.clean_log_probs,
                target_tokens,
                prediction.reliability,
                supervision,
            )
            if update.status == "accepted":
                self.expert_bank.mark_accepted(
                    adaptation_expert_index, prediction.signature
                )

        return ProcessOutcome(
            transcript=prediction.transcript,
            decoder_tokens=prediction.decoder_tokens,
            decoder_nbest=prediction.decoder_nbest,
            ctc_tokens=prediction.ctc_tokens,
            route=prediction.route,
            adaptation_expert_index=adaptation_expert_index,
            reliability=prediction.reliability,
            update=update,
        )

    def process(self, video, feedback_tokens=None, sample_key=None):
        prediction = self.predict(video)
        return self.adapt(
            prediction, feedback_tokens=feedback_tokens, sample_key=sample_key
        )
