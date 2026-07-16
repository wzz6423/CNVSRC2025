import math
from dataclasses import asdict, dataclass

import torch


def collapse_ctc_tokens(token_ids, blank_id=0):
    collapsed = []
    previous = None
    for token in token_ids:
        token = int(token)
        if token != previous and token != blank_id:
            collapsed.append(token)
        previous = token
    return collapsed


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def token_similarity(left, right):
    denominator = max(len(left), len(right), 1)
    return max(0.0, 1.0 - edit_distance(left, right) / denominator)


@dataclass(frozen=True)
class ReliabilityDecision:
    accepted: bool
    score: float
    confidence: float
    view_consistency: float
    view_token_agreement: float
    decoder_agreement: float | None
    emission_rate: float
    pseudo_token_count: int
    reasons: tuple[str, ...]

    def to_dict(self):
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


class ReliabilityGate:
    def __init__(
        self,
        blank_id=0,
        min_score=0.72,
        min_confidence=0.55,
        min_view_agreement=0.5,
        min_emission_rate=0.02,
        max_emission_rate=0.95,
        min_pseudo_tokens=1,
        consistency_temperature=0.08,
        confidence_weight=0.3,
        consistency_weight=0.25,
        view_agreement_weight=0.25,
        decoder_agreement_weight=0.2,
    ):
        self.blank_id = int(blank_id)
        self.min_score = float(min_score)
        self.min_confidence = float(min_confidence)
        self.min_view_agreement = float(min_view_agreement)
        self.min_emission_rate = float(min_emission_rate)
        self.max_emission_rate = float(max_emission_rate)
        self.min_pseudo_tokens = int(min_pseudo_tokens)
        self.consistency_temperature = float(consistency_temperature)
        if self.consistency_temperature <= 0:
            raise ValueError("一致性温度必须大于 0")
        self.weights = {
            "confidence": float(confidence_weight),
            "view_consistency": float(consistency_weight),
            "view_token_agreement": float(view_agreement_weight),
            "decoder_agreement": float(decoder_agreement_weight),
        }
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("可靠性信号权重不能为负数")

    def evaluate(self, clean_log_probs, augmented_log_probs, decoder_tokens=None):
        if clean_log_probs.ndim != 3 or augmented_log_probs.ndim != 3:
            raise ValueError("CTC log-prob 必须为 [B, T, V]")
        if clean_log_probs.size(0) != 1 or augmented_log_probs.size(0) != 1:
            raise ValueError("可靠性门控当前只支持 batch size 1")
        if clean_log_probs.size(-1) != augmented_log_probs.size(-1):
            raise ValueError("两个视图的 CTC 词表维度必须一致")
        length = min(clean_log_probs.size(1), augmented_log_probs.size(1))
        if length == 0:
            raise ValueError("CTC 时间维不能为空")
        clean_log_probs = clean_log_probs[:, :length]
        augmented_log_probs = augmented_log_probs[:, :length]

        if not torch.isfinite(clean_log_probs).all() or not torch.isfinite(
            augmented_log_probs
        ).all():
            return [], ReliabilityDecision(
                accepted=False,
                score=0.0,
                confidence=0.0,
                view_consistency=0.0,
                view_token_agreement=0.0,
                decoder_agreement=0.0 if decoder_tokens is not None else None,
                emission_rate=0.0,
                pseudo_token_count=0,
                reasons=("non_finite_probability",),
            )

        reasons = []
        clean_probs = clean_log_probs.exp()
        augmented_probs = augmented_log_probs.exp()
        clean_confidence, clean_ids = clean_probs.max(dim=-1)
        _, augmented_ids = augmented_probs.max(dim=-1)
        emitted = clean_ids.ne(self.blank_id)
        union_emitted = emitted | augmented_ids.ne(self.blank_id)
        emission_rate = float(emitted.float().mean().item())
        confidence = (
            float(clean_confidence[emitted].mean().item()) if emitted.any() else 0.0
        )

        clean_tokens = collapse_ctc_tokens(clean_ids[0].tolist(), self.blank_id)
        augmented_tokens = collapse_ctc_tokens(
            augmented_ids[0].tolist(), self.blank_id
        )
        view_token_agreement = token_similarity(clean_tokens, augmented_tokens)

        midpoint = (clean_probs + augmented_probs).mul(0.5).clamp_min(1e-8)
        clean_kl = clean_probs * (clean_log_probs - midpoint.log())
        augmented_kl = augmented_probs * (augmented_log_probs - midpoint.log())
        js_per_frame = 0.5 * (clean_kl.sum(-1) + augmented_kl.sum(-1))
        selected = js_per_frame[union_emitted] if union_emitted.any() else js_per_frame
        js_divergence = float(selected.mean().item())
        view_consistency = math.exp(
            -js_divergence / self.consistency_temperature
        )

        decoder_agreement = None
        signals = {
            "confidence": confidence,
            "view_consistency": view_consistency,
            "view_token_agreement": view_token_agreement,
        }
        if decoder_tokens is not None:
            decoder_agreement = token_similarity(clean_tokens, decoder_tokens)
            signals["decoder_agreement"] = decoder_agreement
        total_weight = sum(self.weights[name] for name in signals)
        if total_weight <= 0:
            raise ValueError("至少需要一个正的可靠性信号权重")
        score = sum(self.weights[name] * value for name, value in signals.items())
        score = score / max(total_weight, 1e-8)

        if confidence < self.min_confidence:
            reasons.append("low_confidence")
        if view_token_agreement < self.min_view_agreement:
            reasons.append("view_disagreement")
        if not self.min_emission_rate <= emission_rate <= self.max_emission_rate:
            reasons.append("invalid_emission_rate")
        if len(clean_tokens) < self.min_pseudo_tokens:
            reasons.append("empty_pseudo_label")
        if score < self.min_score:
            reasons.append("low_reliability_score")

        return clean_tokens, ReliabilityDecision(
            accepted=not reasons,
            score=float(score),
            confidence=confidence,
            view_consistency=view_consistency,
            view_token_agreement=view_token_agreement,
            decoder_agreement=decoder_agreement,
            emission_rate=emission_rate,
            pseudo_token_count=len(clean_tokens),
            reasons=tuple(reasons),
        )
