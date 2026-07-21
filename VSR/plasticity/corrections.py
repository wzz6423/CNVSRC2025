import hashlib
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class CorrectionDiagnostics:
    predicted_tokens: int
    target_tokens: int
    matched_tokens: int
    substituted_tokens: int
    missing_target_tokens: int
    extra_prediction_tokens: int
    token_error_rate: float
    matched_frame_rate: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TargetConditionedOccupancy:
    """Target-conditioned CTC posterior occupancy localization result.

    Tensor fields intentionally stay off CorrectionDiagnostics.to_dict so the
    legacy greedy correct-span JSON schema remains unchanged.
    """

    matched_target_indices: tuple[int, ...]
    substitution_target_indices: tuple[int, ...]
    deletion_target_indices: tuple[int, ...]
    extra_prediction_frame_mask: torch.Tensor
    matched_target_occupancy: torch.Tensor
    error_target_occupancy: torch.Tensor
    token_occupancy: torch.Tensor
    log_likelihood: torch.Tensor


def _token_alignment(predicted_tokens, target_tokens):
    predicted = tuple(int(token) for token in predicted_tokens)
    target = tuple(int(token) for token in target_tokens)
    rows = len(predicted) + 1
    columns = len(target) + 1
    costs = [[0] * columns for _ in range(rows)]
    backtrace = [[None] * columns for _ in range(rows)]

    for predicted_index in range(1, rows):
        costs[predicted_index][0] = predicted_index
        backtrace[predicted_index][0] = "extra_prediction"
    for target_index in range(1, columns):
        costs[0][target_index] = target_index
        backtrace[0][target_index] = "missing_target"

    for predicted_index in range(1, rows):
        for target_index in range(1, columns):
            diagonal = (
                "match"
                if predicted[predicted_index - 1] == target[target_index - 1]
                else "substitution"
            )
            candidates = (
                (
                    costs[predicted_index - 1][target_index - 1]
                    + (diagonal != "match"),
                    0,
                    diagonal,
                ),
                (
                    costs[predicted_index - 1][target_index] + 1,
                    1,
                    "extra_prediction",
                ),
                (
                    costs[predicted_index][target_index - 1] + 1,
                    2,
                    "missing_target",
                ),
            )
            cost, _, operation = min(candidates)
            costs[predicted_index][target_index] = cost
            backtrace[predicted_index][target_index] = operation

    operations = []
    predicted_index = len(predicted)
    target_index = len(target)
    while predicted_index or target_index:
        operation = backtrace[predicted_index][target_index]
        if operation in {"match", "substitution"}:
            operations.append(
                (operation, predicted_index - 1, target_index - 1)
            )
            predicted_index -= 1
            target_index -= 1
        elif operation == "extra_prediction":
            operations.append((operation, predicted_index - 1, None))
            predicted_index -= 1
        elif operation == "missing_target":
            operations.append((operation, None, target_index - 1))
            target_index -= 1
        else:
            raise RuntimeError("token 编辑对齐回溯失败")
    operations.reverse()
    return operations


def _ctc_token_segments(frame_tokens, blank_id):
    collapsed = []
    segments = []
    previous = None
    for frame_index, raw_token in enumerate(frame_tokens):
        token = int(raw_token)
        if token != previous and token != blank_id:
            collapsed.append(token)
            segments.append([frame_index])
        elif token == previous and token != blank_id:
            segments[-1].append(frame_index)
        previous = token
    return collapsed, segments


def _validate_ctc_occupancy_inputs(log_probs, target_tokens, blank_id):
    if not isinstance(log_probs, torch.Tensor):
        raise ValueError("log_probs 必须为 torch.Tensor")
    if log_probs.ndim != 3 or log_probs.size(0) != 1:
        raise ValueError("CTC occupancy 当前只支持 [1, T, V] 的 log-prob")
    if not log_probs.is_floating_point():
        raise ValueError("CTC occupancy 的 log_probs 必须为浮点张量")
    if log_probs.size(1) == 0:
        raise ValueError("CTC 时间维不能为空")
    if not torch.isfinite(log_probs).all():
        raise ValueError("log_probs 必须全部有限")

    blank = int(blank_id)
    vocabulary_size = int(log_probs.size(-1))
    if blank < 0 or blank >= vocabulary_size:
        raise ValueError("blank_id 越界")

    target = tuple(int(token) for token in target_tokens)
    if not target:
        raise ValueError("target 不能为空")
    for token in target:
        if token == blank:
            raise ValueError("target 不能包含 CTC blank")
        if token < 0 or token >= vocabulary_size:
            raise ValueError("target token 越界")

    repeated = sum(
        1 for index in range(1, len(target)) if target[index] == target[index - 1]
    )
    minimum_frames = len(target) + repeated
    if log_probs.size(1) < minimum_frames:
        raise ValueError(
            "帧数不足，无法覆盖 target 的 CTC 路径"
            f"（需要至少 {minimum_frames} 帧，实际 {log_probs.size(1)}）"
        )
    return target, blank


def ctc_forward_backward_occupancy(log_probs, target_tokens, blank_id=0):
    """Target-conditioned CTC forward-backward token occupancy.

    Returns posterior occupancy of each non-blank target token state over time,
    shape [1, T, L], plus the path log-likelihood. All computations use
    detached log_probs; the returned tensors do not provide parameter gradients.
    """
    target, blank = _validate_ctc_occupancy_inputs(log_probs, target_tokens, blank_id)
    lp = log_probs.detach()
    device = lp.device
    dtype = lp.dtype
    time_steps = int(lp.size(1))
    target_length = len(target)
    state_count = 2 * target_length + 1

    labels = []
    for state_index in range(state_count):
        if state_index % 2 == 0:
            labels.append(blank)
        else:
            labels.append(target[state_index // 2])
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    emissions = lp[0].index_select(dim=-1, index=labels)

    neg_inf = float("-inf")
    alpha = torch.full((time_steps, state_count), neg_inf, device=device, dtype=dtype)
    alpha[0, 0] = emissions[0, 0]
    if state_count > 1:
        alpha[0, 1] = emissions[0, 1]

    for time_index in range(1, time_steps):
        previous = alpha[time_index - 1]
        stay = previous
        advance = torch.full_like(previous, neg_inf)
        advance[1:] = previous[:-1]
        skip = torch.full_like(previous, neg_inf)
        if state_count > 2:
            # Skip blank only into a non-blank state, and never across repeated tokens.
            for state_index in range(2, state_count):
                if labels[state_index].item() == blank:
                    continue
                if labels[state_index].item() == labels[state_index - 2].item():
                    continue
                skip[state_index] = previous[state_index - 2]
        incoming = torch.logaddexp(torch.logaddexp(stay, advance), skip)
        alpha[time_index] = incoming + emissions[time_index]

    final_candidates = [alpha[time_steps - 1, state_count - 1]]
    if state_count > 1:
        final_candidates.append(alpha[time_steps - 1, state_count - 2])
    log_likelihood = torch.logsumexp(torch.stack(final_candidates), dim=0)
    if not torch.isfinite(log_likelihood):
        raise ValueError("无可行 CTC 路径")

    beta = torch.full((time_steps, state_count), neg_inf, device=device, dtype=dtype)
    beta[time_steps - 1, state_count - 1] = 0
    if state_count > 1:
        beta[time_steps - 1, state_count - 2] = 0

    for time_index in range(time_steps - 2, -1, -1):
        next_emissions = emissions[time_index + 1]
        next_beta = beta[time_index + 1]
        stay = next_beta + next_emissions
        advance = torch.full_like(stay, neg_inf)
        advance[:-1] = next_beta[1:] + next_emissions[1:]
        skip = torch.full_like(stay, neg_inf)
        if state_count > 2:
            for state_index in range(state_count - 2):
                destination = state_index + 2
                if labels[destination].item() == blank:
                    continue
                if labels[destination].item() == labels[state_index].item():
                    continue
                skip[state_index] = next_beta[destination] + next_emissions[destination]
        beta[time_index] = torch.logaddexp(torch.logaddexp(stay, advance), skip)

    log_state_occupancy = alpha + beta - log_likelihood
    token_log_occupancy = log_state_occupancy[:, 1::2]
    occupancy = token_log_occupancy.exp().unsqueeze(0)
    return occupancy, log_likelihood


def _sum_token_occupancy(token_occupancy, indices):
    if not indices:
        return torch.zeros(
            token_occupancy.size(0),
            token_occupancy.size(1),
            dtype=token_occupancy.dtype,
            device=token_occupancy.device,
        )
    index_tensor = torch.tensor(
        indices, dtype=torch.long, device=token_occupancy.device
    )
    return token_occupancy.index_select(dim=-1, index=index_tensor).sum(dim=-1)


def localize_feedback_correction(log_probs, target_tokens, blank_id=0):
    if log_probs.ndim != 3 or log_probs.size(0) != 1:
        raise ValueError("纠错定位当前只支持 [1, T, V] 的 CTC log-prob")
    if log_probs.size(1) == 0:
        raise ValueError("纠错定位的 CTC 时间维不能为空")

    target = tuple(int(token) for token in target_tokens)
    frame_tokens = log_probs.detach().argmax(dim=-1)[0].tolist()
    predicted, segments = _ctc_token_segments(frame_tokens, int(blank_id))
    operations = _token_alignment(predicted, target)
    correct_frame_mask = torch.zeros(
        (1, log_probs.size(1)), dtype=torch.bool, device=log_probs.device
    )
    counts = {
        "match": 0,
        "substitution": 0,
        "missing_target": 0,
        "extra_prediction": 0,
    }
    for operation, predicted_index, _ in operations:
        counts[operation] += 1
        if operation == "match":
            correct_frame_mask[0, segments[predicted_index]] = True

    error_count = (
        counts["substitution"]
        + counts["missing_target"]
        + counts["extra_prediction"]
    )
    diagnostics = CorrectionDiagnostics(
        predicted_tokens=len(predicted),
        target_tokens=len(target),
        matched_tokens=counts["match"],
        substituted_tokens=counts["substitution"],
        missing_target_tokens=counts["missing_target"],
        extra_prediction_tokens=counts["extra_prediction"],
        token_error_rate=error_count / max(len(target), 1),
        matched_frame_rate=float(correct_frame_mask.float().mean().item()),
    )
    return diagnostics, correct_frame_mask


def localize_target_conditioned_occupancy(log_probs, target_tokens, blank_id=0):
    """Map greedy edit ops onto target-conditioned CTC posterior occupancy.

    Uses the same greedy hypothesis collapse and token edit alignment as
    localize_feedback_correction, but attributes matched / error regions via
    stop-gradient target-conditioned CTC occupancy rather than greedy spans.
    """
    occupancy, log_likelihood = ctc_forward_backward_occupancy(
        log_probs, target_tokens, blank_id=blank_id
    )
    target = tuple(int(token) for token in target_tokens)
    frame_tokens = log_probs.detach().argmax(dim=-1)[0].tolist()
    predicted, segments = _ctc_token_segments(frame_tokens, int(blank_id))
    operations = _token_alignment(predicted, target)

    matched_target_indices = []
    substitution_target_indices = []
    deletion_target_indices = []
    extra_prediction_frame_mask = torch.zeros(
        (1, log_probs.size(1)), dtype=torch.bool, device=log_probs.device
    )
    for operation, predicted_index, target_index in operations:
        if operation == "match":
            matched_target_indices.append(int(target_index))
        elif operation == "substitution":
            substitution_target_indices.append(int(target_index))
        elif operation == "missing_target":
            deletion_target_indices.append(int(target_index))
        elif operation == "extra_prediction":
            extra_prediction_frame_mask[0, segments[predicted_index]] = True
        else:
            raise RuntimeError(f"未知编辑操作：{operation}")

    matched_target_indices = tuple(matched_target_indices)
    substitution_target_indices = tuple(substitution_target_indices)
    deletion_target_indices = tuple(deletion_target_indices)
    error_target_indices = substitution_target_indices + deletion_target_indices

    return TargetConditionedOccupancy(
        matched_target_indices=matched_target_indices,
        substitution_target_indices=substitution_target_indices,
        deletion_target_indices=deletion_target_indices,
        extra_prediction_frame_mask=extra_prediction_frame_mask,
        matched_target_occupancy=_sum_token_occupancy(
            occupancy, matched_target_indices
        ),
        error_target_occupancy=_sum_token_occupancy(occupancy, error_target_indices),
        token_occupancy=occupancy,
        log_likelihood=log_likelihood,
    )


def randomized_error_support(localization, sample_key, seed=0):
    if not isinstance(localization, TargetConditionedOccupancy):
        raise TypeError("localization 必须为 TargetConditionedOccupancy")
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("random control 需要非空 sample_key")
    time_steps = localization.token_occupancy.size(1)
    digest = hashlib.sha256(
        f"{int(seed)}\0{sample_key}".encode("utf-8")
    ).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    permutation = torch.randperm(time_steps, generator=generator).to(
        localization.token_occupancy.device
    )
    return (
        localization.token_occupancy.index_select(1, permutation),
        localization.extra_prediction_frame_mask.index_select(1, permutation),
    )
