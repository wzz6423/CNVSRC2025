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
