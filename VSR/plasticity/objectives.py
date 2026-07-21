import torch
import torch.nn.functional as F


def ctc_target_error(log_probs, target_tokens, blank_id=0):
    if log_probs.ndim != 3:
        raise ValueError("CTC log-prob 必须为 [B, T, V]")
    if log_probs.size(0) != 1:
        raise ValueError("在线更新当前只支持 batch size 1")
    target = torch.as_tensor(
        target_tokens, dtype=torch.long, device=log_probs.device
    ).flatten()
    if target.numel() == 0:
        return "empty_target"
    if target.lt(0).any() or target.ge(log_probs.size(-1)).any():
        return "target_token_out_of_range"
    if target.eq(int(blank_id)).any():
        return "target_contains_ctc_blank"
    repeated = target[1:].eq(target[:-1]).sum().item()
    minimum_frames = target.numel() + int(repeated)
    if minimum_frames > log_probs.size(1):
        return "insufficient_ctc_frames"
    return None


def ctc_sequence_loss(log_probs, target_tokens, blank_id=0):
    target_error = ctc_target_error(log_probs, target_tokens, blank_id)
    if target_error is not None:
        raise ValueError(f"无效 CTC 目标：{target_error}")
    loss_device = (
        torch.device("cpu") if log_probs.device.type == "mps" else log_probs.device
    )
    target = torch.as_tensor(
        target_tokens, dtype=torch.long, device=loss_device
    ).flatten()
    input_lengths = torch.tensor(
        [log_probs.size(1)], dtype=torch.long, device=loss_device
    )
    target_lengths = torch.tensor(
        [target.numel()], dtype=torch.long, device=loss_device
    )
    loss = F.ctc_loss(
        log_probs.to(loss_device).transpose(0, 1),
        target,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return loss.to(log_probs.device)


def posterior_kl(teacher_log_probs, student_log_probs, frame_mask=None):
    length = min(teacher_log_probs.size(1), student_log_probs.size(1))
    teacher = teacher_log_probs[:, :length].detach()
    student = student_log_probs[:, :length]
    per_frame = (teacher.exp() * (teacher - student)).sum(dim=-1)
    if frame_mask is None:
        return per_frame.mean()
    if frame_mask.ndim != 2 or frame_mask.size(0) != per_frame.size(0):
        raise ValueError("KL 帧掩码必须为 [B, T]")
    selected = frame_mask[:, :length].to(device=per_frame.device, dtype=torch.bool)
    if not selected.any():
        return per_frame.sum() * 0.0
    return per_frame.masked_select(selected).mean()


def occupancy_weighted_posterior_kl(
    teacher_log_probs, student_log_probs, frame_weights
):
    length = min(teacher_log_probs.size(1), student_log_probs.size(1))
    teacher = teacher_log_probs[:, :length].detach()
    student = student_log_probs[:, :length]
    if frame_weights.ndim != 2 or frame_weights.size(0) != teacher.size(0):
        raise ValueError("KL occupancy 权重必须为 [B, T]")
    weights = frame_weights[:, :length].to(
        device=student.device, dtype=student.dtype
    ).detach()
    if not torch.isfinite(weights).all() or weights.lt(0).any():
        raise ValueError("KL occupancy 权重必须为非负有限值")
    denominator = weights.sum()
    if denominator.item() == 0:
        return student.sum() * 0.0
    per_frame = (teacher.exp() * (teacher - student)).sum(dim=-1)
    return (per_frame * weights).sum() / denominator


def occupancy_weighted_token_loss(
    log_probs, target_tokens, token_occupancy, target_indices
):
    if log_probs.ndim != 3 or log_probs.size(0) != 1:
        raise ValueError("局部 token loss 当前只支持 [1, T, V]")
    if token_occupancy.ndim != 3 or token_occupancy.size(0) != 1:
        raise ValueError("target occupancy 必须为 [1, T, L]")
    if token_occupancy.size(1) != log_probs.size(1):
        raise ValueError("target occupancy 与 log-prob 的时间长度不一致")
    target = torch.as_tensor(
        target_tokens, dtype=torch.long, device=log_probs.device
    ).flatten()
    if token_occupancy.size(2) != target.numel():
        raise ValueError("target occupancy 的 token 维与 target 长度不一致")
    indices = torch.as_tensor(
        target_indices, dtype=torch.long, device=log_probs.device
    ).flatten()
    if indices.numel() == 0:
        return log_probs.sum() * 0.0
    if indices.lt(0).any() or indices.ge(target.numel()).any():
        raise ValueError("局部 token 索引越界")
    selected_tokens = target.index_select(0, indices)
    selected_log_probs = log_probs.index_select(-1, selected_tokens)
    weights = token_occupancy.to(
        device=log_probs.device, dtype=log_probs.dtype
    ).detach().index_select(-1, indices)
    if not torch.isfinite(weights).all() or weights.lt(0).any():
        raise ValueError("target occupancy 必须为非负有限值")
    denominator = weights.sum()
    if denominator.item() == 0:
        return log_probs.sum() * 0.0
    return -(selected_log_probs * weights).sum() / denominator


def occupancy_weighted_blank_loss(log_probs, frame_weights, blank_id=0):
    if log_probs.ndim != 3 or log_probs.size(0) != 1:
        raise ValueError("局部 blank loss 当前只支持 [1, T, V]")
    if frame_weights.ndim != 2 or frame_weights.shape != log_probs.shape[:2]:
        raise ValueError("blank 帧权重必须与 log-prob 的 [B, T] 一致")
    blank_id = int(blank_id)
    if not 0 <= blank_id < log_probs.size(-1):
        raise ValueError("blank_id 越界")
    weights = frame_weights.to(
        device=log_probs.device, dtype=log_probs.dtype
    ).detach()
    if not torch.isfinite(weights).all() or weights.lt(0).any():
        raise ValueError("blank 帧权重必须为非负有限值")
    denominator = weights.sum()
    if denominator.item() == 0:
        return log_probs.sum() * 0.0
    return -(log_probs[:, :, blank_id] * weights).sum() / denominator


def ctc_posterior_entropy(log_probs, blank_id=0, frame_selection="all"):
    if log_probs.ndim != 3:
        raise ValueError("CTC entropy 输入必须为 [B, T, V]")
    if frame_selection not in {"all", "nonblank"}:
        raise ValueError("CTC entropy 帧选择仅支持 all 或 nonblank")
    probabilities = log_probs.exp()
    per_frame = -(probabilities * log_probs).sum(dim=-1)
    if frame_selection == "all":
        return per_frame.mean()
    selected = log_probs.detach().argmax(dim=-1).ne(int(blank_id))
    if not selected.any():
        return per_frame.sum() * 0.0
    return per_frame.masked_select(selected).mean()


def posterior_entropy(log_probs):
    return ctc_posterior_entropy(log_probs)


def feature_anchor_loss(adapted_features, frozen_features):
    return F.mse_loss(adapted_features, frozen_features.detach())
