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


def posterior_kl(teacher_log_probs, student_log_probs):
    length = min(teacher_log_probs.size(1), student_log_probs.size(1))
    teacher = teacher_log_probs[:, :length].detach()
    student = student_log_probs[:, :length]
    return (teacher.exp() * (teacher - student)).sum(dim=-1).mean()


def posterior_entropy(log_probs):
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum(dim=-1).mean()


def feature_anchor_loss(adapted_features, frozen_features):
    return F.mse_loss(adapted_features, frozen_features.detach())
