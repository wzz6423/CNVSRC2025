from pathlib import Path

import torch


def _extract_model_state(checkpoint):
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        return {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_base_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到基础模型权重：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = _extract_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    return model


def save_adaptation_checkpoint(path, expert_bank, config, processed_samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert_count": expert_bank.expert_count,
            "expert_bank": expert_bank.state_dict(),
            "expert_summary": expert_bank.summary(),
            "config": config,
            "processed_samples": int(processed_samples),
        },
        path,
    )


def load_adaptation_checkpoint(path, expert_bank, device):
    checkpoint = torch.load(path, map_location=device)
    expert_bank.ensure_experts(int(checkpoint["expert_count"]))
    expert_bank.load_state_dict(checkpoint["expert_bank"], strict=True)
    return checkpoint
