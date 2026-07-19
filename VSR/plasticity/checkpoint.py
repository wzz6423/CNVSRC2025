import os
import random
from pathlib import Path

import torch


CHECKPOINT_VERSION = 2


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
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    state = _extract_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    return model


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps"])


def save_adaptation_checkpoint(
    path,
    expert_bank,
    config,
    processed_samples,
    *,
    optimizer_states=None,
    metrics_state=None,
    rng_state=None,
    stream_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "expert_count": expert_bank.expert_count,
                "expert_bank": expert_bank.state_dict(),
                "expert_summary": expert_bank.summary(),
                "optimizer_states": optimizer_states or {},
                "metrics_state": metrics_state,
                "rng_state": rng_state,
                "stream_state": stream_state,
                "config": config,
                "processed_samples": int(processed_samples),
            },
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_adaptation_checkpoint(path, expert_bank, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expert_bank.ensure_experts(int(checkpoint["expert_count"]))
    expert_bank.load_state_dict(checkpoint["expert_bank"], strict=True)
    return checkpoint
