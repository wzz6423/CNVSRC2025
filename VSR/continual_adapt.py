import hashlib
import json
import os
import random
import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datamodule.transforms import TextTransform
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from plasticity.adapters import ExpertBank
from plasticity.checkpoint import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    load_adaptation_checkpoint,
    load_base_checkpoint,
    restore_rng_state,
    save_adaptation_checkpoint,
)
from plasticity.decoding import BeamDecoder
from plasticity.engine import ContinualAdaptationEngine
from plasticity.metrics import StreamMetrics
from plasticity.reliability import ReliabilityGate
from plasticity.stream import iter_stream_manifest, load_stream_video


def _corrupt_feedback(tokens, probability, vocabulary_size):
    if not 0.0 <= probability <= 1.0:
        raise ValueError("反馈污染率必须介于 0 和 1 之间")
    candidate_count = vocabulary_size - 2
    if candidate_count < 2:
        raise ValueError("词表太小，无法构造非空白的错误反馈")
    corrupted = []
    for token in tokens:
        if random.random() < probability:
            replacement = random.randint(1, candidate_count - 1)
            if 1 <= int(token) <= candidate_count and replacement >= int(token):
                replacement += 1
            corrupted.append(replacement)
        else:
            corrupted.append(int(token))
    return corrupted


def _resolve_device(configured_device):
    value = str(configured_device).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            if _mps_supports_conv3d():
                return torch.device("mps")
            warnings.warn(
                "当前 PyTorch 的 MPS 不支持 Conv3D，自动回退到 CPU",
                stacklevel=2,
            )
        return torch.device("cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"当前机器不支持请求的 CUDA 设备：{device}")
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"当前机器不支持请求的 MPS 设备：{device}")
        if not _mps_supports_conv3d():
            raise RuntimeError(
                "当前 PyTorch 的 MPS 不支持 Conv3D；"
                "请使用 device=cpu 或升级 PyTorch"
            )
    return device


def _mps_supports_conv3d():
    if not torch.backends.mps.is_available():
        return False
    try:
        value = torch.zeros((1, 1, 1, 1, 1), device="mps")
        torch.nn.functional.conv3d(value, value)
        return True
    except RuntimeError:
        return False


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_result_file(path, processed_samples):
    path = Path(path)
    if processed_samples == 0:
        return "w"
    if not path.is_file():
        raise FileNotFoundError(f"恢复适应时找不到原结果文件：{path}")

    with path.open(encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    if line_count < processed_samples:
        raise ValueError(
            f"结果仅有 {line_count} 行，少于 checkpoint 的 "
            f"{processed_samples} 条"
        )
    if line_count > processed_samples:
        temporary_path = path.with_name(f".{path.name}.resume.tmp")
        try:
            with path.open(encoding="utf-8") as source, temporary_path.open(
                "w", encoding="utf-8"
            ) as target:
                for index, line in enumerate(source):
                    if index >= processed_samples:
                        break
                    target.write(line)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return "a"


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_engine(cfg, model, token_list, device):
    mode = str(cfg.plasticity.mode)
    if mode not in {"static", "single_adapter", "expert_bank"}:
        raise ValueError(f"未知持续适应模式：{mode}")
    allow_growth = mode == "expert_bank" and bool(cfg.plasticity.allow_growth)
    max_experts = int(cfg.plasticity.max_experts) if allow_growth else 1
    bank = ExpertBank(
        feature_dim=cfg.model.visual_backbone.adim,
        bottleneck_dim=cfg.plasticity.adapter_bottleneck,
        max_experts=max_experts,
        route_threshold=cfg.plasticity.route_threshold,
        growth_patience=cfg.plasticity.growth_patience,
        pending_similarity=cfg.plasticity.pending_similarity,
        prototype_momentum=cfg.plasticity.prototype_momentum,
        signature_motion_order=cfg.plasticity.signature_motion_order,
        dropout=cfg.plasticity.adapter_dropout,
        allow_growth=allow_growth,
    )
    reliability = ReliabilityGate(
        **OmegaConf.to_container(cfg.plasticity.reliability, resolve=True)
    )
    decoder = BeamDecoder(
        model,
        token_list,
        beam_size=cfg.decoder.beam_size,
        ctc_weight=cfg.decoder.ctc_weight,
    )
    return ContinualAdaptationEngine(
        base_model=model,
        expert_bank=bank,
        reliability_gate=reliability,
        decoder=decoder,
        device=device,
        enabled=mode != "static",
        reliability_enabled=cfg.plasticity.reliability_enabled,
        rollback_enabled=cfg.plasticity.rollback_enabled,
        learning_rate=cfg.plasticity.learning_rate,
        weight_decay=cfg.plasticity.weight_decay,
        adaptation_steps=cfg.plasticity.adaptation_steps,
        gradient_clip=cfg.plasticity.gradient_clip,
        pseudo_ctc_weight=cfg.plasticity.loss.pseudo_ctc,
        consistency_weight=cfg.plasticity.loss.consistency,
        entropy_weight=cfg.plasticity.loss.entropy,
        feature_anchor_weight=cfg.plasticity.loss.feature_anchor,
        max_anchor_kl=cfg.plasticity.rollback.max_anchor_kl,
        max_reliability_drop=cfg.plasticity.rollback.max_reliability_drop,
        max_target_loss_increase=cfg.plasticity.rollback.max_target_loss_increase,
        view_noise_std=cfg.plasticity.view.noise_std,
        temporal_mask_ratio=cfg.plasticity.view.temporal_mask_ratio,
        feedback_confirms_growth=cfg.plasticity.feedback_confirms_growth,
    )


@hydra.main(config_path="conf", config_name="continual_adapt", version_base=None)
def main(cfg: DictConfig):
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = _resolve_device(cfg.device)
    print(f"持续适应设备：{device}")
    text_transform = TextTransform()
    model = E2E(len(text_transform.token_list), cfg.model.visual_backbone)
    load_base_checkpoint(model, cfg.checkpoint_path, device)
    engine = _build_engine(cfg, model, text_transform.token_list, device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "stream_results.jsonl"
    checkpoint_path = output_dir / "adaptation_state.pt"
    metrics = StreamMetrics()
    processed = 0
    stream_state = {
        "manifest_sha256": _file_sha256(cfg.stream_manifest),
    }
    if cfg.resume_adaptation_checkpoint:
        checkpoint = load_adaptation_checkpoint(
            cfg.resume_adaptation_checkpoint, engine.expert_bank, device
        )
        if int(checkpoint.get("version", 1)) < CHECKPOINT_VERSION:
            raise ValueError("旧版 checkpoint 缺少精确恢复所需的运行状态")
        saved_stream_state = checkpoint.get("stream_state") or {}
        if saved_stream_state.get("manifest_sha256") != stream_state["manifest_sha256"]:
            raise ValueError("恢复 checkpoint 的流式清单与当前清单不一致")
        processed = int(checkpoint["processed_samples"])
        if processed and checkpoint.get("metrics_state") is None:
            raise ValueError("checkpoint 缺少累积指标，无法精确恢复")
        engine.load_optimizer_state_dict(checkpoint.get("optimizer_states", {}))
        if checkpoint.get("metrics_state") is not None:
            metrics.load_state_dict(checkpoint["metrics_state"])
        restore_rng_state(checkpoint.get("rng_state"))

    result_mode = _prepare_result_file(result_path, processed)

    def save_state():
        save_adaptation_checkpoint(
            checkpoint_path,
            engine.expert_bank,
            OmegaConf.to_container(cfg.plasticity, resolve=True),
            processed,
            optimizer_states=engine.optimizer_state_dict(),
            metrics_state=metrics.state_dict(),
            rng_state=capture_rng_state(),
            stream_state=stream_state,
        )

    stream = iter_stream_manifest(
        cfg.stream_manifest, cfg.data_root_dir, text_transform
    )
    with result_path.open(result_mode, encoding="utf-8") as result_file:
        for index, item in enumerate(stream):
            if cfg.max_samples is not None and index >= int(cfg.max_samples):
                break
            if index < processed:
                continue
            use_feedback = item.feedback or (
                int(cfg.feedback.every) > 0
                and (index + 1) % int(cfg.feedback.every) == 0
            )
            if use_feedback and not item.target_tokens:
                raise ValueError(f"样本 {item.uid} 被标记为反馈，但没有目标 token")
            feedback_tokens = list(item.target_tokens) if use_feedback else None
            if feedback_tokens and float(cfg.feedback.noise_rate) > 0:
                feedback_tokens = _corrupt_feedback(
                    feedback_tokens,
                    float(cfg.feedback.noise_rate),
                    len(text_transform.token_list),
                )

            outcome = engine.process(
                load_stream_video(item), feedback_tokens=feedback_tokens
            )
            row = {
                "index": index,
                "uid": item.uid,
                "video": str(item.video_path),
                "domain": item.domain,
                "target": item.target_text,
                "feedback_used": use_feedback,
                **outcome.to_dict(),
            }
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_file.flush()
            metrics.update(
                outcome.transcript,
                item.target_text,
                item.domain,
                outcome.reliability.score,
                outcome.update.status,
            )
            processed += 1
            if (
                bool(cfg.save_adaptation_checkpoint)
                and int(cfg.checkpoint_every) > 0
                and processed % int(cfg.checkpoint_every) == 0
            ):
                result_file.flush()
                os.fsync(result_file.fileno())
                save_state()
            if processed % int(cfg.log_every) == 0:
                current = metrics.summary()
                print(
                    f"已处理 {processed} 条，CER={current['cer']}，"
                    f"专家数={engine.expert_bank.expert_count}"
                )
        result_file.flush()
        if bool(cfg.save_adaptation_checkpoint):
            os.fsync(result_file.fileno())

    summary = metrics.summary()
    summary["expert_bank"] = engine.expert_bank.summary()
    summary["mode"] = str(cfg.plasticity.mode)
    summary["base_checkpoint"] = str(cfg.checkpoint_path)
    summary["stream_manifest"] = str(cfg.stream_manifest)
    if cfg.save_adaptation_checkpoint:
        save_state()
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
