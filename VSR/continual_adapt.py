import hashlib
import json
import os
import random
import time
import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datamodule.transforms import DICT_PATH, TextTransform
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from plasticity.adapters import ExpertBank
from plasticity.artifacts import (
    append_metrics_history,
    prepare_metrics_history,
    retain_best_checkpoints,
    reset_best_checkpoints,
)
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


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


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


def _stream_state(manifest_path, require_metadata=True):
    manifest_path = Path(manifest_path)
    state = {"manifest_sha256": _file_sha256(manifest_path)}
    metadata_path = Path(f"{manifest_path}.meta.json")
    if not metadata_path.is_file():
        if require_metadata:
            raise ValueError(f"流清单缺少 sidecar：{metadata_path}")
        return state

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"流清单元数据不是有效 JSON：{metadata_path}") from error
    if metadata.get("schema_version") != 1:
        raise ValueError(f"流清单元数据版本不受支持：{metadata_path}")
    label_mode = metadata.get("label_mode")
    if label_mode not in {"metadata_reencoded", "token_passthrough"}:
        raise ValueError(f"流清单元数据缺少有效 label_mode：{metadata_path}")
    source_csv_hash = metadata.get("source_csv_sha256")
    if not isinstance(source_csv_hash, str) or len(source_csv_hash) != 64:
        raise ValueError(f"流清单元数据缺少 source_csv_sha256：{metadata_path}")
    if label_mode == "metadata_reencoded":
        required_metadata = {
            "text_metadata_csv_sha256",
            "oov_policy",
            "raw_characters",
            "target_characters",
            "dropped_characters",
            "dropped_rate",
            "distinct_dropped_characters",
            "dropped_character_counts",
        }
        if not required_metadata.issubset(metadata):
            raise ValueError(f"重编码流清单元数据字段不完整：{metadata_path}")
        metadata_hash = metadata["text_metadata_csv_sha256"]
        if not isinstance(metadata_hash, str) or len(metadata_hash) != 64:
            raise ValueError(f"文本元数据 SHA-256 无效：{metadata_path}")
        if metadata["oov_policy"] not in {"error", "drop"}:
            raise ValueError(f"流清单 OOV 策略无效：{metadata_path}")
        counts = (
            metadata["raw_characters"],
            metadata["target_characters"],
            metadata["dropped_characters"],
            metadata["distinct_dropped_characters"],
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ValueError(f"流清单字符统计无效：{metadata_path}")
        if counts[0] != counts[1] + counts[2]:
            raise ValueError(f"流清单字符统计不守恒：{metadata_path}")
        dropped_counts = metadata["dropped_character_counts"]
        if (
            not isinstance(dropped_counts, dict)
            or len(dropped_counts) != counts[3]
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in dropped_counts.values()
            )
            or sum(dropped_counts.values()) != counts[2]
        ):
            raise ValueError(f"流清单 OOV 统计无效：{metadata_path}")
        dropped_rate = metadata["dropped_rate"]
        expected_rate = counts[2] / counts[0] if counts[0] else 0.0
        if (
            not isinstance(dropped_rate, (int, float))
            or isinstance(dropped_rate, bool)
            or abs(float(dropped_rate) - expected_rate) > 1e-12
        ):
            raise ValueError(f"流清单 OOV 比例无效：{metadata_path}")
    metadata_samples = metadata.get("samples")
    if not isinstance(metadata_samples, int) or isinstance(metadata_samples, bool):
        raise ValueError(f"流清单元数据缺少有效 samples：{metadata_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest_samples = sum(1 for line in handle if line.strip())
    if metadata_samples != manifest_samples:
        raise ValueError("流清单元数据样本数与 JSONL 不一致")
    expected_vocab_hash = metadata.get("target_vocab_sha256")
    if not isinstance(expected_vocab_hash, str):
        raise ValueError(f"流清单元数据缺少 target_vocab_sha256：{metadata_path}")
    actual_vocab_hash = _file_sha256(DICT_PATH)
    if expected_vocab_hash != actual_vocab_hash:
        raise ValueError("流清单目标词表与当前模型词表不一致")
    state.update(
        {
            "manifest_metadata_sha256": _file_sha256(metadata_path),
            "target_vocab_sha256": actual_vocab_hash,
        }
    )
    return state


def _validate_stream_state(saved_state, current_state):
    if saved_state != current_state:
        raise ValueError("恢复 checkpoint 的流式清单与当前清单不一致")


def _experiment_config_sha256(cfg):
    value = {
        "seed": int(cfg.seed),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "decoder": OmegaConf.to_container(cfg.decoder, resolve=True),
        "feedback": OmegaConf.to_container(cfg.feedback, resolve=True),
        "plasticity": OmegaConf.to_container(cfg.plasticity, resolve=True),
    }
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        feedback_correct_span_kl_enabled=(
            cfg.plasticity.feedback_correct_span_kl_enabled
        ),
        feedback_update_strategy=cfg.plasticity.feedback_update_strategy,
        feedback_local_error_target_weight=(
            cfg.plasticity.feedback_local.error_target_weight
        ),
        feedback_local_insertion_blank_weight=(
            cfg.plasticity.feedback_local.insertion_blank_weight
        ),
        feedback_local_matched_kl_weight=(
            cfg.plasticity.feedback_local.matched_kl_weight
        ),
        feedback_random_control_seed=(
            cfg.plasticity.feedback_local.random_control_seed
        ),
        non_feedback_updates_enabled=cfg.plasticity.non_feedback_updates_enabled,
        adaptation_objective=cfg.plasticity.adaptation_objective,
        entropy_frame_selection=cfg.plasticity.tent.frame_selection,
    )


@hydra.main(config_path="conf", config_name="continual_adapt", version_base=None)
def main(cfg: DictConfig):
    if int(cfg.metrics_history_every) < 1:
        raise ValueError("metrics_history_every 必须大于 0")
    if bool(cfg.save_adaptation_checkpoint) and int(cfg.checkpoint_file_limit) < 2:
        raise ValueError("保存 checkpoint 时 checkpoint_file_limit 至少为 2")
    checkpoint_metric = str(cfg.checkpoint_selection_metric)
    if checkpoint_metric not in {"cer", "mean_reliability"}:
        raise ValueError("checkpoint_selection_metric 仅支持 cer 或 mean_reliability")
    checkpoint_mode = str(cfg.checkpoint_selection_mode)
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_selection_mode 仅支持 min 或 max")
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
    metrics_history_path = output_dir / "metrics_history.jsonl"
    checkpoint_path = output_dir / "adaptation_state.pt"
    best_checkpoint_directory = output_dir / "best_checkpoints"
    metrics = StreamMetrics()
    processed = 0
    stream_state = _stream_state(
        cfg.stream_manifest, require_metadata=bool(cfg.require_manifest_metadata)
    )
    stream_state.update(
        {
            "base_checkpoint_sha256": _file_sha256(cfg.checkpoint_path),
            "experiment_config_sha256": _experiment_config_sha256(cfg),
        }
    )
    if cfg.resume_adaptation_checkpoint:
        checkpoint = load_adaptation_checkpoint(
            cfg.resume_adaptation_checkpoint, engine.expert_bank, device
        )
        if int(checkpoint.get("version", 1)) < CHECKPOINT_VERSION:
            raise ValueError("旧版 checkpoint 缺少精确恢复所需的运行状态")
        saved_stream_state = checkpoint.get("stream_state") or {}
        _validate_stream_state(saved_stream_state, stream_state)
        processed = int(checkpoint["processed_samples"])
        if processed and checkpoint.get("metrics_state") is None:
            raise ValueError("checkpoint 缺少累积指标，无法精确恢复")
        engine.load_optimizer_state_dict(checkpoint.get("optimizer_states", {}))
        if checkpoint.get("metrics_state") is not None:
            metrics.load_state_dict(checkpoint["metrics_state"])
        restore_rng_state(checkpoint.get("rng_state"))

    result_mode = _prepare_result_file(result_path, processed)
    if not processed:
        reset_best_checkpoints(best_checkpoint_directory)
    last_history_sample = prepare_metrics_history(metrics_history_path, processed)

    def history_record(checkpoint=False, resumed=False):
        return {
            "processed_samples": processed,
            "checkpoint": bool(checkpoint),
            "resumed": bool(resumed),
            **metrics.summary(),
            "expert_bank": engine.expert_bank.summary(),
        }

    def record_history(checkpoint=False, resumed=False):
        nonlocal last_history_sample
        if processed <= last_history_sample:
            return
        append_metrics_history(
            metrics_history_path,
            history_record(checkpoint=checkpoint, resumed=resumed),
        )
        last_history_sample = processed

    def save_state(retain_best=True):
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
        checkpoint_score = metrics.summary()[checkpoint_metric]
        if (
            retain_best
            and checkpoint_score is not None
        ):
            retain_best_checkpoints(
                checkpoint_path,
                best_checkpoint_directory,
                score=checkpoint_score,
                processed_samples=processed,
                keep=int(cfg.checkpoint_file_limit) - 1,
                metric_name=checkpoint_metric,
                mode=checkpoint_mode,
            )

    if processed:
        record_history(checkpoint=True, resumed=True)
        if bool(cfg.save_adaptation_checkpoint):
            save_state()

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

            sample_started = time.perf_counter()
            video_started = time.perf_counter()
            video = load_stream_video(item)
            video_load_seconds = time.perf_counter() - video_started
            _synchronize_device(device)
            process_started = time.perf_counter()
            prediction = engine.predict(video)
            if use_feedback and not item.target_tokens:
                raise ValueError(
                    f"样本 {item.uid} 被标记为反馈，但没有目标 token"
                )
            feedback_tokens = list(item.target_tokens) if use_feedback else None
            if feedback_tokens and float(cfg.feedback.noise_rate) > 0:
                feedback_tokens = _corrupt_feedback(
                    feedback_tokens,
                    float(cfg.feedback.noise_rate),
                    len(text_transform.token_list),
                )
            outcome = engine.adapt(
                prediction,
                feedback_tokens=feedback_tokens,
                sample_key=item.uid,
            )
            _synchronize_device(device)
            process_seconds = time.perf_counter() - process_started
            total_seconds = time.perf_counter() - sample_started
            runtime = {
                "video_load_seconds": video_load_seconds,
                "process_seconds": process_seconds,
                "total_seconds": total_seconds,
            }
            if device.type == "cuda":
                runtime.update(
                    {
                        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(
                            device
                        ),
                        "gpu_max_memory_allocated_bytes": (
                            torch.cuda.max_memory_allocated(device)
                        ),
                    }
                )
            row = {
                "index": index,
                "uid": item.uid,
                "video": str(item.video_path),
                "domain": item.domain,
                "target": item.target_text,
                "feedback_used": use_feedback,
                "runtime": runtime,
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
                video_load_seconds=video_load_seconds,
                process_seconds=process_seconds,
                total_seconds=total_seconds,
            )
            processed += 1
            is_checkpoint = (
                bool(cfg.save_adaptation_checkpoint)
                and int(cfg.checkpoint_every) > 0
                and processed % int(cfg.checkpoint_every) == 0
            )
            if (
                int(cfg.metrics_history_every) > 0
                and processed % int(cfg.metrics_history_every) == 0
            ):
                record_history(checkpoint=is_checkpoint)
            if is_checkpoint:
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
    summary["adaptation_objective"] = str(cfg.plasticity.adaptation_objective)
    summary["feedback_update_strategy"] = str(
        cfg.plasticity.feedback_update_strategy
    )
    summary["non_feedback_updates_enabled"] = bool(
        cfg.plasticity.non_feedback_updates_enabled
    )
    summary["base_checkpoint"] = str(cfg.checkpoint_path)
    summary["stream_manifest"] = str(cfg.stream_manifest)
    summary["stream_state"] = stream_state
    record_history(checkpoint=bool(cfg.save_adaptation_checkpoint))
    if cfg.save_adaptation_checkpoint:
        save_state()
    best_checkpoint_index = best_checkpoint_directory / "index.json"
    summary["artifacts"] = {
        "stream_results": str(result_path),
        "metrics_history": str(metrics_history_path),
        "latest_checkpoint": str(checkpoint_path)
        if bool(cfg.save_adaptation_checkpoint)
        else None,
        "best_checkpoint_index": str(best_checkpoint_index)
        if best_checkpoint_index.is_file()
        else None,
        "checkpoint_file_limit": int(cfg.checkpoint_file_limit),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
