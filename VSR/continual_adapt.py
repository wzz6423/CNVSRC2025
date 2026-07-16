import json
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datamodule.transforms import TextTransform
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from plasticity.adapters import ExpertBank
from plasticity.checkpoint import (
    load_adaptation_checkpoint,
    load_base_checkpoint,
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


def _build_engine(cfg, model, token_list):
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
        device=cfg.device,
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

    text_transform = TextTransform()
    model = E2E(len(text_transform.token_list), cfg.model.visual_backbone)
    load_base_checkpoint(model, cfg.checkpoint_path, cfg.device)
    engine = _build_engine(cfg, model, text_transform.token_list)
    if cfg.resume_adaptation_checkpoint:
        load_adaptation_checkpoint(
            cfg.resume_adaptation_checkpoint, engine.expert_bank, cfg.device
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "stream_results.jsonl"
    metrics = StreamMetrics()
    processed = 0

    stream = iter_stream_manifest(
        cfg.stream_manifest, cfg.data_root_dir, text_transform
    )
    with result_path.open("w", encoding="utf-8") as result_file:
        for index, item in enumerate(stream):
            if cfg.max_samples is not None and index >= int(cfg.max_samples):
                break
            use_feedback = item.feedback or (
                int(cfg.feedback.every) > 0
                and (index + 1) % int(cfg.feedback.every) == 0
            )
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
            metrics.update(
                outcome.transcript,
                item.target_text,
                item.domain,
                outcome.reliability.score,
                outcome.update.status,
            )
            processed += 1
            if processed % int(cfg.log_every) == 0:
                current = metrics.summary()
                print(
                    f"已处理 {processed} 条，CER={current['cer']}，"
                    f"专家数={engine.expert_bank.expert_count}"
                )

    summary = metrics.summary()
    summary["expert_bank"] = engine.expert_bank.summary()
    summary["mode"] = str(cfg.plasticity.mode)
    summary["base_checkpoint"] = str(cfg.checkpoint_path)
    summary["stream_manifest"] = str(cfg.stream_manifest)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if cfg.save_adaptation_checkpoint:
        save_adaptation_checkpoint(
            output_dir / "adaptation_state.pt",
            engine.expert_bank,
            OmegaConf.to_container(cfg.plasticity, resolve=True),
            processed,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
