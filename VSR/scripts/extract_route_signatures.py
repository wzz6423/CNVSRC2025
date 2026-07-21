import json
import os
import random
import subprocess
import sys
import tempfile
from numbers import Integral
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continual_adapt import (
    _build_engine,
    _file_sha256,
    _resolve_device,
    _stream_state,
)
from datamodule.transforms import DICT_PATH, TextTransform
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from plasticity.checkpoint import load_base_checkpoint
from plasticity.stream import iter_stream_manifest, load_stream_video


ARTIFACT_NAME = "route_signatures.npz"
ARTIFACT_KIND = "frozen_base_route_signatures"


def validate_signature_motion_order(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != 2:
        raise ValueError(
            "签名导出只支持 plasticity.signature_motion_order=2，"
            "以便在 validation 上统一校准 order 0/1/2"
        )
    return 2


def _metadata_path(output_path):
    return Path(f"{Path(output_path)}.meta.json")


def _ensure_outputs_absent(output_path):
    output_path = Path(output_path)
    metadata_path = _metadata_path(output_path)
    existing = [
        path for path in (output_path, metadata_path) if os.path.lexists(path)
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"拒绝覆盖已有签名产物：{paths}")
    return metadata_path


def _require_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"metadata.{name} 必须是 64 位小写十六进制 SHA-256")
    return value


def _unlink_if_owned(path, temporary_path):
    try:
        published = os.stat(path, follow_symlinks=False)
        temporary = os.stat(temporary_path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (published.st_dev, published.st_ino) == (temporary.st_dev, temporary.st_ino):
        Path(path).unlink()


def _publish_no_clobber(temporary_path, output_path):
    try:
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise FileExistsError(f"拒绝覆盖已有签名产物：{output_path}") from error


def _string_array(values, name):
    values = list(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} 必须全部为非空字符串")
    return np.asarray(values, dtype=np.str_)


def write_signature_artifacts(
    output_path,
    *,
    signatures,
    uids,
    domains,
    indices,
    metadata,
):
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("签名产物路径必须以 .npz 结尾")
    metadata_path = _metadata_path(output_path)

    signatures = np.asarray(signatures, dtype=np.float32)
    uids = _string_array(uids, "uids")
    domains = _string_array(domains, "domains")
    indices = np.asarray(indices, dtype=np.int64)
    if signatures.ndim != 2:
        raise ValueError("signatures 必须为 [N, D] 二维数组")
    if indices.ndim != 1 or uids.ndim != 1 or domains.ndim != 1:
        raise ValueError("uids、domains 和 indices 必须为一维数组")
    sample_count = signatures.shape[0]
    if not all(len(values) == sample_count for values in (uids, domains, indices)):
        raise ValueError("签名、UID、域和索引的样本数不一致")
    if not np.isfinite(signatures).all():
        raise ValueError("signatures 包含非有限值")
    if not isinstance(metadata, dict):
        raise TypeError("metadata 必须为字典")
    if metadata.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(
            f"metadata.artifact_kind 必须是 {ARTIFACT_KIND!r}"
        )
    _require_sha256(metadata.get("manifest_sha256"), "manifest_sha256")
    _require_sha256(
        metadata.get("base_checkpoint_sha256"), "base_checkpoint_sha256"
    )
    if metadata.get("samples") != sample_count:
        raise ValueError("metadata.samples 与签名样本数不一致")
    if metadata.get("signature_dim") != signatures.shape[1]:
        raise ValueError("metadata.signature_dim 与签名维度不一致")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_temporary_path = None
    metadata_temporary_path = None
    artifact_published = False
    metadata_published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            artifact_temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                signatures=signatures,
                uids=uids,
                domains=domains,
                indices=indices,
            )
            handle.flush()
            os.fsync(handle.fileno())
        written_metadata = {
            **metadata,
            "artifact_sha256": _file_sha256(artifact_temporary_path),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            metadata_temporary_path = Path(handle.name)
            json.dump(
                written_metadata,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        _publish_no_clobber(artifact_temporary_path, output_path)
        artifact_published = True
        _publish_no_clobber(metadata_temporary_path, metadata_path)
        metadata_published = True
    except Exception:
        if metadata_published:
            _unlink_if_owned(metadata_path, metadata_temporary_path)
        if artifact_published:
            _unlink_if_owned(output_path, artifact_temporary_path)
        raise
    finally:
        if metadata_temporary_path is not None:
            metadata_temporary_path.unlink(missing_ok=True)
        if artifact_temporary_path is not None:
            artifact_temporary_path.unlink(missing_ok=True)
    return output_path, metadata_path


def _git_commit(path):
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("无法读取当前代码的 Git commit") from error
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("当前代码的 Git commit 格式无效")
    return commit


def _build_metadata(cfg, stream_state, engine, sample_count, signature_dim, device):
    metadata = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "samples": int(sample_count),
        "signature_dim": int(signature_dim),
        "motion_order": int(engine.expert_bank.signature_motion_order),
        "device": str(device),
        "stream_manifest": str(Path(cfg.stream_manifest)),
        "manifest_sha256": stream_state["manifest_sha256"],
        "base_checkpoint": str(Path(cfg.checkpoint_path)),
        "base_checkpoint_sha256": _file_sha256(cfg.checkpoint_path),
        "target_vocab": str(Path(DICT_PATH)),
        "target_vocab_sha256": _file_sha256(DICT_PATH),
        "git_commit": _git_commit(ROOT),
    }
    if "manifest_metadata_sha256" in stream_state:
        metadata["manifest_sidecar"] = str(
            Path(f"{cfg.stream_manifest}.meta.json")
        )
        metadata["manifest_sidecar_sha256"] = stream_state[
            "manifest_metadata_sha256"
        ]
    return metadata


@hydra.main(config_path="../conf", config_name="continual_adapt", version_base=None)
def main(cfg: DictConfig):
    max_samples = None if cfg.max_samples is None else int(cfg.max_samples)
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples 不能为负数")
    validate_signature_motion_order(cfg.plasticity.signature_motion_order)
    output_path = Path(cfg.output_dir) / ARTIFACT_NAME
    _ensure_outputs_absent(output_path)

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = _resolve_device(cfg.device)
    text_transform = TextTransform()
    model = E2E(len(text_transform.token_list), cfg.model.visual_backbone)
    load_base_checkpoint(model, cfg.checkpoint_path, device)
    engine = _build_engine(cfg, model, text_transform.token_list, device)
    validate_signature_motion_order(engine.expert_bank.signature_motion_order)
    stream_state = _stream_state(
        cfg.stream_manifest, require_metadata=bool(cfg.require_manifest_metadata)
    )

    signatures = []
    uids = []
    domains = []
    indices = []
    stream = iter_stream_manifest(
        cfg.stream_manifest, cfg.data_root_dir, text_transform
    )
    with torch.inference_mode():
        for index, item in enumerate(stream):
            if max_samples is not None and index >= max_samples:
                break
            signature = engine.extract_route_signature(load_stream_video(item))
            signatures.append(signature.cpu().numpy())
            uids.append(item.uid)
            domains.append(item.domain)
            indices.append(index)
            if int(cfg.log_every) > 0 and len(signatures) % int(cfg.log_every) == 0:
                print(f"已提取 {len(signatures)} 条路由签名")

    signature_dim = int(engine.expert_bank.prototype_dim)
    signature_array = (
        np.stack(signatures).astype(np.float32, copy=False)
        if signatures
        else np.empty((0, signature_dim), dtype=np.float32)
    )
    metadata = _build_metadata(
        cfg,
        stream_state,
        engine,
        len(signatures),
        signature_dim,
        device,
    )
    artifact_path, metadata_path = write_signature_artifacts(
        output_path,
        signatures=signature_array,
        uids=uids,
        domains=domains,
        indices=indices,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "metadata": str(metadata_path),
                "samples": len(signatures),
                "signature_dim": signature_dim,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
