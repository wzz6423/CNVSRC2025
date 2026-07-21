import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.adapters import ExpertBank
from plasticity.engine import ContinualAdaptationEngine
from scripts.extract_route_signatures import (
    _build_metadata,
    validate_signature_motion_order,
    write_signature_artifacts,
)


class MockEncoder(nn.Module):
    def __init__(self, channels=1, feature_dim=8):
        super().__init__()
        self.projection = nn.Linear(channels, feature_dim, bias=False)

    def forward(self, video, _mask):
        pooled = video.mean(dim=(-1, -2))
        return self.projection(pooled), None


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MockEncoder()


class FailingDecoder:
    def __call__(self, _features):
        raise AssertionError("提取路由签名时不应调用解码器")


def _snapshot(module):
    return {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }


def _assert_unchanged(before, module):
    after = module.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def _assert_rejected(callable_, expected_exception, message_fragment):
    try:
        callable_()
    except expected_exception as error:
        assert message_fragment in str(error)
    else:
        raise AssertionError(f"预期抛出 {expected_exception.__name__}")


def main():
    assert validate_signature_motion_order(2) == 2
    _assert_rejected(
        lambda: validate_signature_motion_order(1),
        ValueError,
        "signature_motion_order=2",
    )

    torch.manual_seed(17)
    model = MockModel()
    bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=3,
        signature_motion_order=2,
    )
    engine = ContinualAdaptationEngine(
        model,
        bank,
        reliability_gate=object(),
        decoder=FailingDecoder(),
        device="cpu",
    )
    model_before = _snapshot(model)
    bank_before = _snapshot(bank)
    video = torch.randn(12, 1, 6, 6)

    signature = engine.extract_route_signature(video)
    assert signature.shape == (32,)
    assert signature.dtype == torch.float32
    assert torch.allclose(signature.norm(), torch.tensor(1.0), atol=1e-6)
    assert not signature.requires_grad
    assert not model.training
    assert bank.expert_count == 0
    assert engine.optimizer_state_dict() == {}
    _assert_unchanged(model_before, model)
    _assert_unchanged(bank_before, bank)

    _assert_rejected(
        lambda: engine.extract_route_signature(torch.randn(2, 3, 4)),
        ValueError,
        "[T, C, H, W]",
    )
    _assert_rejected(
        lambda: engine.extract_route_signature(torch.empty(0, 1, 6, 6)),
        ValueError,
        "时间维不能为空",
    )
    _assert_rejected(
        lambda: engine.extract_route_signature([[1.0]]),
        TypeError,
        "torch.Tensor",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "base.pth"
        checkpoint_path.write_bytes(b"base-checkpoint")
        provenance = _build_metadata(
            SimpleNamespace(
                stream_manifest=Path(temporary_directory) / "stream.jsonl",
                checkpoint_path=checkpoint_path,
            ),
            {
                "manifest_sha256": "a" * 64,
                "manifest_metadata_sha256": "b" * 64,
            },
            engine,
            sample_count=2,
            signature_dim=32,
            device=torch.device("cpu"),
        )
        assert provenance["manifest_sha256"] == "a" * 64
        assert provenance["manifest_sidecar_sha256"] == "b" * 64
        assert provenance["base_checkpoint_sha256"] == hashlib.sha256(
            b"base-checkpoint"
        ).hexdigest()
        assert len(provenance["target_vocab_sha256"]) == 64
        assert len(provenance["git_commit"]) == 40
        assert provenance["motion_order"] == 2
        assert provenance["samples"] == 2
        assert provenance["signature_dim"] == 32
        assert provenance["artifact_kind"] == "frozen_base_route_signatures"

        output_path = Path(temporary_directory) / "route_signatures.npz"
        signatures = np.stack(
            [signature.numpy(), (signature * 0.5).numpy()]
        ).astype(np.float64)
        metadata = {
            "schema_version": 1,
            "artifact_kind": "frozen_base_route_signatures",
            "samples": 2,
            "signature_dim": 32,
            "motion_order": 2,
            "manifest_sha256": "a" * 64,
            "base_checkpoint_sha256": "b" * 64,
        }
        artifact_path, metadata_path = write_signature_artifacts(
            output_path,
            signatures=signatures,
            uids=["sample-a", "sample-b"],
            domains=["130", "024"],
            indices=[4, 9],
            metadata=metadata,
        )
        assert artifact_path == output_path
        assert metadata_path == Path(f"{output_path}.meta.json")
        assert artifact_path.is_file()
        assert metadata_path.is_file()
        with np.load(artifact_path, allow_pickle=False) as artifact:
            assert set(artifact.files) == {
                "signatures",
                "uids",
                "domains",
                "indices",
            }
            assert artifact["signatures"].dtype == np.float32
            assert artifact["signatures"].shape == (2, 32)
            assert artifact["uids"].dtype.kind == "U"
            assert artifact["uids"].tolist() == ["sample-a", "sample-b"]
            assert artifact["domains"].dtype.kind == "U"
            assert artifact["domains"].tolist() == ["130", "024"]
            assert artifact["indices"].dtype == np.int64
            assert artifact["indices"].tolist() == [4, 9]
        written_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert written_metadata == {
            **metadata,
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
        assert not list(Path(temporary_directory).glob(".*.tmp"))

        artifact_bytes = artifact_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        _assert_rejected(
            lambda: write_signature_artifacts(
                output_path,
                signatures=signatures,
                uids=["sample-a", "sample-b"],
                domains=["130", "024"],
                indices=[4, 9],
                metadata=metadata,
            ),
            FileExistsError,
            "拒绝覆盖",
        )
        assert artifact_path.read_bytes() == artifact_bytes
        assert metadata_path.read_bytes() == metadata_bytes

        sidecar_conflict_path = (
            Path(temporary_directory) / "sidecar-conflict.npz"
        )
        sidecar_conflict_metadata = Path(f"{sidecar_conflict_path}.meta.json")
        sidecar_conflict_metadata.write_text("concurrent-sidecar", encoding="utf-8")
        _assert_rejected(
            lambda: write_signature_artifacts(
                sidecar_conflict_path,
                signatures=signatures,
                uids=["sample-a", "sample-b"],
                domains=["130", "024"],
                indices=[4, 9],
                metadata=metadata,
            ),
            FileExistsError,
            "拒绝覆盖",
        )
        assert not sidecar_conflict_path.exists()
        assert (
            sidecar_conflict_metadata.read_text(encoding="utf-8")
            == "concurrent-sidecar"
        )
        assert not list(Path(temporary_directory).glob(".*.tmp"))

        artifact_conflict_path = (
            Path(temporary_directory) / "artifact-conflict.npz"
        )
        artifact_conflict_path.write_bytes(b"concurrent-artifact")
        _assert_rejected(
            lambda: write_signature_artifacts(
                artifact_conflict_path,
                signatures=signatures,
                uids=["sample-a", "sample-b"],
                domains=["130", "024"],
                indices=[4, 9],
                metadata=metadata,
            ),
            FileExistsError,
            "拒绝覆盖",
        )
        assert artifact_conflict_path.read_bytes() == b"concurrent-artifact"
        assert not Path(f"{artifact_conflict_path}.meta.json").exists()
        assert not list(Path(temporary_directory).glob(".*.tmp"))

        invalid_path = Path(temporary_directory) / "invalid.npz"
        _assert_rejected(
            lambda: write_signature_artifacts(
                invalid_path,
                signatures=signatures,
                uids=["sample-a"],
                domains=["130", "024"],
                indices=[4, 9],
                metadata=metadata,
            ),
            ValueError,
            "样本数不一致",
        )
        assert not invalid_path.exists()
        assert not Path(f"{invalid_path}.meta.json").exists()

    print("route signature extraction smoke test passed")


if __name__ == "__main__":
    main()
