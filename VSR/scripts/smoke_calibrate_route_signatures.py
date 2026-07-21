import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST_SHA256 = "a" * 64
BASE_CHECKPOINT_SHA256 = "b" * 64

from plasticity.signature_calibration import (
    calibrate_signature_artifacts,
    load_signature_artifacts,
    select_route_threshold,
    unknown_auroc,
    write_calibration_json,
)


def _write_fixture(
    directory,
    *,
    signatures=None,
    uids=None,
    domains=None,
    indices=None,
    metadata_updates=None,
):
    artifact_path = directory / "route_signatures.npz"
    metadata_path = Path(f"{artifact_path}.meta.json")
    block_a = np.asarray([1.0, 0.0], dtype=np.float32)
    block_b = np.asarray([0.0, 1.0], dtype=np.float32)
    signature_a = np.tile(block_a, 4)
    signature_b = np.tile(block_b, 4)
    if signatures is None:
        signatures = np.stack([signature_a] * 4 + [signature_b] * 4)
        signatures /= np.linalg.norm(signatures, axis=1, keepdims=True)
        signatures = signatures.astype(np.float32)
    if uids is None:
        uids = np.asarray(
            [f"a-{index}" for index in range(4)]
            + [f"b-{index}" for index in range(4)]
        )
    if domains is None:
        domains = np.asarray(["A"] * 4 + ["B"] * 4)
    if indices is None:
        indices = np.arange(len(signatures), dtype=np.int64)
    np.savez_compressed(
        artifact_path,
        signatures=signatures,
        uids=uids,
        domains=domains,
        indices=indices,
    )
    metadata = {
        "schema_version": 1,
        "artifact_kind": "frozen_base_route_signatures",
        "samples": len(signatures),
        "signature_dim": signatures.shape[1],
        "motion_order": 2,
        "manifest_sha256": MANIFEST_SHA256,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    metadata.update(metadata_updates or {})
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return artifact_path, metadata_path


def _assert_rejected(callable_, expected_exception, message_fragment):
    try:
        callable_()
    except expected_exception as error:
        assert message_fragment in str(error), str(error)
    else:
        raise AssertionError(f"预期抛出 {expected_exception.__name__}")


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact_path, metadata_path = _write_fixture(Path(temporary_directory))
        result = calibrate_signature_artifacts(
            artifact_path,
            metadata_path=metadata_path,
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            seed=42,
            reference_fraction=0.5,
        )
        assert result["evaluated_motion_orders"] == [0, 1, 2]
        assert result["best_motion_order"] == 0
        assert result["orders"]["0"]["closed_set_accuracy"] == 1.0
        assert result["orders"]["0"]["clustering_purity"] == 1.0
        assert result["orders"]["0"]["unknown_detection"]["auroc"] == 1.0
        assert result["orders"]["0"]["unknown_detection"]["best_f1"] == 1.0
        assert result["orders"]["0"]["unknown_detection"]["route_threshold"] == 0.5
        assert (
            result["orders"]["0"]["unknown_detection"]["protocol"]
            == "leave_own_domain_out_proxy"
        )
        assert "not a true unseen-domain evaluation" in result["orders"]["0"][
            "unknown_detection"
        ]["limitation"]
        assert result["orders"]["0"]["top1_top2_margin"]["mean"] == 1.0
        assert result["orders"]["0"]["own_prototype_within_similarity"]["mean"] == 1.0
        assert result["orders"]["0"]["excluding_own_max_cross_similarity"]["mean"] == 0.0
        assert result["split"]["reference_samples"] == 4
        assert result["split"]["eval_samples"] == 4
        repeated = calibrate_signature_artifacts(
            artifact_path,
            metadata_path=metadata_path,
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            seed=42,
            reference_fraction=0.5,
        )
        assert repeated["split"] == result["split"]

        assert unknown_auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
        tied_threshold = select_route_threshold([0.9, 0.9], [0.1, 0.1])
        assert tied_threshold["route_threshold"] == 0.5
        assert tied_threshold["best_f1"] == 1.0

        order_two_root = Path(temporary_directory) / "order-two"
        order_two_root.mkdir()
        raw_a = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        raw_b = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        order_two_signatures = np.stack([raw_a] * 4 + [raw_b] * 4)
        order_two_signatures /= np.linalg.norm(
            order_two_signatures, axis=1, keepdims=True
        )
        order_two_npz, order_two_meta = _write_fixture(
            order_two_root,
            signatures=order_two_signatures.astype(np.float32),
        )
        order_two_result = calibrate_signature_artifacts(
            order_two_npz,
            metadata_path=order_two_meta,
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            seed=42,
            reference_fraction=0.5,
        )
        assert order_two_result["source"]["feature_dim"] == 1
        assert order_two_result["best_motion_order"] == 2
        assert order_two_result["orders"]["0"]["closed_set_accuracy"] == 0.5
        assert order_two_result["orders"]["1"]["closed_set_accuracy"] == 0.5
        assert order_two_result["orders"]["2"]["closed_set_accuracy"] == 1.0

        output_path = Path(temporary_directory) / "calibration.json"
        write_calibration_json(output_path, result)
        assert json.loads(output_path.read_text(encoding="utf-8")) == result
        output_bytes = output_path.read_bytes()
        _assert_rejected(
            lambda: write_calibration_json(output_path, result),
            FileExistsError,
            "拒绝覆盖",
        )
        assert output_path.read_bytes() == output_bytes
        assert not list(Path(temporary_directory).glob(".*.tmp"))

        cli_output = Path(temporary_directory) / "cli-calibration.json"
        cli = ROOT / "scripts" / "calibrate_route_signatures.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--input",
                str(artifact_path),
                "--metadata",
                str(metadata_path),
                "--output",
                str(cli_output),
                "--expected-manifest-sha256",
                MANIFEST_SHA256,
                "--expected-base-checkpoint-sha256",
                BASE_CHECKPOINT_SHA256,
                "--seed",
                "42",
                "--reference-fraction",
                "0.5",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        cli_summary = json.loads(completed.stdout)
        assert cli_summary["output"] == str(cli_output)
        assert cli_summary["best_motion_order"] == 0
        assert json.loads(cli_output.read_text(encoding="utf-8")) == result
        cli_output_bytes = cli_output.read_bytes()
        rejected = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--input",
                str(artifact_path),
                "--output",
                str(cli_output),
                "--expected-manifest-sha256",
                MANIFEST_SHA256,
                "--expected-base-checkpoint-sha256",
                BASE_CHECKPOINT_SHA256,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "拒绝覆盖" in rejected.stderr
        assert cli_output.read_bytes() == cli_output_bytes

        missing = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--input",
                str(Path(temporary_directory) / "missing.npz"),
                "--output",
                str(Path(temporary_directory) / "missing.json"),
                "--expected-manifest-sha256",
                MANIFEST_SHA256,
                "--expected-base-checkpoint-sha256",
                BASE_CHECKPOINT_SHA256,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing.returncode == 2
        assert "找不到路由签名 NPZ" in missing.stderr

        invalid_root = Path(temporary_directory) / "invalid"
        invalid_root.mkdir()
        wrong_dtype, wrong_dtype_meta = _write_fixture(
            invalid_root,
            signatures=np.ones((8, 8), dtype=np.float64),
        )
        _assert_rejected(
            lambda: load_signature_artifacts(wrong_dtype, wrong_dtype_meta),
            ValueError,
            "float32",
        )

        duplicate_root = Path(temporary_directory) / "duplicate"
        duplicate_root.mkdir()
        duplicate_uids, duplicate_meta = _write_fixture(
            duplicate_root,
            uids=np.asarray(["duplicate"] * 8),
        )
        _assert_rejected(
            lambda: load_signature_artifacts(duplicate_uids, duplicate_meta),
            ValueError,
            "全局唯一",
        )

        metadata_root = Path(temporary_directory) / "metadata"
        metadata_root.mkdir()
        mismatched, mismatched_meta = _write_fixture(
            metadata_root,
            metadata_updates={"samples": 9},
        )
        _assert_rejected(
            lambda: load_signature_artifacts(mismatched, mismatched_meta),
            ValueError,
            "与 NPZ 样本数",
        )

        object_root = Path(temporary_directory) / "object"
        object_root.mkdir()
        object_npz, object_meta = _write_fixture(
            object_root,
            uids=np.asarray([f"uid-{index}" for index in range(8)], dtype=object),
        )
        _assert_rejected(
            lambda: load_signature_artifacts(object_npz, object_meta),
            ValueError,
            "禁止 pickle/object",
        )

        missing_key_root = Path(temporary_directory) / "missing-key"
        missing_key_root.mkdir()
        missing_key_npz, missing_key_meta = _write_fixture(missing_key_root)
        with np.load(missing_key_npz, allow_pickle=False) as artifact:
            np.savez_compressed(
                missing_key_npz,
                signatures=artifact["signatures"],
                uids=artifact["uids"],
                domains=artifact["domains"],
            )
        missing_key_metadata = json.loads(
            missing_key_meta.read_text(encoding="utf-8")
        )
        missing_key_metadata["artifact_sha256"] = hashlib.sha256(
            missing_key_npz.read_bytes()
        ).hexdigest()
        missing_key_meta.write_text(
            json.dumps(missing_key_metadata),
            encoding="utf-8",
        )
        _assert_rejected(
            lambda: load_signature_artifacts(missing_key_npz, missing_key_meta),
            ValueError,
            "keys 无效",
        )

        one_domain_root = Path(temporary_directory) / "one-domain"
        one_domain_root.mkdir()
        one_domain, one_domain_meta = _write_fixture(
            one_domain_root,
            domains=np.asarray(["A"] * 8),
        )
        _assert_rejected(
            lambda: calibrate_signature_artifacts(
                one_domain,
                metadata_path=one_domain_meta,
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            ),
            ValueError,
            "至少需要 2 个 domain",
        )

        _assert_rejected(
            lambda: calibrate_signature_artifacts(
                artifact_path,
                metadata_path=metadata_path,
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
                feature_dim=3,
            ),
            ValueError,
            "签名维度应为",
        )

        _assert_rejected(
            lambda: calibrate_signature_artifacts(
                artifact_path,
                metadata_path=metadata_path,
                expected_manifest_sha256="c" * 64,
                expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            ),
            ValueError,
            "manifest SHA-256 不匹配",
        )
        _assert_rejected(
            lambda: calibrate_signature_artifacts(
                artifact_path,
                metadata_path=metadata_path,
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_base_checkpoint_sha256="c" * 64,
            ),
            ValueError,
            "base checkpoint SHA-256 不匹配",
        )
        _assert_rejected(
            lambda: calibrate_signature_artifacts(
                artifact_path,
                metadata_path=metadata_path,
                expected_manifest_sha256="A" * 64,
                expected_base_checkpoint_sha256=BASE_CHECKPOINT_SHA256,
            ),
            ValueError,
            "必须是 64 位小写十六进制",
        )

        tampered_root = Path(temporary_directory) / "tampered"
        tampered_root.mkdir()
        tampered_npz, tampered_meta = _write_fixture(tampered_root)
        with tampered_npz.open("ab") as handle:
            handle.write(b"tampered")
        _assert_rejected(
            lambda: load_signature_artifacts(tampered_npz, tampered_meta),
            ValueError,
            "artifact SHA-256 不匹配",
        )

        missing_provenance_root = Path(temporary_directory) / "missing-provenance"
        missing_provenance_root.mkdir()
        missing_provenance, missing_provenance_meta = _write_fixture(
            missing_provenance_root,
            metadata_updates={"artifact_kind": "wrong_kind"},
        )
        _assert_rejected(
            lambda: load_signature_artifacts(
                missing_provenance, missing_provenance_meta
            ),
            ValueError,
            "artifact_kind",
        )

        missing_required = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--input",
                str(artifact_path),
                "--output",
                str(Path(temporary_directory) / "missing-required.json"),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing_required.returncode == 2
        assert "--expected-manifest-sha256" in missing_required.stderr
        assert "--expected-base-checkpoint-sha256" in missing_required.stderr

    print("route signature calibration smoke test passed")


if __name__ == "__main__":
    main()
