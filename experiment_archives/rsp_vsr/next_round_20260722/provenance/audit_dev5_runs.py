#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch


EXPECTED_COMMIT = "1e61295411e342ce0d567c2476049cd47595fc02"
EXPECTED_SAMPLES = 681
EXPECTED_STREAM_STATE = {
    "manifest_sha256": "8c8e967e7076562da70d47f45883ead84c5c2dd2ef47f69412467db2e89ebf56",
    "manifest_metadata_sha256": "27d96cedeba419350abb2daabd1ea9e2be03127d62b0eaf1badd58be7d9421c3",
    "target_vocab_sha256": "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0",
    "base_checkpoint_sha256": "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c",
}
RUN_SPECS = {
    "dev5_revisit_feature_film_combined_periodic_feedback10_seed42": {
        "mode": "single_adapter",
        "every": 10,
        "queries": 68,
        "adapter_type": "feature_film",
        "updatable_parameters": 1536,
        "parameter_tensors": 2,
    },
    "dev5_revisit_replay_adapter_combined_periodic_feedback10_seed42": {
        "mode": "single_adapter",
        "every": 10,
        "queries": 68,
        "adapter_type": "bottleneck",
        "updatable_parameters": 75265,
        "parameter_tensors": 5,
    },
    "dev5_revisit_static_seed42": {
        "mode": "static",
        "every": 0,
        "queries": 0,
        "adapter_type": "bottleneck",
        "updatable_parameters": 0,
        "parameter_tensors": 0,
    },
}
ERROR_PATTERN = re.compile(
    r"traceback|cuda out of memory|runtimeerror|exception|non.?finite",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args():
    parser = argparse.ArgumentParser(description="严格审计 RSP-VSR dev5 运行")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_status(path):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key] = value
    return values


def expected_query_indices(every):
    return set(range(every - 1, EXPECTED_SAMPLES, every)) if every else set()


def check_stream_rows(name, spec, rows, manifest_rows):
    if len(rows) > EXPECTED_SAMPLES:
        raise AssertionError(f"{name}: stream 比 manifest 长")
    expected_prefix = manifest_rows[: len(rows)]
    assert [row["index"] for row in rows] == list(range(len(rows))), name
    assert [row["uid"] for row in rows] == [row["uid"] for row in expected_prefix], name
    assert [row["domain"] for row in rows] == [
        row["domain"] for row in expected_prefix
    ], name
    assert len({row["uid"] for row in rows}) == len(rows), name

    query_indices = expected_query_indices(spec["every"])
    for row in rows:
        index = row["index"]
        queried = index in query_indices
        assert row["feedback_used"] is queried, (name, index)
        query = row["feedback_query"]
        assert query["queried"] is queried, (name, index)
        assert query["manifest_requested"] is False, (name, index)
        assert row["update"]["status"] != "failed", (name, index)
        if spec["mode"] == "static":
            assert row["update"]["status"] == "skipped", (name, index)
            assert row["update"]["reasons"] == ["adaptation_disabled"], (
                name,
                index,
            )
        elif queried:
            assert row["update"]["supervision"] == "feedback", (name, index)
        else:
            assert row["update"]["supervision"] == "pseudo", (name, index)


def check_history(name, history, processed, completed):
    steps = [row["processed_samples"] for row in history]
    assert steps == sorted(set(steps)), name
    assert all(step <= processed for step in steps), name
    assert all(step % 25 == 0 or step == EXPECTED_SAMPLES for step in steps), name
    for row in history:
        step = row["processed_samples"]
        expected_checkpoint = step % 100 == 0 or (
            completed and step == EXPECTED_SAMPLES
        )
        assert row["checkpoint"] is expected_checkpoint, (name, step)
        assert row["parameter_adaptation"]["parameter_count"] == RUN_SPECS[name][
            "updatable_parameters"
        ], name
    if completed:
        assert steps == list(range(25, 676, 25)) + [EXPECTED_SAMPLES], name


def check_checkpoint(name, spec, run_dir, summary):
    checkpoint_path = run_dir / "adaptation_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 3, name
    assert checkpoint["processed_samples"] == EXPECTED_SAMPLES, name
    assert checkpoint["stream_state"] == summary["stream_state"], name
    assert checkpoint["expert_count"] == 1, name
    assert checkpoint["expert_summary"]["adapter_type"] == spec["adapter_type"], name
    assert checkpoint["config"]["adapter_type"] == spec["adapter_type"], name
    adaptation_state = checkpoint["adaptation_state"]
    assert adaptation_state["parameter_update_mode"] == "adapter", name

    index_path = run_dir / "best_checkpoints" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 1, name
    assert len(index["checkpoints"]) == 2, name
    for entry in index["checkpoints"]:
        path = run_dir / "best_checkpoints" / entry["path"]
        assert path.is_file(), (name, path)
        best = torch.load(path, map_location="cpu", weights_only=True)
        assert best["version"] == 3, name
        assert best["processed_samples"] == entry["processed_samples"], name
        assert best["stream_state"] == summary["stream_state"], name


def check_summary(name, spec, summary, rows, run_dir):
    assert summary["samples"] == len(rows) == EXPECTED_SAMPLES, name
    assert summary["mode"] == spec["mode"], name
    assert summary["adaptation_objective"] == "rsp", name
    assert summary["feedback_update_strategy"] == "full_sequence", name
    query = summary["feedback_query"]
    assert query["strategy"] == "periodic", name
    assert query["every"] == spec["every"], name
    assert query["planned_budget"] == spec["queries"], name
    assert query["policy_queries"] == spec["queries"], name
    assert query["manifest_queries"] == 0, name
    assert query["total_queries"] == spec["queries"], name

    resources = summary["resources"]
    assert resources["updatable_parameters"] == spec["updatable_parameters"], name
    assert resources["updatable_parameter_tensors"] == spec["parameter_tensors"], name
    assert resources["retained_checkpoint_files"] == 3, name
    parameter = summary["parameter_adaptation"]
    assert parameter["parameter_update_mode"] == "adapter", name
    assert parameter["parameter_count"] == spec["updatable_parameters"], name
    assert parameter["parameter_tensors"] == spec["parameter_tensors"], name
    assert summary["expert_bank"]["adapter_type"] == spec["adapter_type"], name

    stream_state = summary["stream_state"]
    for key, expected in EXPECTED_STREAM_STATE.items():
        assert stream_state[key] == expected, (name, key)
    assert SHA256_PATTERN.fullmatch(stream_state["experiment_config_sha256"]), name
    assert len(list(run_dir.rglob("*.pt"))) == 3, name
    check_checkpoint(name, spec, run_dir, summary)


def main():
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    assert len(manifest_rows) == EXPECTED_SAMPLES
    assert [row["index"] if "index" in row else index for index, row in enumerate(manifest_rows)] == list(range(EXPECTED_SAMPLES))
    report = {"status": "AUDIT_OK", "final": args.final, "runs": {}}
    existing = {path.name for path in args.runs_root.iterdir() if path.is_dir()}
    if args.final:
        assert existing.issuperset(RUN_SPECS), sorted(set(RUN_SPECS) - existing)

    for name, spec in RUN_SPECS.items():
        run_dir = args.runs_root / name
        if not run_dir.is_dir():
            continue
        metadata = (run_dir / "run_metadata.txt").read_text(encoding="utf-8")
        assert f"git_commit={EXPECTED_COMMIT}" in metadata, name
        assert "stream_manifest=/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev5_" in metadata, name
        assert f"plasticity.adapter_type={spec['adapter_type']}" in metadata, name
        status = read_status(run_dir / "supervisor_status.tsv")
        assert status.get("attempt") == "1", name
        assert status.get("state") in {"running", "completed"}, name
        completed = status.get("state") == "completed"
        if args.final:
            assert completed, name
        log_text = (run_dir / "run.log").read_text(encoding="utf-8")
        assert ERROR_PATTERN.search(log_text) is None, name
        rows = read_jsonl(run_dir / "stream_results.jsonl")
        check_stream_rows(name, spec, rows, manifest_rows)
        history = read_jsonl(run_dir / "metrics_history.jsonl")
        check_history(name, history, len(rows), completed)
        pt_count = len(list(run_dir.rglob("*.pt")))
        assert pt_count <= 3, name
        summary_path = run_dir / "summary.json"
        if completed:
            assert summary_path.is_file(), name
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            check_summary(name, spec, summary, rows, run_dir)
        report["runs"][name] = {
            "state": status.get("state"),
            "attempt": int(status.get("attempt", 0)),
            "samples": len(rows),
            "history_rows": len(history),
            "pt_files": pt_count,
            "structured_errors": 0,
        }

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
