#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EXPECTED_COMMIT = "4191437d734f34cb524b049ba42415ec224a7ecb"
EXPECTED_STREAM_STATE = {
    "manifest_sha256": "a3b1a5842d05ec2caf09e63eb087a6d0623770ce4edcaeca9e9d9b14eb1bf132",
    "manifest_metadata_sha256": "a347d97f547498788847c0c9ed9247b7d67cf324d6e5f7cb760ca370542bc2f2",
    "target_vocab_sha256": "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0",
    "base_checkpoint_sha256": "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c",
}
RUN_SPECS = {
    "dev4_revisit_static_seed42": {
        "mode": "static",
        "objective": "rsp",
        "every": 0,
        "queries": 0,
        "updatable_parameters": 0,
        "parameter_tensors": 0,
        "parameter_mode": None,
    },
    "dev4_revisit_bn_tent_nofeedback_seed42": {
        "mode": "parameter_adaptation",
        "objective": "bn_tent",
        "every": 0,
        "queries": 0,
        "updatable_parameters": 9472,
        "parameter_tensors": 38,
        "parameter_mode": "batch_norm",
    },
    "dev4_revisit_eta_nofeedback_seed42": {
        "mode": "parameter_adaptation",
        "objective": "eta",
        "every": 0,
        "queries": 0,
        "updatable_parameters": 9472,
        "parameter_tensors": 38,
        "parameter_mode": "batch_norm",
    },
    "dev4_revisit_combined_periodic_feedback10_seed42": {
        "mode": "single_adapter",
        "objective": "rsp",
        "every": 10,
        "queries": 70,
        "updatable_parameters": 75265,
        "parameter_tensors": 5,
        "parameter_mode": None,
    },
    "dev4_revisit_online_lora_periodic_feedback10_seed42": {
        "mode": "parameter_adaptation",
        "objective": "online_lora",
        "every": 10,
        "queries": 70,
        "updatable_parameters": 73728,
        "parameter_tensors": 96,
        "parameter_mode": "lora",
    },
}
ERROR_PATTERN = re.compile(
    r"traceback|cuda out of memory|runtimeerror|exception|non.?finite",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="严格审计 RSP-VSR dev4 运行")
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


def query_indices(total, every):
    if every == 0:
        return set()
    return set(range(every - 1, total, every))


def check_stream_rows(name, spec, rows, manifest_rows):
    if len(rows) > len(manifest_rows):
        raise AssertionError(f"{name}: stream 比 manifest 长")
    expected_prefix = manifest_rows[: len(rows)]
    assert [row["index"] for row in rows] == list(range(len(rows))), name
    assert [row["uid"] for row in rows] == [row["uid"] for row in expected_prefix], name
    assert [row["domain"] for row in rows] == [
        row["domain"] for row in expected_prefix
    ], name
    assert len({row["uid"] for row in rows}) == len(rows), name
    expected_queries = query_indices(700, spec["every"])
    for row in rows:
        index = row["index"]
        should_query = index in expected_queries
        assert row["feedback_used"] is should_query, (name, index)
        query = row["feedback_query"]
        assert query["queried"] is should_query, (name, index)
        assert query["manifest_requested"] is False, (name, index)
        update = row["update"]
        assert update["status"] != "failed", (name, index, update)
        if name == "dev4_revisit_static_seed42":
            assert update["status"] == "skipped", (name, index)
            assert update["reasons"] == ["adaptation_disabled"], (name, index)
        elif spec["objective"] in {"bn_tent", "eta"}:
            assert row["feedback_used"] is False, (name, index)
            assert update["supervision"] == "entropy", (name, index)
            assert update["status"] in {"accepted", "skipped"}, (name, index)
        elif spec["objective"] == "online_lora":
            if should_query:
                assert update["status"] == "accepted", (name, index, update)
                assert update["supervision"] == "feedback", (name, index)
            else:
                assert update["status"] == "skipped", (name, index)
                assert update["reasons"] == ["non_feedback_updates_disabled"], (
                    name,
                    index,
                )


def check_history(name, history, processed, final):
    samples = [row["processed_samples"] for row in history]
    assert samples == sorted(set(samples)), name
    assert all(sample % 25 == 0 for sample in samples), name
    assert all(sample <= processed for sample in samples), name
    for row in history:
        expected_checkpoint = row["processed_samples"] % 100 == 0
        assert row["checkpoint"] is expected_checkpoint, (name, row)
    if final:
        assert samples == list(range(25, 701, 25)), name


def check_summary(name, spec, summary, rows, run_dir):
    assert summary["samples"] == len(rows) == 700, name
    assert summary["mode"] == spec["mode"], name
    assert summary["adaptation_objective"] == spec["objective"], name
    query = summary["feedback_query"]
    assert query["strategy"] == "periodic", name
    assert query["every"] == spec["every"], name
    assert query["planned_budget"] == spec["queries"], name
    assert query["policy_queries"] == spec["queries"], name
    assert query["manifest_queries"] == 0, name
    assert query["total_queries"] == spec["queries"], name
    resources = summary["resources"]
    assert resources["updatable_parameters"] == spec["updatable_parameters"], name
    assert resources["retained_checkpoint_files"] == 3, name
    parameter = summary["parameter_adaptation"]
    expected_parameter_mode = spec["parameter_mode"] or "adapter"
    assert parameter["parameter_update_mode"] == expected_parameter_mode, name
    assert parameter["parameter_count"] == spec["updatable_parameters"], name
    if "parameter_tensors" in spec:
        assert parameter["parameter_tensors"] == spec["parameter_tensors"], name
    stream_state = summary["stream_state"]
    for key, expected in EXPECTED_STREAM_STATE.items():
        assert stream_state[key] == expected, (name, key)
    assert re.fullmatch(r"[0-9a-f]{64}", stream_state["experiment_config_sha256"])
    assert len(list(run_dir.rglob("*.pt"))) == 3, name


def main():
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    assert len(manifest_rows) == 700
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
        assert "stream_manifest=/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev4_" in metadata, name
        status = read_status(run_dir / "supervisor_status.tsv")
        assert status.get("attempt") == "1", name
        assert status.get("state") in {"running", "completed"}, name
        if args.final:
            assert status["state"] == "completed", name
        log_text = (run_dir / "run.log").read_text(encoding="utf-8")
        assert ERROR_PATTERN.search(log_text) is None, name
        rows = read_jsonl(run_dir / "stream_results.jsonl")
        check_stream_rows(name, spec, rows, manifest_rows)
        history = read_jsonl(run_dir / "metrics_history.jsonl")
        check_history(name, history, len(rows), args.final)
        pt_count = len(list(run_dir.rglob("*.pt")))
        assert pt_count <= 3, name
        summary_path = run_dir / "summary.json"
        if args.final:
            assert summary_path.is_file(), name
        if summary_path.is_file():
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
