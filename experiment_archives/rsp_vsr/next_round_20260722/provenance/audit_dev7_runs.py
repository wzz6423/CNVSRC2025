#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import torch


EXPECTED_COMMIT = "b9d6d499dfe88703cf45a661bf4463880cf0a632"
EXPECTED_SAMPLES = 625
EXPECTED_QUERIES = 62
EXPECTED_STREAM_STATE = {
    "manifest_sha256": "22e94cffece7f496219225058c4547ec038f4046eb2f794a2cf6187d299467b8",
    "manifest_metadata_sha256": "108649b9c751bda793f34ea4d9b7840c483ee996b9519fb8e2b69cf93800bedf",
    "target_vocab_sha256": "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0",
    "base_checkpoint_sha256": "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c",
}
RUN_SPECS = {
    "dev7_revisit_replay_adapter_nbest10_periodic_feedback10_seed42": {
        "counterfactual": False,
        "experiment_config_sha256": "58ab600ab4959add2028b3344d9717df5dac48281718e3f1edefc080711d17d4",
    },
    "dev7_revisit_counterfactual_margin_nbest10_periodic_feedback10_seed42": {
        "counterfactual": True,
        "experiment_config_sha256": "8da0a9e78880873dab454b1143812f1a4b43462360824c4be1900b62bc135b82",
    },
}
ERROR_PATTERN = re.compile(
    r"traceback|cuda out of memory|runtimeerror|exception|assertionerror|"
    r"non.?finite|worker exited|stalled",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="严格审计 RSP-VSR dev7 运行")
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


def expected_query_indices():
    return set(range(9, EXPECTED_SAMPLES, 10))


def assert_close(left, right, tolerance=1e-6):
    assert math.isfinite(float(left))
    assert math.isfinite(float(right))
    assert abs(float(left) - float(right)) <= tolerance, (left, right)


def check_nbest(name, row):
    nbest = row.get("decoder_nbest")
    assert isinstance(nbest, list) and len(nbest) == 10, (name, row["index"])
    assert [candidate["rank"] for candidate in nbest] == list(range(1, 11)), (
        name,
        row["index"],
    )
    for candidate in nbest:
        assert isinstance(candidate.get("transcript"), str), (name, row["index"])
        tokens = candidate.get("tokens")
        assert isinstance(tokens, list), (name, row["index"])
        assert all(isinstance(token, int) and token > 0 for token in tokens), (
            name,
            row["index"],
        )
        assert math.isfinite(float(candidate["score"])), (name, row["index"])
        assert math.isfinite(float(candidate["normalized_score"])), (
            name,
            row["index"],
        )
        scores = candidate.get("scores")
        assert isinstance(scores, dict) and scores, (name, row["index"])
        assert all(math.isfinite(float(value)) for value in scores.values()), (
            name,
            row["index"],
        )


def check_counterfactual(name, spec, row, manifest_row, queried, stats):
    diagnostic = row["update"].get("counterfactual")
    if not spec["counterfactual"] or not queried:
        assert diagnostic is None, (name, row["index"])
        return

    assert isinstance(diagnostic, dict), (name, row["index"])
    stats["feedback_attempts"] += 1
    assert diagnostic["candidate_count"] == 10, (name, row["index"])
    valid_count = int(diagnostic["valid_candidate_count"])
    assert 0 <= valid_count <= 10, (name, row["index"])
    if diagnostic["status"] == "no_valid_negative":
        assert valid_count == 0, (name, row["index"])
        stats["no_valid_negative"] += 1
        return

    assert diagnostic["status"] in {"applied", "rolled_back"}, (
        name,
        row["index"],
    )
    assert valid_count >= 1, (name, row["index"])
    assert 1 <= int(diagnostic["negative_rank"]) <= 10, (name, row["index"])
    negative_tokens = diagnostic["negative_tokens"]
    assert isinstance(negative_tokens, list) and negative_tokens, (
        name,
        row["index"],
    )
    assert negative_tokens != manifest_row["target_tokens"], (name, row["index"])
    for suffix in ("before", "after"):
        target_loss = float(diagnostic[f"target_loss_{suffix}"])
        negative_loss = float(diagnostic[f"negative_loss_{suffix}"])
        gap = float(diagnostic[f"gap_{suffix}"])
        violation = float(diagnostic[f"violation_{suffix}"])
        assert target_loss >= 0 and negative_loss >= 0 and violation >= 0
        assert_close(gap, negative_loss - target_loss)
        assert_close(violation, max(0.0, 0.2 - gap))
    violation_before = float(diagnostic["violation_before"])
    violation_after = float(diagnostic["violation_after"])
    assert_close(
        diagnostic["violation_reduction"], violation_before - violation_after
    )
    if violation_before > 0:
        stats["positive_violations_before"] += 1
    if violation_after > violation_before + 1e-6:
        assert diagnostic["status"] == "rolled_back", (name, row["index"])
        assert "counterfactual_violation_regressed" in row["update"]["reasons"], (
            name,
            row["index"],
        )
    if diagnostic["status"] == "applied":
        assert row["update"]["status"] == "accepted", (name, row["index"])
    else:
        assert row["update"]["status"] == "rolled_back", (name, row["index"])
    stats[diagnostic["status"]] += 1
    stats["valid_negatives"] += 1
    stats["violation_before_sum"] += violation_before
    stats["violation_after_sum"] += violation_after


def check_stream_rows(name, spec, rows, manifest_rows):
    assert len(rows) <= EXPECTED_SAMPLES, name
    expected_prefix = manifest_rows[: len(rows)]
    assert [row["index"] for row in rows] == list(range(len(rows))), name
    assert [row["uid"] for row in rows] == [row["uid"] for row in expected_prefix], name
    assert [row["domain"] for row in rows] == [
        row["domain"] for row in expected_prefix
    ], name
    assert len({row["uid"] for row in rows}) == len(rows), name

    query_indices = expected_query_indices()
    stats = {
        "feedback_attempts": 0,
        "valid_negatives": 0,
        "positive_violations_before": 0,
        "applied": 0,
        "rolled_back": 0,
        "failed": 0,
        "no_valid_negative": 0,
        "violation_before_sum": 0.0,
        "violation_after_sum": 0.0,
    }
    for row, manifest_row in zip(rows, expected_prefix):
        index = row["index"]
        queried = index in query_indices
        assert row["feedback_used"] is queried, (name, index)
        query = row["feedback_query"]
        assert query["queried"] is queried, (name, index)
        assert query["manifest_requested"] is False, (name, index)
        expected_reason = (
            "periodic_slot"
            if queried
            else "partial_window"
            if index >= 620
            else "not_periodic_slot"
        )
        assert query["reason"] == expected_reason, (name, index)
        assert row["update"]["status"] != "failed", (name, index)
        assert row["update"]["supervision"] == (
            "feedback" if queried else "pseudo"
        ), (name, index)
        check_nbest(name, row)
        check_counterfactual(name, spec, row, manifest_row, queried, stats)

    if spec["counterfactual"] and stats["feedback_attempts"] >= 10:
        assert stats["valid_negatives"] >= 8, (name, "early_valid_negative_gate")
        assert stats["positive_violations_before"] >= 1, (
            name,
            "early_positive_violation_gate",
        )
    return stats


def check_history(name, spec, history, processed, completed):
    steps = [row["processed_samples"] for row in history]
    assert steps == list(range(25, processed + 1, 25)), name
    for row in history:
        step = row["processed_samples"]
        expected_checkpoint = step % 100 == 0 or (
            completed and step == EXPECTED_SAMPLES
        )
        assert row["checkpoint"] is expected_checkpoint, (name, step)
        parameter = row["parameter_adaptation"]
        assert parameter["parameter_update_mode"] == "adapter", (name, step)
        assert parameter["parameter_count"] == 75265, (name, step)
        assert parameter["parameter_tensors"] == 5, (name, step)
        if spec["counterfactual"]:
            assert parameter["counterfactual_margin"]["enabled"] is True, (
                name,
                step,
            )
        else:
            assert "counterfactual_margin" not in parameter, (name, step)
    if completed:
        assert len(history) == 25 and steps[-1] == EXPECTED_SAMPLES, name


def check_checkpoint(name, spec, run_dir, summary, stats):
    checkpoint_path = run_dir / "adaptation_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 3, name
    assert checkpoint["processed_samples"] == EXPECTED_SAMPLES, name
    assert checkpoint["stream_state"] == summary["stream_state"], name
    assert checkpoint["expert_count"] == 1, name
    assert checkpoint["expert_summary"]["adapter_type"] == "bottleneck", name
    config = checkpoint["config"]
    assert config["adapter_type"] == "bottleneck", name
    assert config["feedback_update_strategy"] == "full_sequence", name
    counterfactual_config = config["counterfactual_margin"]
    assert counterfactual_config == {
        "enabled": spec["counterfactual"],
        "margin": 0.2,
        "weight": 0.25,
        "rollback_tolerance": 1e-6,
    }, name
    adaptation_state = checkpoint["adaptation_state"]
    assert adaptation_state["parameter_update_mode"] == "adapter", name
    if spec["counterfactual"]:
        assert adaptation_state["counterfactual_margin"] == summary[
            "counterfactual_margin"
        ], name
    else:
        assert "counterfactual_margin" not in adaptation_state, name

    index_path = run_dir / "best_checkpoints" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 1 and len(index["checkpoints"]) == 2, name
    for entry in index["checkpoints"]:
        path = run_dir / "best_checkpoints" / entry["path"]
        assert path.is_file(), (name, path)
        best = torch.load(path, map_location="cpu", weights_only=True)
        assert best["version"] == 3, name
        assert best["processed_samples"] == entry["processed_samples"], name
        assert best["stream_state"] == summary["stream_state"], name


def check_summary(name, spec, summary, rows, run_dir, stats):
    assert summary["samples"] == len(rows) == EXPECTED_SAMPLES, name
    assert summary["mode"] == "single_adapter", name
    assert summary["adaptation_objective"] == "rsp", name
    assert summary["feedback_update_strategy"] == "full_sequence", name
    assert summary["non_feedback_updates_enabled"] is True, name
    query = summary["feedback_query"]
    assert query["strategy"] == "periodic" and query["every"] == 10, name
    assert query["planned_budget"] == EXPECTED_QUERIES, name
    assert query["policy_queries"] == EXPECTED_QUERIES, name
    assert query["manifest_queries"] == 0, name
    assert query["total_queries"] == EXPECTED_QUERIES, name

    parameter = summary["parameter_adaptation"]
    resources = summary["resources"]
    assert parameter["parameter_update_mode"] == "adapter", name
    assert parameter["parameter_count"] == resources["updatable_parameters"] == 75265
    assert parameter["parameter_tensors"] == resources[
        "updatable_parameter_tensors"
    ] == 5
    assert summary["expert_bank"]["adapter_type"] == "bottleneck", name
    assert resources["retained_checkpoint_files"] == 3, name
    assert len(list(run_dir.rglob("*.pt"))) == 3, name

    stream_state = summary["stream_state"]
    for key, expected in EXPECTED_STREAM_STATE.items():
        assert stream_state[key] == expected, (name, key)
    assert stream_state["experiment_config_sha256"] == spec[
        "experiment_config_sha256"
    ], name

    if spec["counterfactual"]:
        counterfactual = summary["counterfactual_margin"]
        assert counterfactual == parameter["counterfactual_margin"], name
        assert counterfactual["enabled"] is True, name
        assert counterfactual["margin"] == 0.2, name
        assert counterfactual["weight"] == 0.25, name
        assert counterfactual["rollback_tolerance"] == 1e-6, name
        for key, value in stats.items():
            if key.endswith("_sum"):
                assert_close(counterfactual[key], value, tolerance=1e-5)
            else:
                assert counterfactual[key] == value, (name, key)
        assert counterfactual["feedback_attempts"] == EXPECTED_QUERIES, name
        assert counterfactual["valid_negatives"] >= 56, name
        assert counterfactual["positive_violations_before"] >= 1, name
    else:
        assert "counterfactual_margin" not in summary, name
        assert "counterfactual_margin" not in parameter, name
    check_checkpoint(name, spec, run_dir, summary, stats)


def main():
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    assert len(manifest_rows) == EXPECTED_SAMPLES
    assert len({row["uid"] for row in manifest_rows}) == EXPECTED_SAMPLES
    assert all(row["feedback"] is False for row in manifest_rows)
    report = {"status": "AUDIT_OK", "final": args.final, "runs": {}}
    existing = {path.name for path in args.runs_root.iterdir() if path.is_dir()}
    if args.final:
        assert existing == set(RUN_SPECS), (existing, set(RUN_SPECS))
    else:
        assert existing.issubset(RUN_SPECS), sorted(existing - set(RUN_SPECS))

    for name, spec in RUN_SPECS.items():
        run_dir = args.runs_root / name
        if not run_dir.is_dir():
            continue
        metadata = (run_dir / "run_metadata.txt").read_text(encoding="utf-8")
        assert f"git_commit={EXPECTED_COMMIT}" in metadata, name
        required_overrides = {
            "stream_manifest=/hy-tmp/datasets/manifests/chinese_lips_trainpool_dev7_revisit_188_011_036_seed42.jsonl",
            "decoder.beam_size=12",
            "decoder.ctc_weight=0.3",
            "decoder.nbest_size=10",
            "plasticity.mode=single_adapter",
            "plasticity.adapter_type=bottleneck",
            "plasticity.feedback_update_strategy=full_sequence",
            "plasticity.counterfactual_margin.margin=0.2",
            "plasticity.counterfactual_margin.weight=0.25",
            "plasticity.counterfactual_margin.rollback_tolerance=0.000001",
            f"plasticity.counterfactual_margin.enabled={str(spec['counterfactual']).lower()}",
            "feedback.strategy=periodic",
            "feedback.every=10",
            "seed=42",
        }
        assert all(value in metadata for value in required_overrides), name
        status = read_status(run_dir / "supervisor_status.tsv")
        assert status.get("attempt") == "1", name
        assert status.get("state") in {"running", "completed"}, name
        completed = status.get("state") == "completed"
        if args.final:
            assert completed, name
        log_text = (run_dir / "run.log").read_text(encoding="utf-8")
        assert ERROR_PATTERN.search(log_text) is None, name
        rows = read_jsonl(run_dir / "stream_results.jsonl")
        stats = check_stream_rows(name, spec, rows, manifest_rows)
        history = read_jsonl(run_dir / "metrics_history.jsonl")
        check_history(name, spec, history, len(rows), completed)
        pt_count = len(list(run_dir.rglob("*.pt")))
        assert pt_count <= 3, name
        summary_path = run_dir / "summary.json"
        if completed:
            assert summary_path.is_file(), name
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            check_summary(name, spec, summary, rows, run_dir, stats)
        report["runs"][name] = {
            "state": status.get("state"),
            "attempt": 1,
            "samples": len(rows),
            "history_rows": len(history),
            "feedback_queries": sum(
                bool(row["feedback_query"]["queried"]) for row in rows
            ),
            "valid_negatives": stats["valid_negatives"],
            "positive_violations_before": stats[
                "positive_violations_before"
            ],
            "pt_files": pt_count,
            "structured_errors": 0,
        }

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
