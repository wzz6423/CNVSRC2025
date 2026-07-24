#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.artifacts import _write_json_atomic
from plasticity.reliability import edit_distance


ERROR_PATTERN = re.compile(
    r"traceback|cuda out of memory|runtimeerror|exception|non.?finite",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="严格审计 N-best replay 与冻结基线的逐条等价性"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-samples", type=int, default=681)
    parser.add_argument("--expected-queries", type=int, default=68)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--history-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-gpu-hours", type=float, default=6.0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"不是有效 JSON：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"JSONL 第 {line_number} 行为空：{path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 第 {line_number} 行非法：{path}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象：{path}")
            records.append(value)
    return records


def read_status(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key] = value
    return values


def finite_number(value, source):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AssertionError(f"{source} 不是有限数值")
    return float(value)


def integer_tokens(value, source):
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise AssertionError(f"{source} 不是整数 token 列表")
    return value


def check_nbest(row, top_k):
    uid = row["uid"]
    hypotheses = row.get("decoder_nbest")
    assert isinstance(hypotheses, list) and len(hypotheses) == top_k, uid
    for rank, hypothesis in enumerate(hypotheses, start=1):
        assert isinstance(hypothesis, dict), (uid, rank)
        assert hypothesis.get("rank") == rank, (uid, rank)
        assert isinstance(hypothesis.get("transcript"), str), (uid, rank)
        integer_tokens(hypothesis.get("tokens"), f"{uid} rank {rank} tokens")
        finite_number(hypothesis.get("score"), f"{uid} rank {rank} score")
        finite_number(
            hypothesis.get("normalized_score"),
            f"{uid} rank {rank} normalized_score",
        )
        scores = hypothesis.get("scores")
        assert isinstance(scores, dict), (uid, rank)
        for name, value in scores.items():
            finite_number(value, f"{uid} rank {rank} scores.{name}")
    assert hypotheses[0]["transcript"] == row["transcript"], uid
    assert hypotheses[0]["tokens"] == row["decoder_tokens"], uid


def check_stream(candidate, baseline, manifest, top_k, expected_samples, final):
    if final:
        assert len(candidate) == expected_samples
    else:
        assert len(candidate) <= expected_samples
    assert len(baseline) == len(manifest) == expected_samples
    expected_prefix = manifest[: len(candidate)]
    baseline_prefix = baseline[: len(candidate)]
    assert [row["index"] for row in candidate] == list(range(len(candidate)))
    assert [row["uid"] for row in candidate] == [row["uid"] for row in expected_prefix]
    assert len({row["uid"] for row in candidate}) == len(candidate)

    equality_fields = (
        "uid",
        "domain",
        "target",
        "transcript",
        "decoder_tokens",
        "ctc_tokens",
        "feedback_used",
        "feedback_query",
        "adaptation_expert_index",
    )
    for index, (row, reference) in enumerate(
        zip(candidate, baseline_prefix, strict=True)
    ):
        assert row.get("index") == index
        for field in equality_fields:
            assert row.get(field) == reference.get(field), (index, field)
        update = row.get("update")
        reference_update = reference.get("update")
        assert isinstance(update, dict) and isinstance(reference_update, dict), index
        for field in ("status", "supervision", "reasons"):
            assert update.get(field) == reference_update.get(field), (index, field)
        assert update.get("status") != "failed", index
        check_nbest(row, top_k)

    candidate_edits = sum(
        edit_distance(list(row["transcript"]), list(row["target"])) for row in candidate
    )
    baseline_edits = sum(
        edit_distance(list(row["transcript"]), list(row["target"]))
        for row in baseline_prefix
    )
    characters = sum(len(row["target"]) for row in candidate)
    assert candidate_edits == baseline_edits
    return candidate_edits, characters


def expected_history_steps(processed, every, final):
    steps = list(range(every, processed + 1, every))
    if final and (not steps or steps[-1] != processed):
        steps.append(processed)
    return steps


def check_history(history, processed, every, checkpoint_every, final):
    steps = [row.get("processed_samples") for row in history]
    assert steps == expected_history_steps(processed, every, final)
    for row in history:
        step = row["processed_samples"]
        expected_checkpoint = step % checkpoint_every == 0 or (
            final and step == processed
        )
        assert row.get("checkpoint") is expected_checkpoint, step


def check_final_artifacts(args, summary, baseline_summary):
    assert summary["samples"] == args.expected_samples
    assert summary["cer"] == baseline_summary["cer"]
    assert summary["edits"] == baseline_summary["edits"]
    assert summary["characters"] == baseline_summary["characters"]
    assert summary["feedback_query"]["total_queries"] == args.expected_queries
    assert summary["feedback_query"]["policy_queries"] == args.expected_queries
    current_state = summary["stream_state"]
    baseline_state = baseline_summary["stream_state"]
    for key, value in baseline_state.items():
        if key != "experiment_config_sha256":
            assert current_state[key] == value, key
    assert SHA256_PATTERN.fullmatch(current_state["experiment_config_sha256"])
    assert (
        current_state["experiment_config_sha256"]
        != baseline_state["experiment_config_sha256"]
    )
    assert summary["timing"]["total_seconds"] <= args.max_gpu_hours * 3600

    checkpoint_path = args.run_dir / "adaptation_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 3
    assert checkpoint["processed_samples"] == args.expected_samples
    assert checkpoint["stream_state"] == current_state
    index = read_json(args.run_dir / "best_checkpoints" / "index.json")
    assert index["schema_version"] == 1
    assert len(index["checkpoints"]) == 2
    for entry in index["checkpoints"]:
        best_path = args.run_dir / "best_checkpoints" / entry["path"]
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        assert best["version"] == 3
        assert best["processed_samples"] == entry["processed_samples"]
        assert best["stream_state"] == current_state
    assert len(list(args.run_dir.rglob("*.pt"))) == 3


def run(args):
    assert COMMIT_PATTERN.fullmatch(args.expected_commit)
    assert args.expected_samples > 0
    assert args.expected_queries >= 0
    assert args.top_k >= 2
    assert args.history_every > 0 and args.checkpoint_every > 0
    assert math.isfinite(args.max_gpu_hours) and args.max_gpu_hours > 0

    status = read_status(args.run_dir / "supervisor_status.tsv")
    assert status.get("attempt") == "1"
    if args.final:
        assert status.get("state") == "completed"
    else:
        assert status.get("state") in {"running", "completed"}
    metadata = (args.run_dir / "run_metadata.txt").read_text(encoding="utf-8")
    assert f"git_commit={args.expected_commit}" in metadata
    assert f"decoder.nbest_size={args.top_k}" in metadata
    log_text = (args.run_dir / "run.log").read_text(encoding="utf-8")
    assert ERROR_PATTERN.search(log_text) is None

    manifest = read_jsonl(args.manifest)
    baseline = read_jsonl(args.baseline_dir / "stream_results.jsonl")
    candidate = read_jsonl(args.run_dir / "stream_results.jsonl")
    edits, characters = check_stream(
        candidate,
        baseline,
        manifest,
        args.top_k,
        args.expected_samples,
        args.final,
    )
    history = read_jsonl(args.run_dir / "metrics_history.jsonl")
    check_history(
        history,
        len(candidate),
        args.history_every,
        args.checkpoint_every,
        args.final,
    )
    assert len(list(args.run_dir.rglob("*.pt"))) <= 3

    summary = None
    if args.final:
        summary = read_json(args.run_dir / "summary.json")
        baseline_summary = read_json(args.baseline_dir / "summary.json")
        check_final_artifacts(args, summary, baseline_summary)
        assert edits == summary["edits"]
        assert characters == summary["characters"]

    report = {
        "schema_version": 1,
        "status": "AUDIT_OK",
        "final": bool(args.final),
        "run": str(args.run_dir.resolve()),
        "baseline": str(args.baseline_dir.resolve()),
        "expected_commit": args.expected_commit,
        "samples": len(candidate),
        "queries": sum(bool(row["feedback_used"]) for row in candidate),
        "history_rows": len(history),
        "pt_files": len(list(args.run_dir.rglob("*.pt"))),
        "top_k": args.top_k,
        "rank1_replay_equivalent": True,
        "edits": edits,
        "characters": characters,
        "cer": edits / characters if characters else None,
        "gpu_hours": (summary["timing"]["total_seconds"] / 3600 if summary else None),
    }
    if args.final:
        assert report["queries"] == args.expected_queries
    if args.output:
        _write_json_atomic(args.output, report)
    return report


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
