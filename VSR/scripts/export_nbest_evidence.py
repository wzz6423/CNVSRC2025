import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.artifacts import _write_jsonl_atomic


def build_parser():
    parser = argparse.ArgumentParser(
        description="从持续适应结果导出紧凑、可审计的 decoder N-best 证据"
    )
    parser.add_argument("--stream-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行不是有效 JSON：{path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象：{path}")
            records.append(record)
    return records


def _finite_number(value, source):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{source} 必须是有限数值")
    return float(value)


def _token_ids(value, source):
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise ValueError(f"{source} 必须是整数列表")
    return [int(token) for token in value]


def compact_nbest_evidence(records, top_k=10):
    top_k = int(top_k)
    if top_k < 2:
        raise ValueError("top-k 必须至少为 2")
    compact = []
    seen_uids = set()
    for expected_index, record in enumerate(records):
        if record.get("index") != expected_index:
            raise ValueError(
                f"stream index 不连续：期望 {expected_index}，实际 {record.get('index')}"
            )
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"样本 {expected_index} 缺少非空 uid")
        if uid in seen_uids:
            raise ValueError(f"uid 重复：{uid}")
        seen_uids.add(uid)
        target = record.get("target")
        one_best = record.get("transcript")
        if not isinstance(target, str) or not isinstance(one_best, str):
            raise ValueError(f"样本 {uid} 的 target/transcript 必须是字符串")
        hypotheses = record.get("decoder_nbest")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError(f"样本 {uid} 缺少 decoder_nbest")
        selected = []
        for rank, hypothesis in enumerate(hypotheses[:top_k], start=1):
            if not isinstance(hypothesis, dict) or hypothesis.get("rank") != rank:
                raise ValueError(f"样本 {uid} 的 N-best rank 不连续")
            transcript = hypothesis.get("transcript")
            scores = hypothesis.get("scores")
            if not isinstance(transcript, str) or not isinstance(scores, dict):
                raise ValueError(f"样本 {uid} 的 N-best transcript/scores 非法")
            selected.append(
                {
                    "rank": rank,
                    "transcript": transcript,
                    "tokens": _token_ids(
                        hypothesis.get("tokens"), f"样本 {uid} rank {rank} tokens"
                    ),
                    "score": _finite_number(
                        hypothesis.get("score"), f"样本 {uid} rank {rank} score"
                    ),
                    "normalized_score": _finite_number(
                        hypothesis.get("normalized_score"),
                        f"样本 {uid} rank {rank} normalized_score",
                    ),
                    "scores": {
                        str(name): _finite_number(
                            component, f"样本 {uid} rank {rank} scores.{name}"
                        )
                        for name, component in scores.items()
                    },
                }
            )
        if selected[0]["transcript"] != one_best:
            raise ValueError(f"样本 {uid} 的 rank-1 与已记录 transcript 不一致")
        one_best_tokens = _token_ids(
            record.get("decoder_tokens"), f"样本 {uid} decoder_tokens"
        )
        if selected[0]["tokens"] != one_best_tokens:
            raise ValueError(f"样本 {uid} 的 rank-1 与已记录 decoder_tokens 不一致")
        compact.append(
            {
                "schema_version": 1,
                "index": expected_index,
                "uid": uid,
                "domain": record.get("domain"),
                "target": target,
                "one_best": one_best,
                "one_best_tokens": one_best_tokens,
                "nbest": selected,
            }
        )
    if not compact:
        raise ValueError("stream_results 不能为空")
    return compact


def run(args):
    compact = compact_nbest_evidence(read_jsonl(args.stream_results), args.top_k)
    _write_jsonl_atomic(args.output, compact)
    counts = [len(record["nbest"]) for record in compact]
    return {
        "samples": len(compact),
        "requested_top_k": int(args.top_k),
        "minimum_hypotheses": min(counts),
        "maximum_hypotheses": max(counts),
        "output": str(args.output.resolve()),
    }


def main():
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
