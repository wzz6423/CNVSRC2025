import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.routing_diagnostics import summarize_route_records


def _parse_segment_lengths(value):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "回访段长必须依次提供 A1,B,C,A2 四个整数"
        )
    try:
        lengths = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("回访段长必须是整数") from error
    if any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("回访段长必须是正整数")
    return lengths


def main():
    parser = argparse.ArgumentParser(
        description="汇总流式实验的路由诊断指标"
    )
    parser.add_argument("--input", required=True, help="stream_results.jsonl 路径")
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="路由低相似度阈值"
    )
    parser.add_argument(
        "--revisit-segment-lengths",
        type=_parse_segment_lengths,
        metavar="A1,B,C,A2",
        help="可选的 A-B-C-A 四段长度；validation 为 130,195,186,130",
    )
    args = parser.parse_args()

    records = []
    try:
        with Path(args.input).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"JSONL 第 {line_number} 行为空")
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL 第 {line_number} 行无法解析: {error.msg}"
                    ) from error
        summary = summarize_route_records(
            records,
            threshold=args.threshold,
            segment_lengths=args.revisit_segment_lengths,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
