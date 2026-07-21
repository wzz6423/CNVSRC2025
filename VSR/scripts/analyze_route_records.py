import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.routing_diagnostics import summarize_route_records


def main():
    parser = argparse.ArgumentParser(
        description="汇总流式实验的路由诊断指标"
    )
    parser.add_argument("--input", required=True, help="stream_results.jsonl 路径")
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="路由低相似度阈值"
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
        summary = summarize_route_records(records, threshold=args.threshold)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
