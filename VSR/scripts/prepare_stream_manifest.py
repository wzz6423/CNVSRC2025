import argparse
import csv
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="生成持续适应流式清单")
    parser.add_argument("--csv", required=True, help="CNVSRC 四列 CSV")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument(
        "--domain-regex",
        help="从相对视频路径提取域；优先使用命名组 domain，其次使用第一个组",
    )
    parser.add_argument(
        "--order",
        choices=("original", "domain-block", "round-robin"),
        default="domain-block",
    )
    parser.add_argument("--shuffle-within-domain", action="store_true")
    parser.add_argument("--feedback-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def domain_from_path(relative_path, pattern):
    if pattern is None:
        parts = Path(relative_path).parts
        return parts[0] if parts else "unknown"
    match = pattern.search(relative_path)
    if match is None:
        return "unknown"
    if "domain" in match.groupdict():
        return match.group("domain")
    if match.groups():
        return match.group(1)
    return match.group(0)


def ordered_rows(rows, order, shuffle_within_domain, rng):
    if order == "original":
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    if shuffle_within_domain:
        for values in grouped.values():
            rng.shuffle(values)
    domains = sorted(grouped)
    if order == "domain-block":
        return [row for domain in domains for row in grouped[domain]]
    queues = {domain: deque(grouped[domain]) for domain in domains}
    output = []
    while any(queues.values()):
        for domain in domains:
            if queues[domain]:
                output.append(queues[domain].popleft())
    return output


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    pattern = re.compile(args.domain_regex) if args.domain_regex else None
    rows = []
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        for index, values in enumerate(csv.reader(handle)):
            if len(values) != 4:
                raise ValueError(f"第 {index + 1} 行不是四列 CNVSRC 清单")
            dataset, relative_path, _, token_string = values
            relative_without_suffix = Path(relative_path).with_suffix("").as_posix()
            rows.append(
                {
                    "uid": f"{dataset}:{relative_without_suffix}",
                    "video": str(Path(dataset) / relative_path),
                    "target_tokens": [int(token) for token in token_string.split()],
                    "domain": domain_from_path(relative_path, pattern),
                    "feedback": False,
                }
            )
    rows = ordered_rows(rows, args.order, args.shuffle_within_domain, rng)
    if args.feedback_every > 0:
        for index, row in enumerate(rows, start=1):
            row["feedback"] = index % args.feedback_every == 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写入 {len(rows)} 条流式样本：{output}")


if __name__ == "__main__":
    main()
