import argparse
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.analysis import (
    DEFAULT_REVISIT_SEGMENT_LENGTHS,
    REVISIT_SEGMENT_NAMES,
    aggregate_character_cer,
    aggregate_revisit_segments,
    paired_bootstrap_cer_difference,
    paired_bootstrap_revisit_forgetting_difference,
    static_corrected_forgetting,
    summarize_seed_cers,
)
from plasticity.artifacts import _write_json_atomic


_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parse_segment_lengths(value):
    parts = value.split(",")
    if len(parts) != len(REVISIT_SEGMENT_NAMES):
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="聚合持续适应实验 CER、回访遗忘与配对置信区间"
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="实验名称与包含 summary.json/stream_results.jsonl 的目录",
    )
    parser.add_argument(
        "--revisit",
        action="append",
        default=[],
        metavar="NAME",
        help="按固定 A1/B/C/A2 分段分析的实验，可重复指定",
    )
    parser.add_argument(
        "--static-revisit",
        metavar="NAME",
        help="用于校正回访遗忘的 static 实验名称",
    )
    parser.add_argument(
        "--revisit-segment-lengths",
        type=_parse_segment_lengths,
        default=DEFAULT_REVISIT_SEGMENT_LENGTHS,
        metavar="A1,B,C,A2",
        help="回访四段长度，测试集默认 133,255,222,132",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="CANDIDATE:BASELINE",
        help=(
            "按 UID 配对计算 candidate-baseline CER 与 95%% CI；"
            "双方为回访实验时同时计算 A2-A1 遗忘差"
        ),
    )
    parser.add_argument(
        "--seed-group",
        action="append",
        default=[],
        metavar="GROUP=RUN1,RUN2,RUN3",
        help="恰好三个实验组成的 seed 统计组",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-batch-size", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _validate_name(name, source):
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"{source} 名称包含非法字符：{name}"
        )
    return name


def _parse_assignment(value, source):
    if "=" not in value:
        raise ValueError(f"{source} 必须使用 NAME=VALUE 格式：{value}")
    name, assigned = value.split("=", 1)
    _validate_name(name, source)
    if not assigned:
        raise ValueError(f"{source} 的值不能为空：{value}")
    return name, assigned


def _read_json(path, source):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到 {source}：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} 不是有效 JSON：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} 必须是 JSON 对象：{path}")
    return value


def _read_jsonl(path):
    records = []
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到流式结果：{path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"流式结果第 {line_number} 行不是有效 JSON：{path}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"流式结果第 {line_number} 行必须是 JSON 对象：{path}"
                )
            records.append(record)
    return records


def _load_experiment(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"实验目录不存在：{directory}")
    records = _read_jsonl(directory / "stream_results.jsonl")
    summary = _read_json(directory / "summary.json", "实验 summary")
    overall = aggregate_character_cer(records)
    if overall["cer"] is None:
        raise ValueError(f"流式结果无法计算有限 CER：{directory}")
    for name in ("samples", "edits", "characters"):
        value = summary.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"实验 summary 的 {name} 必须是非负整数：{directory}"
            )
        if value != overall[name]:
            raise ValueError(
                f"summary {name}={value} 与流式结果聚合值="
                f"{overall[name]} 不一致：{directory}"
            )
    summary_cer = summary.get("cer")
    if (
        not isinstance(summary_cer, (int, float))
        or isinstance(summary_cer, bool)
        or not math.isfinite(float(summary_cer))
        or summary_cer < 0
    ):
        raise ValueError(
            f"实验 summary 的 cer 必须是非负有限数值：{directory}"
        )
    if not math.isclose(
        float(summary_cer), overall["cer"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"summary cer={summary_cer} 与流式结果聚合值="
            f"{overall['cer']} 不一致：{directory}"
        )
    return directory, records, summary, overall


def _require_experiment(name, experiments, source):
    if name not in experiments:
        raise ValueError(f"{source} 引用了未声明的实验：{name}")
    return experiments[name]


def run(args):
    experiments = {}
    records_by_name = {}
    for specification in args.experiment:
        name, directory_value = _parse_assignment(specification, "experiment")
        if name in experiments:
            raise ValueError(f"实验名称重复：{name}")
        directory, records, summary, overall = _load_experiment(directory_value)
        records_by_name[name] = records
        experiments[name] = {
            "directory": str(directory),
            "overall": overall,
            "run_summary": summary,
        }

    revisit_names = []
    for name in args.revisit:
        _validate_name(name, "revisit")
        _require_experiment(name, experiments, "revisit")
        if name in revisit_names:
            raise ValueError(f"revisit 名称重复：{name}")
        revisit_names.append(name)
        experiments[name]["segments"] = aggregate_revisit_segments(
            records_by_name[name],
            segment_lengths=args.revisit_segment_lengths,
        )

    if args.static_revisit:
        static_name = _validate_name(args.static_revisit, "static-revisit")
        if static_name not in revisit_names:
            raise ValueError("static-revisit 必须同时通过 --revisit 声明")
        static_segments = experiments[static_name]["segments"]
        for name in revisit_names:
            experiments[name]["forgetting"] = static_corrected_forgetting(
                experiments[name]["segments"], static_segments
            )

    comparisons = {}
    for specification in args.comparison:
        if specification.count(":") != 1:
            raise ValueError(
                f"comparison 必须使用 CANDIDATE:BASELINE 格式：{specification}"
            )
        candidate, baseline = specification.split(":")
        _validate_name(candidate, "comparison candidate")
        _validate_name(baseline, "comparison baseline")
        _require_experiment(candidate, experiments, "comparison")
        _require_experiment(baseline, experiments, "comparison")
        comparison_name = f"{candidate}_minus_{baseline}"
        if comparison_name in comparisons:
            raise ValueError(f"comparison 重复：{specification}")
        comparison = {
            "candidate": candidate,
            "baseline": baseline,
            **paired_bootstrap_cer_difference(
                records_by_name[candidate],
                records_by_name[baseline],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
                batch_size=args.bootstrap_batch_size,
            ),
        }
        if candidate in revisit_names and baseline in revisit_names:
            comparison["revisit_forgetting_difference"] = (
                paired_bootstrap_revisit_forgetting_difference(
                    records_by_name[candidate],
                    records_by_name[baseline],
                    segment_lengths=args.revisit_segment_lengths,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                    batch_size=args.bootstrap_batch_size,
                )
            )
        comparisons[comparison_name] = comparison

    seed_groups = {}
    for specification in args.seed_group:
        group, names_value = _parse_assignment(specification, "seed-group")
        if group in seed_groups:
            raise ValueError(f"seed-group 名称重复：{group}")
        names = names_value.split(",")
        if len(names) != 3 or len(set(names)) != 3:
            raise ValueError(
                f"seed-group 必须包含三个不同实验：{specification}"
            )
        for name in names:
            _validate_name(name, "seed-group experiment")
            _require_experiment(name, experiments, "seed-group")
        statistics = summarize_seed_cers(
            [experiments[name]["overall"]["cer"] for name in names]
        )
        seed_groups[group] = {"experiments": names, **statistics}

    return {
        "schema_version": 1,
        "experiments": experiments,
        "comparisons": comparisons,
        "seed_groups": seed_groups,
        "revisit_protocol": {
            "segment_lengths": dict(
                zip(REVISIT_SEGMENT_NAMES, args.revisit_segment_lengths)
            )
        },
        "bootstrap": {
            "iterations": int(args.bootstrap_iterations),
            "seed": int(args.bootstrap_seed),
            "batch_size": int(args.bootstrap_batch_size),
        },
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    analysis = run(args)
    _write_json_atomic(args.output, analysis)
    print(f"分析结果已写入：{args.output}")


if __name__ == "__main__":
    main()
