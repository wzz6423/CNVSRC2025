import math

import numpy as np

from .reliability import edit_distance


REVISIT_SEGMENT_NAMES = ("A1", "B", "C", "A2")
DEFAULT_REVISIT_SEGMENT_LENGTHS = (133, 255, 222, 132)


def _text_pair(record, position):
    if not isinstance(record, dict):
        raise ValueError(f"第 {position} 条结果必须是 JSON 对象")
    target = record.get("target")
    transcript = record.get("transcript")
    if not isinstance(target, str) or not isinstance(transcript, str):
        raise ValueError(
            f"第 {position} 条结果的 target 和 transcript 必须是字符串"
        )
    return target, transcript


def aggregate_character_cer(records):
    records = list(records)
    edits = 0
    characters = 0
    for position, record in enumerate(records, start=1):
        target, transcript = _text_pair(record, position)
        edits += edit_distance(list(transcript), list(target))
        characters += len(target)
    return {
        "samples": len(records),
        "edits": edits,
        "characters": characters,
        "cer": edits / characters if characters else None,
    }


def _revisit_segments(segment_lengths):
    segment_lengths = tuple(segment_lengths)
    if len(segment_lengths) != len(REVISIT_SEGMENT_NAMES):
        raise ValueError("回访段长必须依次提供 A1/B/C/A2 四个整数")
    if any(
        not isinstance(length, int) or isinstance(length, bool) or length < 1
        for length in segment_lengths
    ):
        raise ValueError("回访段长必须是四个正整数")
    segments = []
    start = 0
    for name, length in zip(REVISIT_SEGMENT_NAMES, segment_lengths):
        end = start + length
        segments.append((name, start, end))
        start = end
    return tuple(segments)


def aggregate_revisit_segments(
    records, *, segment_lengths=DEFAULT_REVISIT_SEGMENT_LENGTHS
):
    records = list(records)
    segments = _revisit_segments(segment_lengths)
    expected_samples = segments[-1][2]
    if len(records) != expected_samples:
        raise ValueError(
            "A-B-C-A 回访结果必须恰好包含 "
            f"{expected_samples} 条，实际为 {len(records)} 条"
        )
    seen_uids = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"第 {position + 1} 条结果必须是 JSON 对象")
        if record.get("index") != position:
            raise ValueError(
                "A-B-C-A 回访结果 index 必须从 0 开始连续递增"
            )
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"第 {position + 1} 条结果缺少非空 uid")
        if uid in seen_uids:
            raise ValueError(f"A-B-C-A 回访结果包含重复 uid：{uid}")
        seen_uids.add(uid)
    return {
        name: aggregate_character_cer(records[start:end])
        for name, start, end in segments
    }


def _segment_cer(segments, name, source):
    try:
        value = segments[name]["cer"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source} 缺少 {name} 段 CER") from error
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{source} 的 {name} 段 CER 必须是数值")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{source} 的 {name} 段 CER 必须是非负有限数值")
    return value


def static_corrected_forgetting(method_segments, static_segments):
    method_delta = _segment_cer(method_segments, "A2", "method") - _segment_cer(
        method_segments, "A1", "method"
    )
    static_delta = _segment_cer(static_segments, "A2", "static") - _segment_cer(
        static_segments, "A1", "static"
    )
    return {
        "method_a2_minus_a1": method_delta,
        "static_a2_minus_a1": static_delta,
        "static_corrected": method_delta - static_delta,
    }


def _records_by_uid(records, source):
    indexed = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{source} 第 {position} 条结果必须是 JSON 对象")
        uid = record.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{source} 第 {position} 条结果缺少非空 uid")
        if uid in indexed:
            raise ValueError(f"{source} 包含重复 uid：{uid}")
        indexed[uid] = record
    return indexed


def _paired_edit_arrays(candidate_records, baseline_records, source):
    candidate_by_uid = _records_by_uid(candidate_records, f"{source} candidate")
    baseline_by_uid = _records_by_uid(baseline_records, f"{source} baseline")
    if set(candidate_by_uid) != set(baseline_by_uid):
        raise ValueError(
            f"{source} candidate 与 baseline 的 uid 集合必须完全一致"
        )

    edit_differences = []
    characters = []
    for position, candidate in enumerate(candidate_records, start=1):
        uid = candidate["uid"]
        baseline = baseline_by_uid[uid]
        candidate_target, candidate_text = _text_pair(candidate, position)
        baseline_target, baseline_text = _text_pair(baseline, position)
        if candidate_target != baseline_target:
            raise ValueError(
                f"{source} uid={uid} 的 candidate 与 baseline target 不一致"
            )
        if not candidate_target:
            raise ValueError(f"{source} uid={uid} 的 target 不能为空")
        edit_differences.append(
            edit_distance(list(candidate_text), list(candidate_target))
            - edit_distance(list(baseline_text), list(baseline_target))
        )
        characters.append(len(candidate_target))
    return (
        np.asarray(edit_differences, dtype=np.int64),
        np.asarray(characters, dtype=np.int64),
    )


def paired_bootstrap_cer_difference(
    candidate_records,
    baseline_records,
    *,
    iterations=10000,
    seed=42,
    batch_size=256,
):
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    if not candidate_records:
        raise ValueError("配对 bootstrap 至少需要一条结果")
    iterations = int(iterations)
    batch_size = int(batch_size)
    if iterations < 1 or batch_size < 1:
        raise ValueError("bootstrap iterations 和 batch_size 必须大于 0")
    edit_differences, characters = _paired_edit_arrays(
        candidate_records, baseline_records, "overall"
    )
    point_estimate = float(edit_differences.sum() / characters.sum())
    generator = np.random.default_rng(int(seed))
    samples = np.empty(iterations, dtype=np.float64)
    written = 0
    while written < iterations:
        current_batch = min(batch_size, iterations - written)
        indices = generator.integers(
            0, len(candidate_records), size=(current_batch, len(candidate_records))
        )
        sampled_edits = edit_differences[indices].sum(axis=1)
        sampled_characters = characters[indices].sum(axis=1)
        samples[written : written + current_batch] = (
            sampled_edits / sampled_characters
        )
        written += current_batch
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "candidate_minus_baseline": point_estimate,
        "ci_95": {"lower": float(lower), "upper": float(upper)},
        "iterations": iterations,
        "seed": int(seed),
        "paired_samples": len(candidate_records),
    }


def paired_bootstrap_revisit_forgetting_difference(
    candidate_records,
    baseline_records,
    *,
    segment_lengths=DEFAULT_REVISIT_SEGMENT_LENGTHS,
    iterations=10000,
    seed=42,
    batch_size=256,
):
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    segments = _revisit_segments(segment_lengths)
    expected_samples = segments[-1][2]
    for source, records in (
        ("candidate", candidate_records),
        ("baseline", baseline_records),
    ):
        if len(records) != expected_samples:
            raise ValueError(
                f"{source} 回访结果必须恰好包含 {expected_samples} 条，"
                f"实际为 {len(records)} 条"
            )
    iterations = int(iterations)
    batch_size = int(batch_size)
    if iterations < 1 or batch_size < 1:
        raise ValueError("bootstrap iterations 和 batch_size 必须大于 0")

    segment_bounds = {name: (start, end) for name, start, end in segments}
    paired = {}
    for name in ("A1", "A2"):
        start, end = segment_bounds[name]
        paired[name] = _paired_edit_arrays(
            candidate_records[start:end],
            baseline_records[start:end],
            name,
        )

    point_by_segment = {
        name: float(edits.sum() / characters.sum())
        for name, (edits, characters) in paired.items()
    }
    point = point_by_segment["A2"] - point_by_segment["A1"]

    seed_sequence = np.random.SeedSequence(int(seed))
    a1_generator, a2_generator = (
        np.random.default_rng(child) for child in seed_sequence.spawn(2)
    )
    samples = np.empty(iterations, dtype=np.float64)
    written = 0
    while written < iterations:
        current_batch = min(batch_size, iterations - written)
        segment_samples = {}
        for name, generator in (
            ("A1", a1_generator),
            ("A2", a2_generator),
        ):
            edits, characters = paired[name]
            indices = generator.integers(
                0, len(edits), size=(current_batch, len(edits))
            )
            segment_samples[name] = (
                edits[indices].sum(axis=1) / characters[indices].sum(axis=1)
            )
        samples[written : written + current_batch] = (
            segment_samples["A2"] - segment_samples["A1"]
        )
        written += current_batch

    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "point": point,
        "ci_95": {"lower": float(lower), "upper": float(upper)},
        "iterations": iterations,
        "seed": int(seed),
        "paired_samples": {
            name: len(paired[name][0]) for name in ("A1", "A2")
        },
    }


def summarize_seed_cers(values):
    values = list(values)
    if not values:
        raise ValueError("至少需要一个 seed CER")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in values
    ):
        raise ValueError("seed CER 必须是非负有限数值")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("seed CER 必须是非负有限数值")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "values": values,
        "mean": mean,
        "population_variance": variance,
        "population_std": math.sqrt(variance),
    }
