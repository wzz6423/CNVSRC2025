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


def summarize_feedback_queries(records):
    records = list(records)
    available = ["feedback_query" in record for record in records]
    if not any(available):
        return {"available": False}
    if not all(available):
        raise ValueError("反馈查询审计字段只能全部存在或全部缺失")

    queried_records = []
    queried_errors = 0
    nonqueried_errors = 0
    policy_queries = 0
    manifest_queries = 0
    sources = {}
    reasons = {}
    policy_queries_by_block = {}
    queried_reliability = []
    nonqueried_reliability = []
    for position, record in enumerate(records, start=1):
        query = record["feedback_query"]
        if not isinstance(query, dict):
            raise ValueError(f"第 {position} 条 feedback_query 必须是 JSON 对象")
        boolean_names = (
            "queried",
            "policy_requested",
            "manifest_requested",
        )
        if any(not isinstance(query.get(name), bool) for name in boolean_names):
            raise ValueError(f"第 {position} 条反馈查询布尔字段无效")
        queried = query["queried"]
        if queried != (
            query["policy_requested"] or query["manifest_requested"]
        ):
            raise ValueError(f"第 {position} 条反馈查询决策不守恒")
        if record.get("feedback_used") is not queried:
            raise ValueError(f"第 {position} 条 feedback_used 与查询决策不一致")
        source = query.get("source")
        reason = query.get("reason")
        if not isinstance(source, str) or not isinstance(reason, str):
            raise ValueError(f"第 {position} 条反馈查询来源或原因无效")
        sources[source] = sources.get(source, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1

        reliability = record.get("reliability")
        score = reliability.get("score") if isinstance(reliability, dict) else None
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"第 {position} 条可靠度分数无效")
        target, transcript = _text_pair(record, position)
        has_error = edit_distance(list(transcript), list(target)) > 0
        if queried:
            queried_records.append(record)
            queried_errors += int(has_error)
            queried_reliability.append(float(score))
        else:
            nonqueried_errors += int(has_error)
            nonqueried_reliability.append(float(score))

        if query["policy_requested"]:
            policy_queries += 1
            block_index = query.get("block_index")
            if (
                not isinstance(block_index, int)
                or isinstance(block_index, bool)
                or block_index < 0
            ):
                raise ValueError(f"第 {position} 条策略查询缺少有效 block_index")
            policy_queries_by_block[block_index] = (
                policy_queries_by_block.get(block_index, 0) + 1
            )
        manifest_queries += int(query["manifest_requested"])

    query_count = len(queried_records)
    nonquery_count = len(records) - query_count
    query_cer = aggregate_character_cer(queried_records)
    return {
        "available": True,
        "samples": len(records),
        "queries": query_count,
        "policy_queries": policy_queries,
        "manifest_queries": manifest_queries,
        "query_rate": query_count / len(records) if records else 0.0,
        "queried_true_errors": queried_errors,
        "queried_true_error_rate": (
            queried_errors / query_count if query_count else None
        ),
        "nonqueried_true_errors": nonqueried_errors,
        "nonqueried_true_error_rate": (
            nonqueried_errors / nonquery_count if nonquery_count else None
        ),
        "queried_cer": query_cer["cer"],
        "queried_edits": query_cer["edits"],
        "queried_characters": query_cer["characters"],
        "mean_queried_reliability": (
            math.fsum(queried_reliability) / query_count
            if query_count
            else None
        ),
        "mean_nonqueried_reliability": (
            math.fsum(nonqueried_reliability) / nonquery_count
            if nonquery_count
            else None
        ),
        "sources": sources,
        "reasons": reasons,
        "policy_query_blocks": len(policy_queries_by_block),
        "max_policy_queries_per_block": max(
            policy_queries_by_block.values(), default=0
        ),
    }


def summarize_runtime_resources(records):
    records = list(records)
    runtime_available = ["runtime" in record for record in records]
    if not any(runtime_available):
        return {"available": False}
    if not all(runtime_available):
        raise ValueError("runtime 审计字段只能全部存在或全部缺失")

    fields = ("video_load_seconds", "process_seconds", "total_seconds")
    values = {name: [] for name in fields}
    gpu_peaks = []
    for position, record in enumerate(records, start=1):
        runtime = record["runtime"]
        if not isinstance(runtime, dict):
            raise ValueError(f"第 {position} 条 runtime 必须是 JSON 对象")
        for name in fields:
            value = runtime.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"第 {position} 条 runtime.{name} 无效")
            values[name].append(float(value))
        peak = runtime.get("gpu_max_memory_allocated_bytes")
        if peak is not None:
            if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
                raise ValueError(f"第 {position} 条 GPU 峰值显存无效")
            gpu_peaks.append(peak)

    statistics = {}
    for name, field_values in values.items():
        array = np.asarray(field_values, dtype=np.float64)
        statistics[name] = {
            "total": float(array.sum()),
            "mean": float(array.mean()) if len(array) else None,
            "p50": float(np.quantile(array, 0.5)) if len(array) else None,
            "p95": float(np.quantile(array, 0.95)) if len(array) else None,
            "max": float(array.max()) if len(array) else None,
        }
    return {
        "available": True,
        "samples": len(records),
        "timing": statistics,
        "peak_gpu_memory_allocated_bytes": max(gpu_peaks, default=None),
    }


def summarize_feedback_corrections(records):
    feedback_samples = 0
    diagnosed_samples = 0
    totals = {
        "predicted_tokens": 0,
        "target_tokens": 0,
        "matched_tokens": 0,
        "substituted_tokens": 0,
        "missing_target_tokens": 0,
        "extra_prediction_tokens": 0,
    }
    token_error_rates = []
    matched_frame_rates = []
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {position} 条结果必须是 JSON 对象")
        if not record.get("feedback_used", False):
            continue
        feedback_samples += 1
        update = record.get("update")
        correction = update.get("correction") if isinstance(update, dict) else None
        if correction is None:
            continue
        if not isinstance(correction, dict):
            raise ValueError(f"第 {position} 条结果的 correction 必须是 JSON 对象")
        for name in totals:
            value = correction.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"第 {position} 条 correction.{name} 必须是非负整数")
            totals[name] += value
        rates = []
        for name in ("token_error_rate", "matched_frame_rate"):
            value = correction.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"第 {position} 条 correction.{name} 必须是非负有限数值")
            rates.append(float(value))
        token_error_rates.append(rates[0])
        matched_frame_rates.append(rates[1])
        diagnosed_samples += 1
    return {
        "feedback_samples": feedback_samples,
        "diagnosed_feedback_samples": diagnosed_samples,
        "diagnostic_coverage": (
            diagnosed_samples / feedback_samples if feedback_samples else 0.0
        ),
        **totals,
        "mean_token_error_rate": (
            math.fsum(token_error_rates) / diagnosed_samples
            if diagnosed_samples
            else None
        ),
        "mean_matched_frame_rate": (
            math.fsum(matched_frame_rates) / diagnosed_samples
            if diagnosed_samples
            else None
        ),
    }


def summarize_localized_feedback_updates(records):
    feedback_samples = 0
    localized_samples = 0
    randomized_samples = 0
    strategies = {}
    update_statuses = {}
    integer_totals = {
        "ctc_frames": 0,
        "matched_target_tokens": 0,
        "error_target_tokens": 0,
        "substitution_target_tokens": 0,
        "deletion_target_tokens": 0,
        "insertion_frames": 0,
    }
    measure_names = (
        "target_log_likelihood",
        "matched_occupancy_mass",
        "error_occupancy_mass",
        "matched_effective_frames",
        "error_effective_frames",
    )
    measures = {name: [] for name in measure_names}
    objective_before_values = []
    objective_after_values = []

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {position} 条结果必须是 JSON 对象")
        if not record.get("feedback_used", False):
            continue
        feedback_samples += 1
        update = record.get("update")
        localization = update.get("localization") if isinstance(update, dict) else None
        if localization is None:
            continue
        if not isinstance(localization, dict):
            raise ValueError(f"第 {position} 条 update.localization 必须是 JSON 对象")

        strategy = localization.get("strategy")
        randomized = localization.get("randomized_support")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError(f"第 {position} 条 localization.strategy 必须是非空字符串")
        if not isinstance(randomized, bool):
            raise ValueError(f"第 {position} 条 localization.randomized_support 必须是布尔值")
        strategies[strategy] = strategies.get(strategy, 0) + 1
        randomized_samples += int(randomized)

        for name in integer_totals:
            value = localization.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if name == "ctc_frames" else 0)
            ):
                requirement = "正整数" if name == "ctc_frames" else "非负整数"
                raise ValueError(
                    f"第 {position} 条 localization.{name} 必须是{requirement}"
                )
            integer_totals[name] += value

        for name in measure_names:
            value = localization.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or (name != "target_log_likelihood" and value < 0)
            ):
                raise ValueError(
                    f"第 {position} 条 localization.{name} 必须是有限数值"
                )
            measures[name].append(float(value))

        status = update.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"第 {position} 条 update.status 必须是非空字符串")
        update_statuses[status] = update_statuses.get(status, 0) + 1

        objective_before = update.get("objective_before")
        objective_after = update.get("objective_after")
        if (objective_before is None) != (objective_after is None):
            raise ValueError(
                f"第 {position} 条局部目标 before/after 必须同时存在或同时为空"
            )
        if objective_before is not None:
            pair = (objective_before, objective_after)
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in pair
            ):
                raise ValueError(f"第 {position} 条局部目标必须是有限数值")
            objective_before_values.append(float(objective_before))
            objective_after_values.append(float(objective_after))
        localized_samples += 1

    objective_samples = len(objective_before_values)
    objective_deltas = [
        after - before
        for before, after in zip(objective_before_values, objective_after_values)
    ]
    return {
        "feedback_samples": feedback_samples,
        "localized_feedback_samples": localized_samples,
        "localization_coverage": (
            localized_samples / feedback_samples if feedback_samples else 0.0
        ),
        "strategies": strategies,
        "randomized_support_samples": randomized_samples,
        "update_statuses": update_statuses,
        **integer_totals,
        "insertion_frame_coverage": (
            integer_totals["insertion_frames"] / integer_totals["ctc_frames"]
            if integer_totals["ctc_frames"]
            else None
        ),
        **{
            f"mean_{name}": (
                math.fsum(values) / localized_samples if localized_samples else None
            )
            for name, values in measures.items()
        },
        "objective_samples": objective_samples,
        "mean_objective_before": (
            math.fsum(objective_before_values) / objective_samples
            if objective_samples
            else None
        ),
        "mean_objective_after": (
            math.fsum(objective_after_values) / objective_samples
            if objective_samples
            else None
        ),
        "mean_objective_delta": (
            math.fsum(objective_deltas) / objective_samples
            if objective_samples
            else None
        ),
        "objective_improved_samples": sum(delta < 0 for delta in objective_deltas),
    }


def feedback_followup_records(records, horizon):
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("反馈后窗口必须大于 0")
    selected = []
    remaining = 0
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"第 {position + 1} 条结果必须是 JSON 对象")
        if record.get("index") != position:
            raise ValueError("反馈后窗口要求 index 从 0 开始连续递增")
        if remaining:
            selected.append(record)
            remaining -= 1
        if record.get("feedback_used", False):
            remaining = horizon
    return selected


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


def paired_feedback_followup_records(candidate_records, baseline_records, horizon):
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    candidate_followup = feedback_followup_records(candidate_records, horizon)
    baseline_followup = feedback_followup_records(baseline_records, horizon)
    if candidate_followup and baseline_followup:
        if {record["uid"] for record in candidate_followup} != {
            record["uid"] for record in baseline_followup
        }:
            raise ValueError("candidate 与 baseline 的反馈后窗口必须覆盖相同 uid")
        baseline_by_uid = _records_by_uid(baseline_followup, "baseline followup")
        return candidate_followup, [
            baseline_by_uid[record["uid"]] for record in candidate_followup
        ]
    if candidate_followup:
        baseline_by_uid = _records_by_uid(baseline_records, "baseline")
        return candidate_followup, [
            baseline_by_uid[record["uid"]] for record in candidate_followup
        ]
    if baseline_followup:
        candidate_by_uid = _records_by_uid(candidate_records, "candidate")
        return (
            [candidate_by_uid[record["uid"]] for record in baseline_followup],
            baseline_followup,
        )
    return [], []


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


def paired_sample_edit_transitions(candidate_records, baseline_records):
    candidate_records = list(candidate_records)
    baseline_records = list(baseline_records)
    edit_differences, _ = _paired_edit_arrays(
        candidate_records, baseline_records, "transitions"
    )
    return {
        "paired_samples": len(edit_differences),
        "candidate_better": int((edit_differences < 0).sum()),
        "same": int((edit_differences == 0).sum()),
        "candidate_worse": int((edit_differences > 0).sum()),
        "net_edit_difference": int(edit_differences.sum()),
    }


def _query_errors_by_block(records, source):
    errors = {}
    for position, record in enumerate(records, start=1):
        query = record.get("feedback_query")
        if not isinstance(query, dict) or not query.get("policy_requested", False):
            continue
        block_index = query.get("block_index")
        if (
            not isinstance(block_index, int)
            or isinstance(block_index, bool)
            or block_index < 0
        ):
            raise ValueError(f"{source} 第 {position} 条策略查询 block_index 无效")
        if block_index in errors:
            raise ValueError(f"{source} 第 {block_index} 个窗口包含多次策略查询")
        target, transcript = _text_pair(record, position)
        errors[block_index] = int(
            edit_distance(list(transcript), list(target)) > 0
        )
    if not errors:
        raise ValueError(f"{source} 没有可比较的策略查询")
    return errors


def paired_bootstrap_query_error_rate_difference(
    candidate_records,
    baseline_records,
    *,
    iterations=10000,
    seed=42,
    batch_size=256,
):
    candidate = _query_errors_by_block(candidate_records, "candidate")
    baseline = _query_errors_by_block(baseline_records, "baseline")
    if set(candidate) != set(baseline):
        raise ValueError("candidate 与 baseline 的反馈查询窗口必须完全一致")
    iterations = int(iterations)
    batch_size = int(batch_size)
    if iterations < 1 or batch_size < 1:
        raise ValueError("bootstrap iterations 和 batch_size 必须大于 0")

    blocks = sorted(candidate)
    differences = np.asarray(
        [candidate[block] - baseline[block] for block in blocks],
        dtype=np.int64,
    )
    point = float(differences.mean())
    generator = np.random.default_rng(int(seed))
    samples = np.empty(iterations, dtype=np.float64)
    written = 0
    while written < iterations:
        current_batch = min(batch_size, iterations - written)
        indices = generator.integers(
            0, len(blocks), size=(current_batch, len(blocks))
        )
        samples[written : written + current_batch] = differences[indices].mean(
            axis=1
        )
        written += current_batch
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "candidate_minus_baseline": point,
        "ci_95": {"lower": float(lower), "upper": float(upper)},
        "candidate_error_rate": math.fsum(candidate.values()) / len(blocks),
        "baseline_error_rate": math.fsum(baseline.values()) / len(blocks),
        "iterations": iterations,
        "seed": int(seed),
        "paired_blocks": len(blocks),
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
