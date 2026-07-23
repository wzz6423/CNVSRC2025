import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.analysis import (
    aggregate_character_cer,
    aggregate_revisit_segments,
    feedback_followup_records,
    paired_bootstrap_cer_difference,
    paired_bootstrap_revisit_forgetting_difference,
    paired_feedback_followup_records,
    paired_bootstrap_query_error_rate_difference,
    paired_sample_edit_transitions,
    static_corrected_forgetting,
    summarize_feedback_corrections,
    summarize_feedback_queries,
    summarize_localized_feedback_updates,
    summarize_runtime_resources,
    summarize_seed_cers,
)
from scripts.analyze_continual_results import _load_experiment


def assert_close(actual, expected, tolerance=1e-12):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), (
        actual,
        expected,
    )


def make_record(index, target, transcript, *, update="skipped", route=0):
    return {
        "index": index,
        "uid": f"sample-{index:04d}",
        "target": target,
        "transcript": transcript,
        "update": {"status": update},
        "route": {"expert_index": route},
    }


def write_experiment(directory, records, *, mode, summary_overrides=None):
    directory.mkdir(parents=True)
    with (directory / "stream_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        **aggregate_character_cer(records),
        "mode": mode,
        "updates": {"skipped": len(records)},
        "expert_bank": {"expert_count": 1, "route_counts": [len(records)]},
    }
    if summary_overrides:
        summary.update(summary_overrides)
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )


def make_revisit_records(error_counts, segment_lengths=(133, 255, 222, 132)):
    records = []
    index = 0
    for length, errors in zip(segment_lengths, error_counts):
        for offset in range(length):
            records.append(
                make_record(index, "字", "错" if offset < errors else "字")
            )
            index += 1
    return records


def main():
    aggregate = aggregate_character_cer(
        [
            make_record(0, "ab", "ab"),
            make_record(1, "xy", "x"),
        ]
    )
    assert aggregate == {
        "samples": 2,
        "edits": 1,
        "characters": 4,
        "cer": 0.25,
    }
    candidate_query_records = []
    baseline_query_records = []
    for index in range(4):
        block_index = index // 2
        candidate_queried = index % 2 == 0
        baseline_queried = index % 2 == 1
        for records, queried, transcript, score in (
            (
                candidate_query_records,
                candidate_queried,
                "错" if candidate_queried else "字",
                0.2 if candidate_queried else 0.9,
            ),
            (baseline_query_records, baseline_queried, "字", 0.8),
        ):
            record = make_record(index, "字", transcript)
            record["feedback_used"] = queried
            record["feedback_query"] = {
                "queried": queried,
                "policy_requested": queried,
                "manifest_requested": False,
                "source": "policy" if queried else "none",
                "reason": "query" if queried else "skip",
                "block_index": block_index,
                "position_in_block": index % 2,
                "reliability_threshold": 0.5,
            }
            record["reliability"] = {"score": score}
            record["runtime"] = {
                "video_load_seconds": 0.1,
                "process_seconds": 0.4 + index * 0.1,
                "total_seconds": 0.5 + index * 0.1,
                "gpu_max_memory_allocated_bytes": 1000 + index,
            }
            records.append(record)

    query_summary = summarize_feedback_queries(candidate_query_records)
    assert query_summary["available"]
    assert query_summary["queries"] == 2
    assert query_summary["policy_queries"] == 2
    assert query_summary["queried_true_error_rate"] == 1.0
    assert query_summary["nonqueried_true_error_rate"] == 0.0
    assert query_summary["max_policy_queries_per_block"] == 1
    assert_close(query_summary["mean_queried_reliability"], 0.2)
    runtime_summary = summarize_runtime_resources(candidate_query_records)
    assert runtime_summary["available"]
    assert runtime_summary["samples"] == 4
    assert_close(runtime_summary["timing"]["video_load_seconds"]["total"], 0.4)
    assert_close(runtime_summary["timing"]["process_seconds"]["p50"], 0.55)
    assert runtime_summary["peak_gpu_memory_allocated_bytes"] == 1003

    query_difference = paired_bootstrap_query_error_rate_difference(
        candidate_query_records,
        baseline_query_records,
        iterations=101,
        seed=9,
        batch_size=7,
    )
    assert query_difference["candidate_minus_baseline"] == 1.0
    assert query_difference["ci_95"] == {"lower": 1.0, "upper": 1.0}
    assert query_difference["paired_blocks"] == 2

    feedback_records = [
        {
            **make_record(0, "字", "错", update="accepted"),
            "feedback_used": True,
            "update": {
                "status": "accepted",
                "objective_before": 1.25,
                "objective_after": 0.75,
                "localization": {
                    "strategy": "ctc_error_local",
                    "randomized_support": False,
                    "ctc_frames": 8,
                    "target_log_likelihood": -3.0,
                    "matched_target_tokens": 1,
                    "error_target_tokens": 2,
                    "substitution_target_tokens": 1,
                    "deletion_target_tokens": 1,
                    "insertion_frames": 2,
                    "matched_occupancy_mass": 2.0,
                    "error_occupancy_mass": 3.0,
                    "matched_effective_frames": 1.5,
                    "error_effective_frames": 2.5,
                },
                "correction": {
                    "predicted_tokens": 2,
                    "target_tokens": 3,
                    "matched_tokens": 1,
                    "substituted_tokens": 1,
                    "missing_target_tokens": 1,
                    "extra_prediction_tokens": 0,
                    "token_error_rate": 2 / 3,
                    "matched_frame_rate": 0.25,
                },
            },
        },
        make_record(1, "字", "字"),
        make_record(2, "字", "错"),
        {**make_record(3, "字", "字"), "feedback_used": True},
        make_record(4, "字", "字"),
    ]
    correction_summary = summarize_feedback_corrections(feedback_records)
    assert correction_summary["feedback_samples"] == 2
    assert correction_summary["diagnosed_feedback_samples"] == 1
    assert correction_summary["diagnostic_coverage"] == 0.5
    assert correction_summary["substituted_tokens"] == 1
    assert correction_summary["missing_target_tokens"] == 1
    assert_close(correction_summary["mean_token_error_rate"], 2 / 3)
    assert_close(correction_summary["mean_matched_frame_rate"], 0.25)
    localized_summary = summarize_localized_feedback_updates(feedback_records)
    assert localized_summary["feedback_samples"] == 2
    assert localized_summary["localized_feedback_samples"] == 1
    assert localized_summary["strategies"] == {"ctc_error_local": 1}
    assert localized_summary["update_statuses"] == {"accepted": 1}
    assert localized_summary["error_target_tokens"] == 2
    assert_close(localized_summary["insertion_frame_coverage"], 0.25)
    assert_close(localized_summary["mean_error_occupancy_mass"], 3.0)
    assert_close(localized_summary["mean_objective_delta"], -0.5)
    assert localized_summary["objective_improved_samples"] == 1
    followup = feedback_followup_records(feedback_records, 2)
    assert [record["index"] for record in followup] == [1, 2, 4]
    no_feedback_records = [
        {key: value for key, value in record.items() if key != "feedback_used"}
        for record in feedback_records
    ]
    candidate_followup, baseline_followup = paired_feedback_followup_records(
        feedback_records, no_feedback_records, 2
    )
    assert [record["uid"] for record in candidate_followup] == [
        record["uid"] for record in baseline_followup
    ]

    transition_candidate = [
        make_record(0, "字", "字"),
        make_record(1, "字", "错"),
        make_record(2, "字", "错"),
    ]
    transition_baseline = [
        make_record(0, "字", "错"),
        make_record(1, "字", "字"),
        make_record(2, "字", "错"),
    ]
    assert paired_sample_edit_transitions(
        transition_candidate, transition_baseline
    ) == {
        "paired_samples": 3,
        "candidate_better": 1,
        "same": 1,
        "candidate_worse": 1,
        "net_edit_difference": 0,
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        valid_directory = root / "valid"
        records = [make_record(0, "ab", "ab"), make_record(1, "xy", "x")]
        write_experiment(valid_directory, records, mode="static")
        _, loaded_records, loaded_summary, loaded_overall = _load_experiment(
            valid_directory
        )
        assert loaded_records == records
        assert loaded_summary["cer"] == 0.25
        assert loaded_overall == aggregate

        tolerated_directory = root / "tolerated-cer-rounding"
        write_experiment(
            tolerated_directory,
            records,
            mode="static",
            summary_overrides={"cer": 0.25 + 5e-13},
        )
        assert _load_experiment(tolerated_directory)[3] == aggregate

        invalid_summaries = (
            ("stale-edits", {"edits": 2}, "edits"),
            ("stale-cer", {"cer": 0.25 + 2e-12}, "cer"),
            ("non-finite-cer", {"cer": float("nan")}, "cer"),
            ("boolean-characters", {"characters": True}, "characters"),
            ("floating-edits", {"edits": 1.0}, "edits"),
            ("negative-samples", {"samples": -1}, "samples"),
        )
        for name, overrides, expected_message in invalid_summaries:
            invalid_directory = root / name
            write_experiment(
                invalid_directory,
                records,
                mode="static",
                summary_overrides=overrides,
            )
            try:
                _load_experiment(invalid_directory)
            except ValueError as error:
                assert expected_message in str(error), str(error)
            else:
                raise AssertionError(f"损坏 summary 必须被拒绝：{name}")

    seeds = summarize_seed_cers([0.1, 0.2, 0.3])
    assert seeds["count"] == 3
    assert_close(seeds["mean"], 0.2)
    assert_close(seeds["population_variance"], 1 / 150)
    assert_close(seeds["population_std"], math.sqrt(1 / 150))
    try:
        summarize_seed_cers([0.1, None, 0.3])
    except ValueError as error:
        assert "seed CER" in str(error)
    else:
        raise AssertionError("非法 seed CER 必须抛出 ValueError")

    static_segments = aggregate_revisit_segments(
        make_revisit_records((13, 0, 0, 26))
    )
    method_segments = aggregate_revisit_segments(
        make_revisit_records((20, 0, 0, 21))
    )
    assert static_segments["A1"]["samples"] == 133
    assert static_segments["A1"]["edits"] == 13
    assert static_segments["B"]["samples"] == 255
    assert static_segments["C"]["samples"] == 222
    assert static_segments["A2"]["samples"] == 132
    assert static_segments["A2"]["edits"] == 26
    forgetting = static_corrected_forgetting(method_segments, static_segments)
    assert_close(forgetting["method_a2_minus_a1"], 21 / 132 - 20 / 133)
    assert_close(forgetting["static_a2_minus_a1"], 26 / 132 - 13 / 133)
    assert_close(
        forgetting["static_corrected"],
        (21 / 132 - 20 / 133) - (26 / 132 - 13 / 133),
    )

    validation_lengths = (130, 195, 186, 130)
    validation_segments = aggregate_revisit_segments(
        make_revisit_records((13, 0, 0, 26), validation_lengths),
        segment_lengths=validation_lengths,
    )
    assert [
        validation_segments[name]["samples"] for name in ("A1", "B", "C", "A2")
    ] == [130, 195, 186, 130]

    baseline = [make_record(index, "字", "错") for index in range(8)]
    candidate = [make_record(index, "字", "字") for index in range(8)]
    comparison = paired_bootstrap_cer_difference(
        candidate,
        list(reversed(baseline)),
        iterations=101,
        seed=123,
        batch_size=7,
    )
    assert_close(comparison["candidate_minus_baseline"], -1.0)
    assert_close(comparison["ci_95"]["lower"], -1.0)
    assert_close(comparison["ci_95"]["upper"], -1.0)
    assert comparison["iterations"] == 101
    assert comparison["seed"] == 123

    varied_candidate = [
        make_record(0, "字", "字"),
        make_record(1, "字", "字"),
        make_record(2, "字", "错"),
        make_record(3, "字", "错"),
    ]
    varied_baseline = [
        make_record(0, "字", "错"),
        make_record(1, "字", "字"),
        make_record(2, "字", "字"),
        make_record(3, "字", "错"),
    ]
    unbatched = paired_bootstrap_cer_difference(
        varied_candidate,
        varied_baseline,
        iterations=103,
        seed=7,
        batch_size=103,
    )
    batched = paired_bootstrap_cer_difference(
        varied_candidate,
        varied_baseline,
        iterations=103,
        seed=7,
        batch_size=5,
    )
    assert unbatched == batched
    assert_close(batched["ci_95"]["lower"], -0.6125)
    assert_close(batched["ci_95"]["upper"], 0.75)

    did_lengths = (4, 1, 1, 4)
    candidate_errors = (0, 0, 1, 1, 0, 0, 0, 0, 0, 1)
    baseline_errors = (1, 0, 0, 1, 0, 0, 1, 1, 0, 0)
    did_candidate = [
        make_record(index, "字", "错" if error else "字")
        for index, error in enumerate(candidate_errors)
    ]
    did_baseline = [
        make_record(index, "字", "错" if error else "字")
        for index, error in enumerate(baseline_errors)
    ]
    reordered_baseline = (
        list(reversed(did_baseline[:4]))
        + did_baseline[4:6]
        + list(reversed(did_baseline[6:]))
    )
    did_unbatched = paired_bootstrap_revisit_forgetting_difference(
        did_candidate,
        reordered_baseline,
        segment_lengths=did_lengths,
        iterations=1003,
        seed=13,
        batch_size=1003,
    )
    did_batched = paired_bootstrap_revisit_forgetting_difference(
        did_candidate,
        reordered_baseline,
        segment_lengths=did_lengths,
        iterations=1003,
        seed=13,
        batch_size=7,
    )
    assert did_batched == did_unbatched
    assert_close(did_batched["point"], -0.25)
    assert did_batched["ci_95"]["lower"] < did_batched["point"]
    assert did_batched["ci_95"]["upper"] > did_batched["point"]
    assert did_batched["iterations"] == 1003
    assert did_batched["seed"] == 13
    assert did_batched["paired_samples"] == {"A1": 4, "A2": 4}

    mismatched_uid = [dict(record) for record in did_baseline]
    mismatched_uid[-1]["uid"] = "different-a2-uid"
    try:
        paired_bootstrap_revisit_forgetting_difference(
            did_candidate,
            mismatched_uid,
            segment_lengths=did_lengths,
            iterations=10,
        )
    except ValueError as error:
        assert "A2" in str(error) and "uid" in str(error)
    else:
        raise AssertionError("A2 uid 集合不一致必须被拒绝")

    mismatched_target = [dict(record) for record in did_baseline]
    mismatched_target[0]["target"] = "词"
    try:
        paired_bootstrap_revisit_forgetting_difference(
            did_candidate,
            mismatched_target,
            segment_lengths=did_lengths,
            iterations=10,
        )
    except ValueError as error:
        assert "target" in str(error) and "sample-0000" in str(error)
    else:
        raise AssertionError("配对 target 不一致必须被拒绝")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        static_directory = root / "static"
        single_directory = root / "single"
        expert_directory = root / "expert"
        write_experiment(
            static_directory,
            make_revisit_records((13, 0, 0, 26), validation_lengths),
            mode="static",
        )
        write_experiment(
            single_directory,
            make_revisit_records((20, 0, 0, 21), validation_lengths),
            mode="single_adapter",
        )
        write_experiment(
            expert_directory,
            make_revisit_records((18, 0, 0, 18), validation_lengths),
            mode="expert_bank",
        )
        output = root / "analysis.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_continual_results.py"),
            "--experiment",
            f"static={static_directory}",
            "--experiment",
            f"single={single_directory}",
            "--experiment",
            f"expert={expert_directory}",
            "--revisit",
            "static",
            "--revisit",
            "single",
            "--revisit",
            "expert",
            "--static-revisit",
            "static",
            "--revisit-segment-lengths",
            "130,195,186,130",
            "--comparison",
            "expert:single",
            "--comparison",
            "expert:static",
            "--seed-group",
            "toy=static,single,expert",
            "--bootstrap-iterations",
            "101",
            "--bootstrap-seed",
            "123",
            "--bootstrap-batch-size",
            "7",
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        analysis = json.loads(output.read_text(encoding="utf-8"))
        assert analysis["schema_version"] == 1
        assert analysis["experiments"]["static"]["overall"]["edits"] == 39
        assert analysis["experiments"]["expert"]["segments"]["A2"]["edits"] == 18
        assert_close(
            analysis["experiments"]["expert"]["forgetting"][
                "static_corrected"
            ],
            (18 / 130 - 18 / 130) - (26 / 130 - 13 / 130),
        )
        paired = analysis["comparisons"]["expert_minus_single"]
        assert_close(paired["candidate_minus_baseline"], -5 / 641)
        expert_single_forgetting = paired["revisit_forgetting_difference"]
        assert_close(expert_single_forgetting["point"], -1 / 130)
        assert expert_single_forgetting["ci_95"]["lower"] < 0
        assert expert_single_forgetting["iterations"] == 101
        assert expert_single_forgetting["seed"] == 123
        expert_static_forgetting = analysis["comparisons"][
            "expert_minus_static"
        ]["revisit_forgetting_difference"]
        assert_close(expert_static_forgetting["point"], -13 / 130)
        assert analysis["seed_groups"]["toy"]["count"] == 3
        assert analysis["experiments"]["expert"]["run_summary"][
            "expert_bank"
        ]["route_counts"] == [641]
        assert analysis["experiments"]["expert"][
            "localized_feedback_updates"
        ]["localized_feedback_samples"] == 0
        assert not analysis["experiments"]["expert"]["feedback_queries"][
            "available"
        ]
        assert not analysis["experiments"]["expert"]["runtime_resources"][
            "available"
        ]
        assert analysis["revisit_protocol"]["segment_lengths"] == {
            "A1": 130,
            "B": 195,
            "C": 186,
            "A2": 130,
        }
        assert not output.with_name(f".{output.name}.tmp").exists()

    print("RSP-VSR continual result analysis smoke 通过")


if __name__ == "__main__":
    main()
