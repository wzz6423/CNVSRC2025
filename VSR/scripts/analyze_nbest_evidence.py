import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.artifacts import _write_json_atomic
from plasticity.corrections import _token_alignment
from plasticity.reliability import edit_distance
from scripts.export_nbest_evidence import read_jsonl


def build_parser():
    parser = argparse.ArgumentParser(
        description="计算 decoder N-best oracle、字符覆盖率与 Phase-0a 门控"
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-oracle-headroom", type=float, default=0.02)
    parser.add_argument("--min-substitution-coverage", type=float, default=0.55)
    parser.add_argument("--reference-stream-results", type=Path)
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _char_alignment(prediction, target):
    return _token_alignment(
        [ord(character) for character in prediction],
        [ord(character) for character in target],
    )


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def coverage_reachability_bound(
    records, reference_records, *, min_substitution_coverage=0.55
):
    if len(records) > len(reference_records):
        raise ValueError("evidence 不能长于完整 reference stream")
    total_substitutions = 0
    prefix_substitutions = 0
    prefix_covered = 0
    for index, reference in enumerate(reference_records):
        if reference.get("index") != index:
            raise ValueError(f"reference stream index 不连续：{index}")
        target = reference.get("target")
        one_best = reference.get("transcript")
        if not isinstance(target, str) or not isinstance(one_best, str):
            raise ValueError(f"reference 样本 {index} target/transcript 非法")
        substitutions = {
            target_index
            for operation, _, target_index in _char_alignment(one_best, target)
            if operation == "substitution"
        }
        total_substitutions += len(substitutions)
        if index >= len(records):
            continue

        record = records[index]
        if (
            record.get("index") != index
            or record.get("uid") != reference.get("uid")
            or record.get("target") != target
            or record.get("one_best") != one_best
        ):
            raise ValueError(f"evidence 与 reference 在样本 {index} 不一致")
        matched_target_indices = set()
        for hypothesis in record["nbest"]:
            matched_target_indices.update(
                target_index
                for operation, _, target_index in _char_alignment(
                    hypothesis["transcript"], target
                )
                if operation == "match"
            )
        prefix_substitutions += len(substitutions)
        prefix_covered += len(substitutions & matched_target_indices)

    remaining_substitutions = total_substitutions - prefix_substitutions
    if total_substitutions:
        maximum_final_coverage = (
            prefix_covered + remaining_substitutions
        ) / total_substitutions
        required_remaining_coverage = (
            float(min_substitution_coverage) * total_substitutions - prefix_covered
        )
        required_remaining_coverage = (
            required_remaining_coverage / remaining_substitutions
            if remaining_substitutions
            else None
        )
        unreachable = maximum_final_coverage < float(min_substitution_coverage)
    else:
        maximum_final_coverage = None
        required_remaining_coverage = None
        unreachable = False
    return {
        "reference_samples": len(reference_records),
        "observed_samples": len(records),
        "total_reference_substitutions": total_substitutions,
        "observed_substitutions": prefix_substitutions,
        "observed_substitutions_covered": prefix_covered,
        "remaining_substitutions": remaining_substitutions,
        "required_remaining_coverage": required_remaining_coverage,
        "maximum_final_coverage": maximum_final_coverage,
        "minimum_substitution_coverage": float(min_substitution_coverage),
        "mathematically_unreachable": unreachable,
    }


def analyze_nbest_evidence(
    records,
    *,
    top_k=10,
    min_oracle_headroom=0.02,
    min_substitution_coverage=0.55,
):
    top_k = int(top_k)
    if top_k < 2:
        raise ValueError("top-k 必须至少为 2")
    for name, value in (
        ("min-oracle-headroom", min_oracle_headroom),
        ("min-substitution-coverage", min_substitution_coverage),
    ):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} 必须位于 [0,1]")

    total_characters = 0
    one_best_edits = 0
    nbest_oracle_edits = 0
    compositional_uncovered = 0
    substitution_total = 0
    substitution_covered = 0
    deletion_total = 0
    deletion_covered = 0
    exact_oracle_samples = 0
    correct_one_best_samples = 0
    correct_with_wrong_alternative = 0
    unique_hypotheses = 0
    hypothesis_counts = []
    oracle_ranks = Counter()
    complete_top_k_samples = 0

    for expected_index, record in enumerate(records):
        if record.get("index") != expected_index:
            raise ValueError(f"evidence index 不连续：{expected_index}")
        target = record.get("target")
        one_best = record.get("one_best")
        hypotheses = record.get("nbest")
        if not isinstance(target, str) or not target:
            raise ValueError(f"样本 {expected_index} target 不能为空")
        if not isinstance(one_best, str) or not isinstance(hypotheses, list):
            raise ValueError(f"样本 {expected_index} evidence 结构非法")
        candidates = hypotheses[:top_k]
        if not candidates or candidates[0].get("transcript") != one_best:
            raise ValueError(f"样本 {expected_index} rank-1 与 one_best 不一致")
        transcripts = []
        for rank, candidate in enumerate(candidates, start=1):
            if candidate.get("rank") != rank or not isinstance(
                candidate.get("transcript"), str
            ):
                raise ValueError(f"样本 {expected_index} N-best rank 非法")
            transcripts.append(candidate["transcript"])
        complete_top_k_samples += int(len(candidates) == top_k)

        total_characters += len(target)
        sample_one_best_edits = edit_distance(list(one_best), list(target))
        one_best_edits += sample_one_best_edits
        distances = [
            edit_distance(list(transcript), list(target)) for transcript in transcripts
        ]
        best_distance = min(distances)
        best_rank = distances.index(best_distance) + 1
        nbest_oracle_edits += best_distance
        oracle_ranks[str(best_rank)] += 1
        exact_oracle_samples += int(best_distance == 0)

        matched_target_indices = set()
        for transcript in transcripts:
            matched_target_indices.update(
                target_index
                for operation, _, target_index in _char_alignment(transcript, target)
                if operation == "match"
            )
        compositional_uncovered += len(target) - len(matched_target_indices)

        one_best_operations = _char_alignment(one_best, target)
        substitutions = {
            target_index
            for operation, _, target_index in one_best_operations
            if operation == "substitution"
        }
        deletions = {
            target_index
            for operation, _, target_index in one_best_operations
            if operation == "missing_target"
        }
        substitution_total += len(substitutions)
        substitution_covered += len(substitutions & matched_target_indices)
        deletion_total += len(deletions)
        deletion_covered += len(deletions & matched_target_indices)

        if sample_one_best_edits == 0:
            correct_one_best_samples += 1
            correct_with_wrong_alternative += int(
                any(transcript != target for transcript in transcripts[1:])
            )
        hypothesis_counts.append(len(transcripts))
        unique_hypotheses += len(set(transcripts))

    if not records or not total_characters:
        raise ValueError("evidence 必须包含至少一个非空 target")
    one_best_cer = one_best_edits / total_characters
    nbest_oracle_cer = nbest_oracle_edits / total_characters
    compositional_oracle_cer = compositional_uncovered / total_characters
    oracle_headroom = one_best_cer - nbest_oracle_cer
    substitution_coverage = _ratio(substitution_covered, substitution_total)
    oracle_pass = oracle_headroom >= float(min_oracle_headroom)
    coverage_pass = (
        substitution_coverage is not None
        and substitution_coverage >= float(min_substitution_coverage)
    )
    complete_top_k_pass = complete_top_k_samples == len(records)
    decision = (
        "BEAM_GO" if oracle_pass and coverage_pass and complete_top_k_pass else "NO_GO"
    )

    return {
        "schema_version": 1,
        "samples": len(records),
        "top_k": top_k,
        "characters": total_characters,
        "one_best": {"cer": one_best_cer, "edits": one_best_edits},
        "nbest_oracle": {
            "cer": nbest_oracle_cer,
            "edits": nbest_oracle_edits,
            "headroom": oracle_headroom,
            "exact_match_samples": exact_oracle_samples,
            "oracle_rank_histogram": dict(
                sorted(oracle_ranks.items(), key=lambda item: int(item[0]))
            ),
        },
        "permissive_compositional_oracle": {
            "cer": compositional_oracle_cer,
            "uncovered_target_characters": compositional_uncovered,
            "headroom": one_best_cer - compositional_oracle_cer,
            "limitation": "按 target 对齐汇总跨候选字符，忽略生成与额外字符代价，仅作乐观上界",
        },
        "error_position_coverage": {
            "substitutions": substitution_total,
            "substitutions_covered": substitution_covered,
            "substitution_coverage_at_k": substitution_coverage,
            "deletions": deletion_total,
            "deletions_covered": deletion_covered,
            "deletion_coverage_at_k": _ratio(deletion_covered, deletion_total),
        },
        "identity_risk": {
            "correct_one_best_samples": correct_one_best_samples,
            "with_incorrect_alternative": correct_with_wrong_alternative,
            "incorrect_alternative_rate": _ratio(
                correct_with_wrong_alternative, correct_one_best_samples
            ),
        },
        "hypotheses": {
            "minimum_per_sample": min(hypothesis_counts),
            "maximum_per_sample": max(hypothesis_counts),
            "mean_per_sample": sum(hypothesis_counts) / len(hypothesis_counts),
            "mean_unique_per_sample": unique_hypotheses / len(records),
            "complete_top_k_samples": complete_top_k_samples,
        },
        "phase0a_gate": {
            "minimum_oracle_headroom": float(min_oracle_headroom),
            "minimum_substitution_coverage": float(min_substitution_coverage),
            "oracle_headroom_pass": oracle_pass,
            "substitution_coverage_pass": coverage_pass,
            "complete_top_k_pass": complete_top_k_pass,
            "decision": decision,
            "authorizes": (
                "phase0b_pinyin_visual_evidence_probe"
                if decision == "BEAM_GO"
                else "stop_llm_route"
            ),
        },
    }


def run(args):
    records = read_jsonl(args.evidence)
    analysis = analyze_nbest_evidence(
        records,
        top_k=args.top_k,
        min_oracle_headroom=args.min_oracle_headroom,
        min_substitution_coverage=args.min_substitution_coverage,
    )
    analysis["source"] = {
        "path": str(args.evidence.resolve()),
        "sha256": _sha256(args.evidence),
    }
    if args.reference_stream_results:
        reference_records = read_jsonl(args.reference_stream_results)
        reachability = coverage_reachability_bound(
            records,
            reference_records,
            min_substitution_coverage=args.min_substitution_coverage,
        )
        if (
            reachability["observed_substitutions"]
            != analysis["error_position_coverage"]["substitutions"]
        ):
            raise ValueError("reference 与 evidence 的 substitution 统计不一致")
        if (
            reachability["observed_substitutions_covered"]
            != analysis["error_position_coverage"]["substitutions_covered"]
        ):
            raise ValueError("reference 与 evidence 的 coverage 统计不一致")
        reachability["reference"] = {
            "path": str(args.reference_stream_results.resolve()),
            "sha256": _sha256(args.reference_stream_results),
        }
        analysis["phase0a_gate"]["coverage_reachability"] = reachability
        if reachability["mathematically_unreachable"]:
            analysis["phase0a_gate"]["decision"] = "NO_GO"
            analysis["phase0a_gate"]["termination"] = "EARLY_NO_GO"
            analysis["phase0a_gate"]["authorizes"] = "stop_llm_route"
    _write_json_atomic(args.output, analysis)
    return analysis


def main():
    analysis = run(build_parser().parse_args())
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
