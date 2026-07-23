import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_active_feedback_results import validate_analysis


METHODS = (
    "static",
    "pseudo_only",
    "feedback_only",
    "periodic",
    "random",
    "uncertainty",
)


def _comparison(candidate, baseline, point, lower, upper):
    return {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_minus_baseline": point,
        "ci_95": {"lower": lower, "upper": upper},
        "revisit_forgetting_difference": {
            "point": point,
            "ci_95": {"lower": lower, "upper": upper},
        },
    }


def make_analysis():
    cers = {
        "static": 0.64,
        "pseudo_only": 0.62,
        "feedback_only": 0.615,
        "periodic": 0.60,
        "random": 0.61,
        "uncertainty": 0.59,
    }
    forgetting = {
        "static": 0.0,
        "pseudo_only": -0.005,
        "feedback_only": -0.008,
        "periodic": -0.012,
        "random": 0.004,
        "uncertainty": -0.018,
    }
    experiments = {}
    for name in METHODS:
        experiment = {
            "overall": {"samples": 758, "cer": cers[name]},
            "forgetting": {"static_corrected": forgetting[name]},
        }
        if name in {"periodic", "random", "uncertainty"}:
            experiment["feedback_queries"] = {
                "available": True,
                "queries": 75,
                "policy_queries": 75,
                "manifest_queries": 0,
                "policy_query_blocks": 75,
                "max_policy_queries_per_block": 1,
                "queried_true_error_rate": {
                    "periodic": 0.72,
                    "random": 0.68,
                    "uncertainty": 0.84,
                }[name],
            }
        experiments[name] = experiment

    comparisons = {}
    for name in METHODS[1:]:
        point = forgetting[name]
        comparisons[f"{name}_minus_static"] = _comparison(
            name, "static", point, point - 0.004, point + 0.004
        )
    comparisons["uncertainty_minus_periodic"] = _comparison(
        "uncertainty", "periodic", -0.01, -0.015, -0.005
    )
    comparisons["uncertainty_minus_random"] = _comparison(
        "uncertainty", "random", -0.02, -0.026, -0.014
    )
    comparisons["uncertainty_minus_random"]["query_error_rate_difference"] = {
        "candidate_minus_baseline": 0.16,
        "ci_95": {"lower": 0.05, "upper": 0.27},
    }
    return {
        "schema_version": 1,
        "experiments": experiments,
        "comparisons": comparisons,
        "bootstrap": {"iterations": 10000, "seed": 42, "batch_size": 256},
    }


def main():
    invalid_analysis = make_analysis()
    invalid_analysis["experiments"]["uncertainty"]["feedback_queries"][
        "policy_queries"
    ] = 74
    try:
        validate_analysis(
            invalid_analysis,
            expected_samples=758,
            expected_policy_queries=75,
            min_iterations=10000,
        )
    except ValueError as error:
        assert "policy_queries" in str(error), str(error)
    else:
        raise AssertionError("查询预算不完整时必须拒绝出图")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        analysis_path = root / "analysis.json"
        analysis_path.write_text(json.dumps(make_analysis()), encoding="utf-8")
        output_prefix = root / "active_feedback_dev3"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "plot_active_feedback_results.py"),
            "--analysis",
            str(analysis_path),
            "--output-prefix",
            str(output_prefix),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        for suffix in (".pdf", ".svg", ".png", ".tiff", ".source_data.csv"):
            path = output_prefix.with_suffix(suffix)
            assert path.is_file() and path.stat().st_size > 0, path
        header = output_prefix.with_suffix(".source_data.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        assert header == "panel,method,metric,point,ci_lower,ci_upper"

    print("RSP-VSR active feedback figure smoke 通过")


if __name__ == "__main__":
    main()
