import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_dev4_strong_baselines import validate_analysis


METHODS = ("static", "bn_tent", "eta", "incumbent", "online_lora")


def _comparison(candidate, baseline, point, lower, upper, forgetting):
    return {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_minus_baseline": point,
        "ci_95": {"lower": lower, "upper": upper},
        "revisit_forgetting_difference": {
            "point": forgetting,
            "ci_95": {"lower": forgetting - 0.004, "upper": forgetting + 0.004},
        },
    }


def make_analysis():
    cers = {
        "static": 0.67,
        "bn_tent": 1.03,
        "eta": 1.05,
        "incumbent": 0.65,
        "online_lora": 0.66,
    }
    resources = {
        "static": (0, 0, "adapter", 0.59, 1.8, 1.0),
        "bn_tent": (9472, 38, "batch_norm", 0.56, 3.4, 0.6),
        "eta": (9472, 38, "batch_norm", 0.54, 2.7, 0.7),
        "incumbent": (75265, 5, "adapter", 0.54, 1.9, 2.8),
        "online_lora": (73728, 96, "lora", 0.59, 1.9, 3.1),
    }
    updates = {
        "static": {"skipped": 700},
        "bn_tent": {"accepted": 699, "skipped": 1},
        "eta": {"accepted": 2, "skipped": 698},
        "incumbent": {"accepted": 81, "skipped": 619},
        "online_lora": {"accepted": 70, "skipped": 630},
    }
    experiments = {}
    for name in METHODS:
        parameters, tensors, mode, throughput, peak_gib, checkpoint_mib = resources[name]
        queries = 70 if name in {"incumbent", "online_lora"} else 0
        experiments[name] = {
            "overall": {"samples": 700, "cer": cers[name]},
            "run_summary": {
                "updates": updates[name],
                "feedback_query": {
                    "planned_budget": queries,
                    "policy_queries": queries,
                    "manifest_queries": 0,
                    "total_queries": queries,
                },
                "resources": {
                    "updatable_parameters": parameters,
                    "updatable_parameter_tensors": tensors,
                    "parameter_update_mode": mode,
                    "samples_per_process_second": throughput,
                    "peak_gpu_memory_allocated_bytes": int(peak_gib * 1024**3),
                    "retained_checkpoint_bytes": int(checkpoint_mib * 1024**2),
                    "retained_checkpoint_files": 3,
                },
            },
        }
    comparisons = {
        "bn_tent_minus_static": _comparison("bn_tent", "static", 0.36, 0.33, 0.41, 0.08),
        "eta_minus_static": _comparison("eta", "static", 0.38, 0.35, 0.43, 0.01),
        "incumbent_minus_static": _comparison("incumbent", "static", -0.02, -0.026, -0.015, -0.036),
        "online_lora_minus_static": _comparison("online_lora", "static", -0.008, -0.013, -0.003, -0.013),
        "online_lora_minus_incumbent": _comparison("online_lora", "incumbent", 0.012, 0.007, 0.017, 0.023),
    }
    return {
        "schema_version": 1,
        "experiments": experiments,
        "comparisons": comparisons,
        "bootstrap": {"iterations": 10000, "seed": 42, "batch_size": 256},
    }


def main():
    invalid = make_analysis()
    invalid["experiments"]["online_lora"]["run_summary"]["feedback_query"][
        "policy_queries"
    ] = 69
    try:
        validate_analysis(
            invalid,
            expected_samples=700,
            expected_feedback_queries=70,
            min_iterations=10000,
        )
    except ValueError as error:
        assert "policy_queries" in str(error), str(error)
    else:
        raise AssertionError("The plotter must reject an incomplete feedback budget")

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        analysis_path = root / "analysis.json"
        analysis_path.write_text(json.dumps(make_analysis()), encoding="utf-8")
        output_prefix = root / "dev4_strong_baselines"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_dev4_strong_baselines.py"),
                "--analysis",
                str(analysis_path),
                "--output-prefix",
                str(output_prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for suffix in (".pdf", ".svg", ".png", ".tiff", ".source_data.csv"):
            path = output_prefix.with_suffix(suffix)
            assert path.is_file() and path.stat().st_size > 0, path
        lines = output_prefix.with_suffix(".source_data.csv").read_text(
            encoding="utf-8"
        ).splitlines()
        assert lines[0] == (
            "panel,method,track,metric,point,ci_lower,ci_upper,unit"
        )
        assert len(lines) == 36

    print("RSP-VSR dev4 baseline figure smoke passed")


if __name__ == "__main__":
    main()
