import argparse
import csv
import json
import math
from pathlib import Path


METHODS = (
    ("static", "Static", "U0", "#A7A9AC"),
    ("bn_tent", "BN-TENT-VSR", "U0", "#C05A5A"),
    ("eta", "ETA-VSR", "U0", "#8F496E"),
    ("incumbent", "Replay adapter", "F10", "#0F4D92"),
    ("online_lora", "Online LoRA", "F10", "#42949E"),
)
COMPARISONS = (
    ("bn_tent_minus_static", "BN-TENT - static", "#C05A5A"),
    ("eta_minus_static", "ETA - static", "#8F496E"),
    ("incumbent_minus_static", "Replay - static", "#0F4D92"),
    ("online_lora_minus_static", "LoRA - static", "#42949E"),
    ("online_lora_minus_incumbent", "LoRA - replay", "#B64342"),
)
FORGETTING = (
    ("bn_tent_minus_static", "BN-TENT", "#C05A5A"),
    ("eta_minus_static", "ETA", "#8F496E"),
    ("incumbent_minus_static", "Replay", "#0F4D92"),
    ("online_lora_minus_static", "Online LoRA", "#42949E"),
    ("online_lora_minus_incumbent", "LoRA - replay", "#B64342"),
)
RESOURCE_FIELDS = (
    ("updatable_parameters", "Params\n(k)", 1_000.0, "{:.1f}"),
    ("throughput", "Throughput\n(samples/s)", 1.0, "{:.3f}"),
    ("peak_gpu_memory", "Peak GPU\n(GiB)", 1024.0**3, "{:.2f}"),
    ("checkpoint_bytes", "3 checkpoints\n(MiB)", 1024.0**2, "{:.2f}"),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot the development-only RSP-VSR dev4 baseline figure"
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=700)
    parser.add_argument("--expected-feedback-queries", type=int, default=70)
    parser.add_argument("--min-bootstrap-iterations", type=int, default=10000)
    return parser


def _read_analysis(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Analysis file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Analysis file is not valid JSON: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Analysis must be a schema_version=1 JSON object")
    return value


def _mapping(value, source):
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be a JSON object")
    return value


def _number(value, source):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{source} must be finite")
    return float(value)


def _interval(comparison, source, field=None):
    value = comparison if field is None else _mapping(comparison.get(field), source)
    ci = _mapping(value.get("ci_95"), f"{source}.ci_95")
    point_key = "candidate_minus_baseline" if field is None else "point"
    point = _number(value.get(point_key), f"{source}.{point_key}")
    lower = _number(ci.get("lower"), f"{source}.ci_95.lower")
    upper = _number(ci.get("upper"), f"{source}.ci_95.upper")
    if not lower <= point <= upper:
        raise ValueError(f"{source} point estimate must lie inside its 95% CI")
    return point, lower, upper


def validate_analysis(
    analysis, *, expected_samples, expected_feedback_queries, min_iterations
):
    experiments = _mapping(analysis.get("experiments"), "experiments")
    comparisons = _mapping(analysis.get("comparisons"), "comparisons")
    bootstrap = _mapping(analysis.get("bootstrap"), "bootstrap")
    iterations = bootstrap.get("iterations")
    if not isinstance(iterations, int) or iterations < min_iterations:
        raise ValueError(
            f"bootstrap iterations must be at least {min_iterations}, got {iterations}"
        )

    data = {"overall": {}, "resources": {}, "comparisons": {}, "forgetting": {}}
    expected_parameters = {
        "static": (0, 0, "adapter"),
        "bn_tent": (9472, 38, "batch_norm"),
        "eta": (9472, 38, "batch_norm"),
        "incumbent": (75265, 5, "adapter"),
        "online_lora": (73728, 96, "lora"),
    }
    expected_updates = {
        "static": {"skipped": expected_samples},
        "bn_tent": {"accepted": 699, "skipped": 1},
        "eta": {"accepted": 2, "skipped": 698},
        "incumbent": {"accepted": 81, "skipped": 619},
        "online_lora": {"accepted": 70, "skipped": 630},
    }

    for name, _, track, _ in METHODS:
        experiment = _mapping(experiments.get(name), f"experiment {name}")
        overall = _mapping(experiment.get("overall"), f"{name}.overall")
        if overall.get("samples") != expected_samples:
            raise ValueError(
                f"{name} must contain {expected_samples} samples, "
                f"got {overall.get('samples')}"
            )
        data["overall"][name] = {
            "cer": _number(overall.get("cer"), f"{name}.overall.cer"),
            "track": track,
        }

        summary = _mapping(experiment.get("run_summary"), f"{name}.run_summary")
        if summary.get("updates") != expected_updates[name]:
            raise ValueError(f"{name}.updates does not match the audited dev4 run")
        resources = _mapping(summary.get("resources"), f"{name}.resources")
        expected_param_count, expected_tensors, expected_mode = expected_parameters[name]
        if resources.get("updatable_parameters") != expected_param_count:
            raise ValueError(f"{name}.updatable_parameters is invalid")
        if resources.get("updatable_parameter_tensors") != expected_tensors:
            raise ValueError(f"{name}.updatable_parameter_tensors is invalid")
        if resources.get("parameter_update_mode") != expected_mode:
            raise ValueError(f"{name}.parameter_update_mode is invalid")
        if resources.get("retained_checkpoint_files") != 3:
            raise ValueError(f"{name} must retain exactly three checkpoints")

        query = _mapping(summary.get("feedback_query"), f"{name}.feedback_query")
        expected_queries = expected_feedback_queries if track == "F10" else 0
        for field in ("planned_budget", "policy_queries", "total_queries"):
            if query.get(field) != expected_queries:
                raise ValueError(f"{name}.{field} must be {expected_queries}")
        if query.get("manifest_queries") != 0:
            raise ValueError(f"{name} must not use manifest-provided feedback")

        data["resources"][name] = {
            "updatable_parameters": expected_param_count,
            "throughput": _number(
                resources.get("samples_per_process_second"), f"{name}.throughput"
            ),
            "peak_gpu_memory": _number(
                resources.get("peak_gpu_memory_allocated_bytes"),
                f"{name}.peak_gpu_memory_allocated_bytes",
            ),
            "checkpoint_bytes": _number(
                resources.get("retained_checkpoint_bytes"),
                f"{name}.retained_checkpoint_bytes",
            ),
        }

    for key, _, _ in COMPARISONS:
        comparison = _mapping(comparisons.get(key), f"comparison {key}")
        data["comparisons"][key] = _interval(comparison, key)

    for key, _, _ in FORGETTING:
        comparison = _mapping(comparisons.get(key), f"comparison {key}")
        data["forgetting"][key] = _interval(
            comparison, f"{key}.revisit_forgetting_difference", "revisit_forgetting_difference"
        )
    return data


def _configure_matplotlib():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required for figure generation") from error
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _panel_label(axis, label):
    axis.text(
        -0.17,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def _forest_point(axis, y, interval, color):
    point, lower, upper = (value * 100 for value in interval)
    axis.errorbar(
        point,
        y,
        xerr=[[point - lower], [upper - point]],
        fmt="o",
        color=color,
        markersize=4,
        capsize=2.5,
        linewidth=1.1,
        zorder=3,
    )


def write_source_data(path, data):
    rows = []
    for name, _, track, _ in METHODS:
        rows.append(
            ("a", name, track, "overall_cer", data["overall"][name]["cer"], "", "", "ratio")
        )
    for key, _, _ in COMPARISONS:
        point, lower, upper = data["comparisons"][key]
        rows.append(("b", key, "paired", "cer_difference", point, lower, upper, "ratio"))
    for key, _, _ in FORGETTING:
        point, lower, upper = data["forgetting"][key]
        rows.append(
            ("c", key, "paired", "static_corrected_forgetting", point, lower, upper, "ratio")
        )
    for name, _, track, _ in METHODS:
        for field, _, divisor, _ in RESOURCE_FIELDS:
            rows.append(
                (
                    "d",
                    name,
                    track,
                    field,
                    data["resources"][name][field] / divisor,
                    "",
                    "",
                    {
                        "updatable_parameters": "thousand_parameters",
                        "throughput": "samples_per_process_second",
                        "peak_gpu_memory": "GiB",
                        "checkpoint_bytes": "MiB",
                    }[field],
                )
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ("panel", "method", "track", "metric", "point", "ci_lower", "ci_upper", "unit")
        )
        writer.writerows(rows)


def plot(data, output_prefix):
    plt = _configure_matplotlib()
    import numpy as np

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.16, 5.35), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.06, 1.0), height_ratios=(1.0, 1.04))

    axis = figure.add_subplot(grid[0, 0])
    labels = [label for _, label, _, _ in METHODS]
    values = [data["overall"][name]["cer"] * 100 for name, _, _, _ in METHODS]
    colors = [color for _, _, _, color in METHODS]
    y = np.arange(len(METHODS))
    axis.scatter(values, y, color=colors, s=29, zorder=3)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(62, 108)
    axis.set_xlabel("Character error rate (%)  lower is better")
    axis.axhline(2.5, color="#B8BDC5", linewidth=0.7)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6, zorder=0)
    for position, value in zip(y, values):
        axis.text(value + 0.7, position, f"{value:.2f}", va="center", fontsize=6.2)
    axis.text(0.99, 0.99, "U0: no feedback", transform=axis.transAxes, ha="right", va="top", fontsize=6.2)
    axis.text(0.99, 0.37, "F10: fixed 10% feedback", transform=axis.transAxes, ha="right", va="top", fontsize=6.2)
    _panel_label(axis, "a")

    axis = figure.add_subplot(grid[0, 1])
    for position, (key, _, color) in enumerate(COMPARISONS):
        _forest_point(axis, position, data["comparisons"][key], color)
    axis.axvline(0, color="#4B5563", linewidth=0.8)
    axis.set_yticks(range(len(COMPARISONS)), [label for _, label, _ in COMPARISONS])
    axis.invert_yaxis()
    axis.set_xlabel("Paired CER difference (percentage points)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    _panel_label(axis, "b")

    axis = figure.add_subplot(grid[1, 0])
    for position, (key, _, color) in enumerate(FORGETTING):
        _forest_point(axis, position, data["forgetting"][key], color)
    axis.axvline(0, color="#4B5563", linewidth=0.8)
    axis.set_yticks(range(len(FORGETTING)), [label for _, label, _ in FORGETTING])
    axis.invert_yaxis()
    axis.set_xlabel("Paired static-corrected A2-A1 forgetting difference (pp)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    _panel_label(axis, "c")

    axis = figure.add_subplot(grid[1, 1])
    raw = np.array(
        [
            [data["resources"][name][field] / divisor for field, _, divisor, _ in RESOURCE_FIELDS]
            for name, _, _, _ in METHODS
        ],
        dtype=float,
    )
    burden = np.zeros_like(raw)
    for column in range(raw.shape[1]):
        values_column = raw[:, column]
        value_range = values_column.max() - values_column.min()
        burden[:, column] = 0 if value_range == 0 else (values_column - values_column.min()) / value_range
    burden[:, 1] = 1.0 - burden[:, 1]
    axis.imshow(burden, cmap="Greys", vmin=-0.25, vmax=1.35, aspect="auto")
    axis.set_xticks(range(len(RESOURCE_FIELDS)), [label for _, label, _, _ in RESOURCE_FIELDS])
    axis.set_yticks(range(len(METHODS)), [label for _, label, _, _ in METHODS])
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    axis.tick_params(axis="y", length=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    for row in range(raw.shape[0]):
        for column, (_, _, _, formatter) in enumerate(RESOURCE_FIELDS):
            color = "white" if burden[row, column] > 0.62 else "#272727"
            axis.text(column, row, formatter.format(raw[row, column]), ha="center", va="center", color=color, fontsize=6.1)
    axis.text(
        0.99,
        -0.08,
        "Darker = greater resource burden within column",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="#4D4D4D",
    )
    _panel_label(axis, "d")

    for suffix, options in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600}),
    ):
        figure.savefig(
            output_prefix.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(figure)
    write_source_data(output_prefix.with_suffix(".source_data.csv"), data)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.expected_samples < 1 or args.expected_feedback_queries < 1:
        raise ValueError("expected sample and query counts must be positive")
    analysis = _read_analysis(args.analysis)
    data = validate_analysis(
        analysis,
        expected_samples=args.expected_samples,
        expected_feedback_queries=args.expected_feedback_queries,
        min_iterations=args.min_bootstrap_iterations,
    )
    plot(data, args.output_prefix)
    print(f"dev4 baseline figure written to {args.output_prefix}.[pdf|svg|png|tiff]")


if __name__ == "__main__":
    main()
