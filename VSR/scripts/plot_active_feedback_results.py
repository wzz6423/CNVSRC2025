import argparse
import csv
import json
import math
from pathlib import Path


METHODS = (
    ("static", "Static", "#A7A9AC"),
    ("pseudo_only", "Pseudo-only", "#8AA6B8"),
    ("feedback_only", "Feedback-only", "#D8B365"),
    ("periodic", "Periodic", "#4C78A8"),
    ("random", "Random", "#E68653"),
    ("uncertainty", "Uncertainty", "#2A9D8F"),
)
POLICIES = ("periodic", "random", "uncertainty")


def build_parser():
    parser = argparse.ArgumentParser(
        description="绘制 target-dev3 主动反馈论文图并导出 source data"
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=758)
    parser.add_argument("--expected-policy-queries", type=int, default=75)
    parser.add_argument("--min-bootstrap-iterations", type=int, default=10000)
    return parser


def _read_analysis(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到分析文件：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"分析文件不是有效 JSON：{path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("分析文件必须是 schema_version=1 的 JSON 对象")
    return value


def _number(value, source):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{source} 必须是有限数值")
    return float(value)


def _require_mapping(value, source):
    if not isinstance(value, dict):
        raise ValueError(f"{source} 必须是 JSON 对象")
    return value


def _comparison(comparisons, candidate, baseline):
    name = f"{candidate}_minus_{baseline}"
    try:
        return _require_mapping(comparisons[name], f"comparison {name}")
    except KeyError as error:
        raise ValueError(f"缺少 comparison：{name}") from error


def _ci(comparison, source, field=None):
    value = comparison if field is None else comparison.get(field)
    value = _require_mapping(value, source)
    ci = _require_mapping(value.get("ci_95"), f"{source}.ci_95")
    point_name = "candidate_minus_baseline" if field is None else "point"
    point = _number(value.get(point_name), f"{source}.{point_name}")
    lower = _number(ci.get("lower"), f"{source}.ci_95.lower")
    upper = _number(ci.get("upper"), f"{source}.ci_95.upper")
    if lower > point or point > upper:
        raise ValueError(f"{source} 的点估计必须位于 95% CI 内")
    return point, lower, upper


def validate_analysis(
    analysis, *, expected_samples, expected_policy_queries, min_iterations
):
    experiments = _require_mapping(analysis.get("experiments"), "experiments")
    comparisons = _require_mapping(analysis.get("comparisons"), "comparisons")
    bootstrap = _require_mapping(analysis.get("bootstrap"), "bootstrap")
    iterations = bootstrap.get("iterations")
    if not isinstance(iterations, int) or iterations < min_iterations:
        raise ValueError(
            f"bootstrap iterations 必须至少为 {min_iterations}，实际为 {iterations}"
        )

    data = {"overall": {}, "queries": {}, "forgetting": {}, "comparisons": {}}
    for name, _, _ in METHODS:
        experiment = _require_mapping(experiments.get(name), f"experiment {name}")
        overall = _require_mapping(experiment.get("overall"), f"{name}.overall")
        if overall.get("samples") != expected_samples:
            raise ValueError(
                f"{name} 必须包含 {expected_samples} 个样本，"
                f"实际为 {overall.get('samples')}"
            )
        data["overall"][name] = _number(overall.get("cer"), f"{name}.overall.cer")

        forgetting = _require_mapping(
            experiment.get("forgetting"), f"{name}.forgetting"
        )
        data["forgetting"][name] = _number(
            forgetting.get("static_corrected"),
            f"{name}.forgetting.static_corrected",
        )

    if not math.isclose(data["forgetting"]["static"], 0.0, abs_tol=1e-12):
        raise ValueError("static 的 static-corrected forgetting 必须为 0")

    for name in POLICIES:
        queries = _require_mapping(
            experiments[name].get("feedback_queries"),
            f"{name}.feedback_queries",
        )
        if not queries.get("available"):
            raise ValueError(f"{name} 缺少反馈查询审计")
        for field in ("queries", "policy_queries", "policy_query_blocks"):
            if queries.get(field) != expected_policy_queries:
                raise ValueError(
                    f"{name}.{field} 必须为 {expected_policy_queries}，"
                    f"实际为 {queries.get(field)}"
                )
        if queries.get("manifest_queries") != 0:
            raise ValueError(f"{name} 不允许 manifest 预置反馈")
        if queries.get("max_policy_queries_per_block") != 1:
            raise ValueError(f"{name} 每个窗口最多只能查询一次")
        data["queries"][name] = _number(
            queries.get("queried_true_error_rate"),
            f"{name}.queried_true_error_rate",
        )

    for name, _, _ in METHODS[1:]:
        comparison = _comparison(comparisons, name, "static")
        point, lower, upper = _ci(
            comparison,
            f"{name}_minus_static.revisit_forgetting_difference",
            "revisit_forgetting_difference",
        )
        if not math.isclose(
            point, data["forgetting"][name], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{name} forgetting 与 paired comparison 点估计不一致"
            )
        data["comparisons"][f"{name}_forgetting"] = (point, lower, upper)

    for baseline in ("periodic", "random"):
        comparison = _comparison(comparisons, "uncertainty", baseline)
        data["comparisons"][f"uncertainty_minus_{baseline}"] = _ci(
            comparison, f"uncertainty_minus_{baseline}"
        )
    query_comparison = _comparison(comparisons, "uncertainty", "random").get(
        "query_error_rate_difference"
    )
    data["comparisons"]["query_uncertainty_minus_random"] = _ci(
        query_comparison, "query_uncertainty_minus_random"
    )
    return data


def _configure_matplotlib():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "绘图需要 matplotlib；请在独立绘图环境安装 matplotlib 后重试"
        ) from error
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
        -0.16,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def _errorbar(axis, y, point, lower, upper, color):
    axis.errorbar(
        point * 100,
        y,
        xerr=[[point * 100 - lower * 100], [upper * 100 - point * 100]],
        fmt="o",
        color=color,
        markersize=4,
        capsize=2.5,
        linewidth=1.1,
        zorder=3,
    )


def write_source_data(path, data):
    rows = []
    for name, _, _ in METHODS:
        rows.append(("a", name, "overall_cer", data["overall"][name], "", ""))
    for baseline in ("periodic", "random"):
        point, lower, upper = data["comparisons"][
            f"uncertainty_minus_{baseline}"
        ]
        rows.append(
            ("b", f"uncertainty_minus_{baseline}", "cer_difference", point, lower, upper)
        )
    for name in POLICIES:
        rows.append(
            ("c", name, "queried_true_error_rate", data["queries"][name], "", "")
        )
    point, lower, upper = data["comparisons"][
        "query_uncertainty_minus_random"
    ]
    rows.append(
        (
            "c",
            "uncertainty_minus_random",
            "query_error_rate_difference",
            point,
            lower,
            upper,
        )
    )
    for name, _, _ in METHODS[1:]:
        point, lower, upper = data["comparisons"][f"{name}_forgetting"]
        rows.append(
            ("d", name, "static_corrected_forgetting", point, lower, upper)
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("panel", "method", "metric", "point", "ci_lower", "ci_upper"))
        writer.writerows(rows)


def plot(data, output_prefix):
    plt = _configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.16, 5.15), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.16, 1.0))

    method_names = [name for name, _, _ in METHODS]
    labels = [label for _, label, _ in METHODS]
    colors = [color for _, _, color in METHODS]

    axis = figure.add_subplot(grid[0, 0])
    y = list(range(len(METHODS)))
    values = [data["overall"][name] * 100 for name in method_names]
    axis.scatter(values, y, color=colors, s=27, zorder=3)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Character error rate (%)  lower is better")
    axis.set_xlim(min(values) - 1.0, max(values) + 1.2)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6, zorder=0)
    for position, value in zip(y, values):
        axis.text(value + 0.08, position, f"{value:.2f}", va="center", fontsize=6.3)
    _panel_label(axis, "a")

    axis = figure.add_subplot(grid[0, 1])
    comparisons = (("vs periodic", "periodic"), ("vs random", "random"))
    for position, (_, baseline) in enumerate(comparisons):
        point, lower, upper = data["comparisons"][
            f"uncertainty_minus_{baseline}"
        ]
        _errorbar(axis, position, point, lower, upper, "#2A9D8F")
    axis.axvline(0, color="#4B5563", linewidth=0.8)
    axis.axvline(-0.3, color="#4C78A8", linewidth=0.8, linestyle="--")
    axis.set_yticks(range(len(comparisons)), [label for label, _ in comparisons])
    axis.invert_yaxis()
    axis.set_xlabel("Uncertainty CER difference (percentage points)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axis.text(
        -0.3,
        1.04,
        "pre-registered -0.3 pp gate",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="#4C78A8",
        fontsize=6,
    )
    _panel_label(axis, "b")

    axis = figure.add_subplot(grid[1, 0])
    policy_labels = [label for name, label, _ in METHODS if name in POLICIES]
    policy_colors = [color for name, _, color in METHODS if name in POLICIES]
    policy_values = [data["queries"][name] * 100 for name in POLICIES]
    axis.bar(range(len(POLICIES)), policy_values, color=policy_colors, width=0.62)
    axis.set_xticks(range(len(POLICIES)), policy_labels)
    axis.set_ylabel("Queried samples with a true error (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    for position, value in enumerate(policy_values):
        axis.text(position, value + 1.4, f"{value:.1f}", ha="center", fontsize=6.3)
    point, lower, upper = data["comparisons"][
        "query_uncertainty_minus_random"
    ]
    axis.text(
        0.02,
        0.98,
        f"Uncertainty - random: {point * 100:+.1f} pp\n"
        f"95% CI [{lower * 100:+.1f}, {upper * 100:+.1f}]",
        transform=axis.transAxes,
        va="top",
        fontsize=6.2,
    )
    _panel_label(axis, "c")

    axis = figure.add_subplot(grid[1, 1])
    forgetting_methods = METHODS[1:]
    for position, (name, _, color) in enumerate(forgetting_methods):
        point, lower, upper = data["comparisons"][f"{name}_forgetting"]
        _errorbar(axis, position, point, lower, upper, color)
    axis.axvline(0, color="#4B5563", linewidth=0.8)
    axis.set_yticks(
        range(len(forgetting_methods)), [label for _, label, _ in forgetting_methods]
    )
    axis.invert_yaxis()
    axis.set_xlabel("Static-corrected A2-A1 forgetting (pp)  lower is better")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
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
    if args.expected_samples < 1 or args.expected_policy_queries < 1:
        raise ValueError("expected samples 和 policy queries 必须大于 0")
    analysis = _read_analysis(args.analysis)
    data = validate_analysis(
        analysis,
        expected_samples=args.expected_samples,
        expected_policy_queries=args.expected_policy_queries,
        min_iterations=args.min_bootstrap_iterations,
    )
    plot(data, args.output_prefix)
    print(f"主动反馈论文图已写入：{args.output_prefix}.[pdf|svg|png|tiff]")


if __name__ == "__main__":
    main()
