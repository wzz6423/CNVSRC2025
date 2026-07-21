import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from numbers import Real


QUANTILES = (
    ("p0", 0.0),
    ("p10", 0.1),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p90", 0.9),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p100", 1.0),
)


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * weight


def _missing(index, field):
    raise ValueError(f"第 {index} 条记录缺少字段 {field}")


def _validate_record(record, index):
    if not isinstance(record, Mapping):
        raise ValueError(f"第 {index} 条记录必须是 JSON 对象")
    if "domain" not in record:
        _missing(index, "domain")
    domain = record["domain"]
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"第 {index} 条记录的 domain 必须是非空字符串")
    if "route" not in record:
        _missing(index, "route")
    route = record["route"]
    if not isinstance(route, Mapping):
        raise ValueError(f"第 {index} 条记录的 route 必须是 JSON 对象")

    values = {}
    for field in ("expert_index", "similarity", "created", "quarantined"):
        if field not in route:
            _missing(index, f"route.{field}")
        values[field] = route[field]

    expert_index = values["expert_index"]
    if (
        not isinstance(expert_index, int)
        or isinstance(expert_index, bool)
        or expert_index < 0
    ):
        raise ValueError(
            f"第 {index} 条记录的 route.expert_index 必须是非负整数"
        )
    similarity = values["similarity"]
    if (
        not isinstance(similarity, Real)
        or isinstance(similarity, bool)
        or not math.isfinite(float(similarity))
    ):
        raise ValueError(
            f"第 {index} 条记录的 route.similarity 必须是有限数值"
        )
    for field in ("created", "quarantined"):
        if not isinstance(values[field], bool):
            raise ValueError(
                f"第 {index} 条记录的 route.{field} 必须是布尔值"
            )
    return (
        domain,
        expert_index,
        float(similarity),
        values["created"],
        values["quarantined"],
    )


def _count_dict(counter):
    return {str(key): counter[key] for key in sorted(counter)}


def _share_dict(counter, total):
    return {str(key): counter[key] / total for key in sorted(counter)}


def summarize_route_records(records: Iterable[Mapping], threshold=0.9):
    """汇总路由记录；reuse 指当次未创建新专家而复用已有专家。"""
    if (
        not isinstance(threshold, Real)
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("路由阈值必须是 -1 到 1 之间的有限数值")
    threshold = float(threshold)
    normalized = [
        _validate_record(record, index)
        for index, record in enumerate(records, start=1)
    ]
    if not normalized:
        raise ValueError("路由诊断至少需要一条记录")

    samples = len(normalized)
    similarities = sorted(record[2] for record in normalized)
    below_threshold_count = sum(value < threshold for value in similarities)
    created_count = sum(record[3] for record in normalized)
    quarantined_count = sum(record[4] for record in normalized)
    route_counts = Counter(record[1] for record in normalized)
    domain_routes = defaultdict(Counter)
    expert_domains = defaultdict(Counter)
    domain_created = Counter()
    domain_quarantined = Counter()
    for domain, expert_index, _similarity, created, quarantined in normalized:
        domain_routes[domain][expert_index] += 1
        expert_domains[expert_index][domain] += 1
        domain_created[domain] += int(created)
        domain_quarantined[domain] += int(quarantined)

    domains = {}
    domain_consistency_total = 0
    for domain in sorted(domain_routes):
        counts = domain_routes[domain]
        domain_samples = sum(counts.values())
        dominant_expert, dominant_count = min(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        unique_experts = len(counts)
        reuse_count = domain_samples - domain_created[domain]
        domain_consistency_total += dominant_count
        domains[domain] = {
            "samples": domain_samples,
            "route_counts": _count_dict(counts),
            "route_share": _share_dict(counts, domain_samples),
            "unique_experts": unique_experts,
            "dominant_expert": dominant_expert,
            "route_purity": dominant_count / domain_samples,
            "reuse_count": reuse_count,
            "reuse_rate": reuse_count / domain_samples,
            "created_count": domain_created[domain],
            "quarantined_count": domain_quarantined[domain],
        }

    experts = {}
    clustering_purity_total = 0
    for expert_index in sorted(expert_domains):
        counts = expert_domains[expert_index]
        expert_samples = sum(counts.values())
        dominant_domain, dominant_count = min(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        clustering_purity_total += dominant_count
        experts[str(expert_index)] = {
            "samples": expert_samples,
            "domain_counts": {domain: counts[domain] for domain in sorted(counts)},
            "domain_share": {
                domain: counts[domain] / expert_samples for domain in sorted(counts)
            },
            "dominant_domain": dominant_domain,
            "domain_purity": dominant_count / expert_samples,
        }

    return {
        "schema_version": 1,
        "samples": samples,
        "threshold": threshold,
        "similarity": {
            "min": similarities[0],
            "max": similarities[-1],
            "mean": sum(similarities) / samples,
            "quantiles": {
                label: _quantile(similarities, probability)
                for label, probability in QUANTILES
            },
            "below_threshold_count": below_threshold_count,
            "below_threshold_rate": below_threshold_count / samples,
        },
        "created_count": created_count,
        "created_rate": created_count / samples,
        "quarantined_count": quarantined_count,
        "quarantined_rate": quarantined_count / samples,
        "route_counts": _count_dict(route_counts),
        "route_share": _share_dict(route_counts, samples),
        "domain_route_consistency": domain_consistency_total / samples,
        "clustering_purity": clustering_purity_total / samples,
        "domains": domains,
        "experts": experts,
    }
