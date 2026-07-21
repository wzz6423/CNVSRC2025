import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_NPZ_KEYS = {"signatures", "uids", "domains", "indices"}
ARTIFACT_KIND = "frozen_base_route_signatures"
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{description} 必须是 64 位小写十六进制 SHA-256"
        )
    return value


def _validate_finite_json(value, location="metadata"):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} 的键必须是字符串")
            _validate_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} 包含非有限数值")


def _required_integer(mapping, key, *, minimum=None, maximum=None):
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"metadata.{key} 必须是整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"metadata.{key} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"metadata.{key} 不能大于 {maximum}")
    return value


def _load_metadata(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到签名 metadata：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"签名 metadata 不是 UTF-8 文本：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"签名 metadata 不是有效 JSON：{path}（第 {error.lineno} 行）"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("签名 metadata 顶层必须是 JSON 对象")
    _validate_finite_json(value)
    if _required_integer(value, "schema_version", minimum=1) != 1:
        raise ValueError("仅支持 schema_version=1 的签名 metadata")
    _required_integer(value, "samples", minimum=1)
    _required_integer(value, "signature_dim", minimum=1)
    _required_integer(value, "motion_order", minimum=0, maximum=2)
    if value.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError(
            f"metadata.artifact_kind 必须是 {ARTIFACT_KIND!r}"
        )
    _require_sha256(value.get("artifact_sha256"), "metadata.artifact_sha256")
    _require_sha256(value.get("manifest_sha256"), "metadata.manifest_sha256")
    _require_sha256(
        value.get("base_checkpoint_sha256"),
        "metadata.base_checkpoint_sha256",
    )
    if "feature_dim" in value:
        _required_integer(value, "feature_dim", minimum=1)
    return value


def load_signature_artifacts(artifact_path, metadata_path=None):
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"找不到路由签名 NPZ：{artifact_path}")
    if artifact_path.suffix != ".npz":
        raise ValueError(f"路由签名输入必须是 .npz 文件：{artifact_path}")
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else Path(f"{artifact_path}.meta.json")
    )
    metadata = _load_metadata(metadata_path)
    artifact_sha256 = _file_sha256(artifact_path)
    if artifact_sha256 != metadata["artifact_sha256"]:
        raise ValueError(
            "签名 artifact SHA-256 不匹配："
            f"期望 {metadata['artifact_sha256']}，实际 {artifact_sha256}"
        )

    try:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            keys = set(artifact.files)
            if keys != REQUIRED_NPZ_KEYS:
                missing = sorted(REQUIRED_NPZ_KEYS - keys)
                extra = sorted(keys - REQUIRED_NPZ_KEYS)
                details = []
                if missing:
                    details.append(f"缺少 {missing}")
                if extra:
                    details.append(f"多出 {extra}")
                raise ValueError("路由签名 NPZ keys 无效：" + "；".join(details))
            signatures = artifact["signatures"]
            uids = artifact["uids"]
            domains = artifact["domains"]
            indices = artifact["indices"]
    except ValueError as error:
        if str(error).startswith("路由签名 NPZ keys 无效"):
            raise
        raise ValueError(
            f"无法安全读取路由签名 NPZ（禁止 pickle/object 数组）：{artifact_path}"
        ) from error
    except (OSError, EOFError) as error:
        raise ValueError(f"路由签名 NPZ 已损坏或不可读：{artifact_path}") from error

    if signatures.dtype != np.float32:
        raise ValueError(
            f"signatures dtype 必须是 float32，实际为 {signatures.dtype}；请重新导出"
        )
    if signatures.ndim != 2 or min(signatures.shape) < 1:
        raise ValueError("signatures 必须是非空的 [N, D] 二维数组")
    if not np.isfinite(signatures).all():
        raise ValueError("signatures 包含 NaN 或 Inf，请重新导出")
    row_norms = np.linalg.norm(signatures, axis=1)
    if np.any(row_norms <= np.finfo(np.float32).eps):
        raise ValueError("signatures 包含零向量，无法进行余弦路由校准")

    for name, values in (("uids", uids), ("domains", domains)):
        if values.ndim != 1 or values.dtype.kind != "U":
            raise ValueError(f"{name} 必须是一维 Unicode 字符串数组")
        if any(not value for value in values.tolist()):
            raise ValueError(f"{name} 不能包含空字符串")
    if indices.ndim != 1 or indices.dtype != np.int64:
        raise ValueError("indices 必须是一维 int64 数组")

    sample_count, signature_dim = signatures.shape
    if not all(len(values) == sample_count for values in (uids, domains, indices)):
        raise ValueError("signatures、uids、domains、indices 的样本数不一致")
    if len(set(uids.tolist())) != sample_count:
        raise ValueError("uids 必须全局唯一；请检查 manifest 或重新导出")
    if len(set(indices.tolist())) != sample_count or np.any(indices < 0):
        raise ValueError("indices 必须是唯一的非负整数")
    if metadata["samples"] != sample_count:
        raise ValueError(
            f"metadata.samples={metadata['samples']} 与 NPZ 样本数 {sample_count} 不一致"
        )
    if metadata["signature_dim"] != signature_dim:
        raise ValueError(
            "metadata.signature_dim="
            f"{metadata['signature_dim']} 与 NPZ 维度 {signature_dim} 不一致"
        )

    return {
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "artifact_sha256": artifact_sha256,
        "metadata_sha256": _file_sha256(metadata_path),
        "signatures": signatures,
        "uids": uids,
        "domains": domains,
        "indices": indices,
        "metadata": metadata,
    }


def _resolve_feature_dim(metadata, requested_feature_dim):
    source_order = metadata["motion_order"]
    signature_dim = metadata["signature_dim"]
    metadata_feature_dim = metadata.get("feature_dim")
    if requested_feature_dim is not None:
        if (
            not isinstance(requested_feature_dim, int)
            or isinstance(requested_feature_dim, bool)
            or requested_feature_dim < 1
        ):
            raise ValueError("feature_dim 必须是大于 0 的整数")
        feature_dim = requested_feature_dim
        if metadata_feature_dim is not None and feature_dim != metadata_feature_dim:
            raise ValueError(
                f"指定 feature_dim={feature_dim} 与 metadata.feature_dim="
                f"{metadata_feature_dim} 不一致"
            )
    elif metadata_feature_dim is not None:
        feature_dim = metadata_feature_dim
    else:
        blocks = 2 + source_order
        if signature_dim % blocks:
            raise ValueError(
                "无法从 signature_dim 和 motion_order 推断 feature_dim；"
                "请使用 --feature-dim 显式指定"
            )
        feature_dim = signature_dim // blocks
    expected_dim = feature_dim * (2 + source_order)
    if signature_dim != expected_dim:
        raise ValueError(
            f"签名维度应为 feature_dim*(2+motion_order)={expected_dim}，"
            f"实际为 {signature_dim}"
        )
    return feature_dim


def _domain_seed(seed, domain):
    digest = hashlib.sha256(f"{seed}\0{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _deterministic_split(uids, domains, indices, seed, reference_fraction):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed 必须是整数")
    if (
        not isinstance(reference_fraction, (int, float))
        or isinstance(reference_fraction, bool)
        or not math.isfinite(float(reference_fraction))
        or not 0.0 < float(reference_fraction) < 1.0
    ):
        raise ValueError("reference_fraction 必须是 0 到 1 之间的有限数值")
    reference_fraction = float(reference_fraction)
    domain_names = sorted(set(domains.tolist()))
    if len(domain_names) < 2:
        raise ValueError("路由校准至少需要 2 个 domain，才能构造跨域 unknown 分数")

    reference_positions = []
    eval_positions = []
    split_domains = {}
    for domain in domain_names:
        positions = np.flatnonzero(domains == domain).tolist()
        positions.sort(key=lambda position: (int(indices[position]), uids[position]))
        if len(positions) < 2:
            raise ValueError(
                f"domain={domain!r} 只有 {len(positions)} 条样本；"
                "每域至少需要 2 条以拆分 reference/eval"
            )
        order = np.random.default_rng(_domain_seed(seed, domain)).permutation(
            len(positions)
        )
        reference_count = math.floor(len(positions) * reference_fraction)
        reference_count = min(max(reference_count, 1), len(positions) - 1)
        selected = {positions[int(offset)] for offset in order[:reference_count]}
        domain_reference = [position for position in positions if position in selected]
        domain_eval = [position for position in positions if position not in selected]
        reference_positions.extend(domain_reference)
        eval_positions.extend(domain_eval)
        split_domains[domain] = {
            "samples": len(positions),
            "reference_samples": len(domain_reference),
            "eval_samples": len(domain_eval),
            "reference_uids": [uids[position] for position in domain_reference],
            "eval_uids": [uids[position] for position in domain_eval],
        }
    return {
        "domain_names": domain_names,
        "reference_positions": np.asarray(reference_positions, dtype=np.int64),
        "eval_positions": np.asarray(eval_positions, dtype=np.int64),
        "summary": {
            "seed": seed,
            "reference_fraction": reference_fraction,
            "reference_samples": len(reference_positions),
            "eval_samples": len(eval_positions),
            "domains": split_domains,
        },
    }


def _normalize_rows(values, *, description):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{description} 包含零向量，无法 L2 normalize")
    return values / norms


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("分布统计需要非空的一维有限数值")
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            label: float(np.quantile(values, probability))
            for label, probability in QUANTILES
        },
    }


def unknown_auroc(known_similarities, unknown_similarities):
    known = np.asarray(known_similarities, dtype=np.float64)
    unknown = np.asarray(unknown_similarities, dtype=np.float64)
    if known.ndim != 1 or unknown.ndim != 1 or not known.size or not unknown.size:
        raise ValueError("AUROC 需要非空的一维 known/unknown 相似度")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("AUROC 输入包含非有限相似度")

    scores = np.concatenate((1.0 - known, 1.0 - unknown))
    labels = np.concatenate(
        (np.zeros(known.size, dtype=np.int8), np.ones(unknown.size, dtype=np.int8))
    )
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    positive_count = unknown.size
    negative_count = known.size
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(mann_whitney / (positive_count * negative_count))


def _threshold_candidates(similarities):
    unique = sorted(set(float(value) for value in similarities))
    candidates = [-1.0]
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    )
    candidates.append(1.0)
    return sorted(set(min(1.0, max(-1.0, value)) for value in candidates))


def select_route_threshold(known_similarities, unknown_similarities):
    known = np.asarray(known_similarities, dtype=np.float64)
    unknown = np.asarray(unknown_similarities, dtype=np.float64)
    if known.ndim != 1 or unknown.ndim != 1 or not known.size or not unknown.size:
        raise ValueError("阈值扫描需要非空的一维 known/unknown 相似度")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("阈值扫描输入包含非有限相似度")

    similarities = np.concatenate((known, unknown))
    labels = np.concatenate(
        (np.zeros(known.size, dtype=np.int8), np.ones(unknown.size, dtype=np.int8))
    )
    best = None
    for threshold in _threshold_candidates(similarities):
        predicted_unknown = similarities < threshold
        true_positive = int(np.sum(predicted_unknown & (labels == 1)))
        false_positive = int(np.sum(predicted_unknown & (labels == 0)))
        false_negative = int(np.sum(~predicted_unknown & (labels == 1)))
        true_negative = int(np.sum(~predicted_unknown & (labels == 0)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / unknown.size
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        candidate = {
            "route_threshold": float(threshold),
            "best_f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        }
        key = (f1, precision, recall, -threshold)
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1]


def _clustering_purity(predicted, true_domain_indices, domain_count):
    dominant_total = 0
    for cluster_index in range(domain_count):
        members = true_domain_indices[predicted == cluster_index]
        if members.size:
            dominant_total += max(Counter(members.tolist()).values())
    return dominant_total / len(predicted)


def _evaluate_order(
    signatures,
    domains,
    split,
    *,
    feature_dim,
    motion_order,
):
    prefix_dim = feature_dim * (2 + motion_order)
    normalized = _normalize_rows(
        signatures[:, :prefix_dim],
        description=f"motion_order={motion_order} 的签名前缀",
    )
    domain_names = split["domain_names"]
    domain_to_index = {domain: index for index, domain in enumerate(domain_names)}
    reference_positions = split["reference_positions"]
    eval_positions = split["eval_positions"]

    prototypes = []
    for domain in domain_names:
        positions = reference_positions[domains[reference_positions] == domain]
        prototype = normalized[positions].mean(axis=0, keepdims=True)
        prototypes.append(
            _normalize_rows(
                prototype,
                description=f"domain={domain!r} 的 reference prototype",
            )[0]
        )
    prototypes = np.stack(prototypes)
    eval_signatures = normalized[eval_positions]
    similarities = eval_signatures @ prototypes.T
    true_indices = np.asarray(
        [domain_to_index[domain] for domain in domains[eval_positions]],
        dtype=np.int64,
    )
    predicted = np.argmax(similarities, axis=1)
    own = similarities[np.arange(len(eval_positions)), true_indices]
    cross = similarities.copy()
    cross[np.arange(len(eval_positions)), true_indices] = -np.inf
    max_cross = cross.max(axis=1)
    sorted_similarities = np.sort(similarities, axis=1)
    margins = sorted_similarities[:, -1] - sorted_similarities[:, -2]
    route_counts = Counter(predicted.tolist())
    threshold = select_route_threshold(own, max_cross)
    auroc = unknown_auroc(own, max_cross)

    per_domain = {}
    for domain, domain_index in domain_to_index.items():
        mask = true_indices == domain_index
        domain_predicted = predicted[mask]
        per_domain[domain] = {
            "eval_samples": int(mask.sum()),
            "closed_set_accuracy": float(np.mean(domain_predicted == domain_index)),
            "predicted_domain_counts": {
                domain_names[index]: int(count)
                for index, count in sorted(Counter(domain_predicted.tolist()).items())
            },
            "own_prototype_within_similarity": _distribution(own[mask]),
            "excluding_own_max_cross_similarity": _distribution(max_cross[mask]),
        }

    return {
        "motion_order": motion_order,
        "signature_dim": prefix_dim,
        "reference_samples": int(len(reference_positions)),
        "eval_samples": int(len(eval_positions)),
        "prototype_domains": domain_names,
        "closed_set_accuracy": float(np.mean(predicted == true_indices)),
        "clustering_purity": float(
            _clustering_purity(predicted, true_indices, len(domain_names))
        ),
        "predicted_domain_counts": {
            domain_names[index]: int(route_counts[index])
            for index in sorted(route_counts)
        },
        "top1_top2_margin": _distribution(margins),
        "own_prototype_within_similarity": _distribution(own),
        "excluding_own_max_cross_similarity": _distribution(max_cross),
        "unknown_detection": {
            "protocol": "leave_own_domain_out_proxy",
            "limitation": (
                "Cross-domain scores reuse each eval sample against non-own "
                "prototypes; this is not a true unseen-domain evaluation."
            ),
            "positive_class": "unknown",
            "novelty_score": "1 - prototype_similarity (higher means more unknown)",
            "known_scores": "eval sample similarity to its own-domain prototype",
            "unknown_scores": (
                "eval sample maximum similarity to prototypes excluding its own domain"
            ),
            "prediction_rule": "unknown when prototype_similarity < route_threshold",
            "auroc": auroc,
            **threshold,
            "threshold_tie_break": [
                "higher_f1",
                "higher_precision",
                "higher_recall",
                "lower_route_threshold",
            ],
        },
        "domains": per_domain,
    }


def calibrate_signature_artifacts(
    artifact_path,
    *,
    expected_manifest_sha256,
    expected_base_checkpoint_sha256,
    metadata_path=None,
    seed=42,
    reference_fraction=0.5,
    feature_dim=None,
):
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected manifest SHA-256",
    )
    expected_base_checkpoint_sha256 = _require_sha256(
        expected_base_checkpoint_sha256,
        "expected base checkpoint SHA-256",
    )
    loaded = load_signature_artifacts(artifact_path, metadata_path)
    metadata = loaded["metadata"]
    if metadata["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(
            "manifest SHA-256 不匹配："
            f"期望 {expected_manifest_sha256}，"
            f"签名产物为 {metadata['manifest_sha256']}"
        )
    if metadata["base_checkpoint_sha256"] != expected_base_checkpoint_sha256:
        raise ValueError(
            "base checkpoint SHA-256 不匹配："
            f"期望 {expected_base_checkpoint_sha256}，"
            f"签名产物为 {metadata['base_checkpoint_sha256']}"
        )
    feature_dim = _resolve_feature_dim(metadata, feature_dim)
    split = _deterministic_split(
        loaded["uids"],
        loaded["domains"],
        loaded["indices"],
        seed,
        reference_fraction,
    )
    evaluated_orders = list(range(metadata["motion_order"] + 1))
    order_results = {
        str(order): _evaluate_order(
            loaded["signatures"],
            loaded["domains"],
            split,
            feature_dim=feature_dim,
            motion_order=order,
        )
        for order in evaluated_orders
    }
    best_order = max(
        evaluated_orders,
        key=lambda order: (
            order_results[str(order)]["unknown_detection"]["best_f1"],
            order_results[str(order)]["unknown_detection"]["auroc"],
            order_results[str(order)]["clustering_purity"],
            -order,
        ),
    )
    return {
        "schema_version": 1,
        "source": {
            "artifact": str(loaded["artifact_path"]),
            "metadata": str(loaded["metadata_path"]),
            "artifact_sha256": loaded["artifact_sha256"],
            "metadata_sha256": loaded["metadata_sha256"],
            "samples": metadata["samples"],
            "source_motion_order": metadata["motion_order"],
            "source_signature_dim": metadata["signature_dim"],
            "feature_dim": feature_dim,
            "stream_manifest": metadata.get("stream_manifest"),
            "manifest_sha256": metadata.get("manifest_sha256"),
            "base_checkpoint": metadata.get("base_checkpoint"),
            "base_checkpoint_sha256": metadata.get("base_checkpoint_sha256"),
            "git_commit": metadata.get("git_commit"),
        },
        "split": split["summary"],
        "evaluated_motion_orders": evaluated_orders,
        "best_motion_order": best_order,
        "best_order_selection": [
            "higher_unknown_f1",
            "higher_unknown_auroc",
            "higher_clustering_purity",
            "lower_motion_order",
        ],
        "orders": order_results,
    }


def write_calibration_json(path, result):
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("校准输出路径必须以 .json 结尾")
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有校准结果：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                result,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"拒绝覆盖已有校准结果：{path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
