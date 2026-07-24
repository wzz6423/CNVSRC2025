import json
import math
import os
import re
import shutil
from pathlib import Path


_METRIC_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_jsonl_atomic(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _history_step(record, line_number=None):
    step = record.get("processed_samples") if isinstance(record, dict) else None
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        location = f"第 {line_number} 行" if line_number is not None else "记录"
        raise ValueError(f"指标历史{location}缺少非负整数 processed_samples")
    return step


def _read_metrics_history(path):
    records = []
    previous_step = -1
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"指标历史第 {line_number} 行不是有效 JSON"
                ) from error
            step = _history_step(record, line_number)
            if step <= previous_step:
                raise ValueError("指标历史 processed_samples 必须严格递增")
            records.append(record)
            previous_step = step
    return records


def append_metrics_history(path, record):
    path = Path(path)
    step = _history_step(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        records = _read_metrics_history(path)
        if records and step <= records[-1]["processed_samples"]:
            raise ValueError("新增指标历史必须晚于已有记录")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_metrics_history(path, processed_samples):
    path = Path(path)
    processed_samples = int(processed_samples)
    if processed_samples < 0:
        raise ValueError("恢复样本数不能为负数")
    if not path.is_file():
        return 0

    records = _read_metrics_history(path)
    retained = [
        record
        for record in records
        if record["processed_samples"] <= processed_samples
    ]
    if len(retained) != len(records):
        temporary_path = path.with_name(f".{path.name}.resume.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for record in retained:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return retained[-1]["processed_samples"] if retained else 0


def reset_best_checkpoints(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return
    (directory / "index.json").unlink(missing_ok=True)
    for path in directory.glob("*.pt"):
        path.unlink()
    for path in directory.glob(".*.tmp"):
        path.unlink()


def _load_checkpoint_index(path, metric_name, mode):
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("最佳 checkpoint 索引不是有效 JSON") from error
    if (
        value.get("schema_version") != 1
        or value.get("metric") != metric_name
        or value.get("mode") != mode
    ):
        raise ValueError("最佳 checkpoint 索引与当前保留策略不兼容")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("最佳 checkpoint 索引缺少 checkpoints 列表")
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise ValueError("最佳 checkpoint 索引包含无效记录")
        name = entry.get("path")
        step = entry.get("processed_samples")
        score = entry.get("score")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("最佳 checkpoint 索引包含无效字段")
    return checkpoints


def retain_best_checkpoints(
    checkpoint_path,
    directory,
    *,
    score,
    processed_samples,
    keep=3,
    metric_name="cer",
    mode="min",
):
    checkpoint_path = Path(checkpoint_path)
    directory = Path(directory)
    processed_samples = int(processed_samples)
    keep = int(keep)
    score = float(score)
    if keep < 1:
        raise ValueError("最佳 checkpoint 保留数量必须大于 0")
    if processed_samples < 0:
        raise ValueError("checkpoint 样本数不能为负数")
    if not math.isfinite(score):
        raise ValueError("checkpoint 排序指标必须是有限数值")
    if not _METRIC_NAME.fullmatch(str(metric_name)):
        raise ValueError("checkpoint 指标名只能包含字母、数字、下划线和连字符")
    if mode not in {"min", "max"}:
        raise ValueError("checkpoint 排序模式必须是 min 或 max")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到待保留的 checkpoint：{checkpoint_path}")

    directory.mkdir(parents=True, exist_ok=True)
    for temporary_path in directory.glob(".*.tmp"):
        temporary_path.unlink()
    index_path = directory / "index.json"
    entries = [
        entry
        for entry in _load_checkpoint_index(index_path, metric_name, mode)
        if entry["processed_samples"] != processed_samples
    ]
    filename = (
        f"step_{processed_samples:08d}_{metric_name}_{score:.8f}.pt"
    )
    candidate = {
        "path": filename,
        "processed_samples": processed_samples,
        "score": score,
    }
    entries.append(candidate)
    direction = 1.0 if mode == "min" else -1.0
    retained = sorted(
        entries,
        key=lambda entry: (
            direction * entry["score"],
            -entry["processed_samples"],
        ),
    )[:keep]

    if candidate in retained:
        target_path = directory / filename
        temporary_path = target_path.with_name(f".{target_path.name}.tmp")
        try:
            shutil.copy2(checkpoint_path, temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    for entry in retained:
        if not (directory / entry["path"]).is_file():
            raise FileNotFoundError(
                f"最佳 checkpoint 索引引用了缺失文件：{entry['path']}"
            )
    _write_json_atomic(
        index_path,
        {
            "schema_version": 1,
            "metric": metric_name,
            "mode": mode,
            "selection_scope": "cumulative_prequential",
            "keep": keep,
            "checkpoints": retained,
        },
    )

    retained_names = {entry["path"] for entry in retained}
    for path in directory.glob("*.pt"):
        if path.name not in retained_names:
            path.unlink()
    return retained
