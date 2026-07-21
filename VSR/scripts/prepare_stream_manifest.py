import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path


def _validate_revisit_domains(domains, error_type):
    if len(domains) != 3:
        raise error_type("--revisit-domains 必须恰好 3 个域")
    if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
        raise error_type("回访域不能为空")
    normalized = tuple(domain.strip() for domain in domains)
    if len(set(normalized)) != 3:
        raise error_type("3 个回访域必须互不相同")
    return normalized


def parse_revisit_domains(value):
    try:
        domains = next(csv.reader([value], strict=True))
    except csv.Error as error:
        raise argparse.ArgumentTypeError("回访域 CSV 格式无效") from error
    return _validate_revisit_domains(domains, argparse.ArgumentTypeError)


def parse_args():
    parser = argparse.ArgumentParser(description="生成持续适应流式清单")
    parser.add_argument("--csv", required=True, help="CNVSRC 四列 CSV")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument(
        "--domain-regex",
        help="从相对视频路径提取域；优先使用命名组 domain，其次使用第一个组",
    )
    parser.add_argument(
        "--order",
        choices=("original", "domain-block", "round-robin"),
        default="domain-block",
    )
    parser.add_argument("--shuffle-within-domain", action="store_true")
    parser.add_argument("--shuffle-domains", action="store_true")
    parser.add_argument(
        "--revisit-domains",
        type=parse_revisit_domains,
        help="以 CSV 指定 3 个互异域 A,B,C，输出 A1-B-C-A2 回访子流",
    )
    parser.add_argument("--feedback-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-metadata-csv",
        help="按样本 ID 提供原始文本的元数据 CSV；指定后忽略源 CSV 第四列并重编码",
    )
    parser.add_argument("--metadata-id-column", default="ID")
    parser.add_argument("--metadata-text-column", default="TEXT")
    parser.add_argument(
        "--target-vocab",
        help="目标词表 char_units.txt（每行“token id”）；使用元数据时必填",
    )
    parser.add_argument(
        "--oov-policy",
        choices=("error", "drop"),
        default="error",
        help="目标词表外字符策略：error 直接失败，drop 丢弃该字符",
    )
    return parser.parse_args()


def domain_from_path(relative_path, pattern):
    if pattern is None:
        parts = Path(relative_path).parts
        return parts[0] if parts else "unknown"
    match = pattern.search(relative_path)
    if match is None:
        raise ValueError(f"视频路径与域正则不匹配：{relative_path}")
    if "domain" in match.groupdict():
        return match.group("domain")
    if match.groups():
        return match.group(1)
    return match.group(0)


def ordered_rows(rows, order, shuffle_within_domain, shuffle_domains, rng):
    if order == "original":
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    if shuffle_within_domain:
        for values in grouped.values():
            rng.shuffle(values)
    domains = sorted(grouped)
    if shuffle_domains:
        rng.shuffle(domains)
    if order == "domain-block":
        return [row for domain in domains for row in grouped[domain]]
    queues = {domain: deque(grouped[domain]) for domain in domains}
    output = []
    while any(queues.values()):
        for domain in domains:
            if queues[domain]:
                output.append(queues[domain].popleft())
    return output


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_lines_atomic(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def load_target_vocab(path):
    """解析 char_units.txt 风格目标词表，返回 (token->id, id->token)。"""
    token_to_id = {}
    id_to_token = {}
    with Path(path).open(encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"目标词表第 {index} 行格式应为“token id”：{raw_line!r}"
                )
            token, id_string = parts
            if not (id_string.isdigit() and int(id_string) > 0):
                raise ValueError(
                    f"目标词表第 {index} 行 ID 必须为正整数：{raw_line!r}"
                )
            token_id = int(id_string)
            if token in token_to_id:
                raise ValueError(f"目标词表 token 重复：{token!r}")
            if token_id in id_to_token:
                raise ValueError(f"目标词表 ID 重复：{token_id}")
            expected_id = len(token_to_id) + 1
            if token_id != expected_id:
                raise ValueError(
                    f"目标词表 ID 必须从 1 按行连续递增，"
                    f"第 {index} 行应为 {expected_id}，实际为 {token_id}"
                )
            token_to_id[token] = token_id
            id_to_token[token_id] = token
    if not token_to_id:
        raise ValueError(f"目标词表为空：{path}")
    return token_to_id, id_to_token


def load_text_metadata(path, id_column, text_column):
    """读取带表头的元数据 CSV（utf-8-sig），返回 ID->原始文本映射。"""
    mapping = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"元数据 CSV 缺少表头：{path}")
        for column in (id_column, text_column):
            if column not in reader.fieldnames:
                raise ValueError(f"元数据 CSV 缺少列 {column!r}：{path}")
        for line_number, record in enumerate(reader, start=2):
            sample_id = record[id_column]
            if sample_id is None:
                raise ValueError(f"元数据第 {line_number} 行缺少 ID 值")
            if sample_id in mapping:
                raise ValueError(f"元数据 ID 重复：{sample_id!r}")
            text = record[text_column]
            if text is None:
                raise ValueError(f"元数据第 {line_number} 行缺少文本值")
            mapping[sample_id] = text
    return mapping


def encode_text(raw_text, token_to_id, oov_policy, dropped_counter):
    """逐字符按目标词表重编码；返回 (规范化文本, token 列表)。"""
    kept_characters = []
    tokens = []
    for character in raw_text:
        token_id = token_to_id.get(character)
        if token_id is None:
            if oov_policy == "error":
                raise ValueError(f"字符不在目标词表中：{character!r}")
            dropped_counter[character] += 1
            continue
        kept_characters.append(character)
        tokens.append(token_id)
    return "".join(kept_characters), tokens


def build_metadata_sidecar(
    args,
    samples,
    raw_characters,
    target_characters,
    dropped_counter,
):
    dropped_characters = sum(dropped_counter.values())
    ordered_dropped = OrderedDict(
        sorted(dropped_counter.items(), key=lambda item: (-item[1], item[0]))
    )
    return OrderedDict(
        [
            ("schema_version", 1),
            ("label_mode", "metadata_reencoded"),
            ("samples", samples),
            ("source_csv", str(Path(args.csv))),
            ("source_csv_sha256", _file_sha256(args.csv)),
            ("text_metadata_csv", str(Path(args.text_metadata_csv))),
            ("text_metadata_csv_sha256", _file_sha256(args.text_metadata_csv)),
            ("target_vocab", str(Path(args.target_vocab))),
            ("target_vocab_sha256", _file_sha256(args.target_vocab)),
            ("oov_policy", args.oov_policy),
            ("raw_characters", raw_characters),
            ("target_characters", target_characters),
            ("dropped_characters", dropped_characters),
            (
                "dropped_rate",
                dropped_characters / raw_characters if raw_characters else 0.0,
            ),
            ("distinct_dropped_characters", len(dropped_counter)),
            ("dropped_character_counts", ordered_dropped),
        ]
    )


def build_passthrough_sidecar(args, samples):
    return OrderedDict(
        [
            ("schema_version", 1),
            ("label_mode", "token_passthrough"),
            ("samples", samples),
            ("source_csv", str(Path(args.csv))),
            ("source_csv_sha256", _file_sha256(args.csv)),
            ("target_vocab", str(Path(args.target_vocab))),
            ("target_vocab_sha256", _file_sha256(args.target_vocab)),
        ]
    )


def run(args):
    if args.text_metadata_csv and not args.target_vocab:
        raise ValueError("指定 --text-metadata-csv 时必须提供 --target-vocab")

    revisit_domains = getattr(args, "revisit_domains", None)
    if revisit_domains is not None:
        if isinstance(revisit_domains, str):
            try:
                revisit_domains = parse_revisit_domains(revisit_domains)
            except argparse.ArgumentTypeError as error:
                raise ValueError(str(error)) from error
        else:
            try:
                revisit_domains = tuple(revisit_domains)
            except TypeError as error:
                raise ValueError("--revisit-domains 必须是 3 个域") from error
            revisit_domains = _validate_revisit_domains(revisit_domains, ValueError)
        if not args.target_vocab:
            raise ValueError("回访模式必须提供 --target-vocab")
        if args.order != "domain-block":
            raise ValueError("回访模式要求 --order domain-block")
        if args.shuffle_domains:
            raise ValueError("回访模式不能同时使用 --shuffle-domains")

    rng = random.Random(args.seed)
    pattern = re.compile(args.domain_regex) if args.domain_regex else None
    use_metadata = bool(args.text_metadata_csv)

    token_to_id = id_to_token = metadata = None
    dropped_counter = Counter()
    raw_characters = 0
    target_characters = 0
    if args.target_vocab:
        token_to_id, id_to_token = load_target_vocab(args.target_vocab)
    if use_metadata:
        metadata = load_text_metadata(
            args.text_metadata_csv,
            args.metadata_id_column,
            args.metadata_text_column,
        )

    rows = []
    minimum_columns = 2 if use_metadata else 4
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        for index, values in enumerate(csv.reader(handle)):
            if len(values) < minimum_columns:
                raise ValueError(f"第 {index + 1} 行少于 {minimum_columns} 列")
            dataset, relative_path = values[0], values[1]
            relative_without_suffix = Path(relative_path).with_suffix("").as_posix()
            row = OrderedDict(
                [
                    ("uid", f"{dataset}:{relative_without_suffix}"),
                    ("video", str(Path(dataset) / relative_path)),
                ]
            )
            if revisit_domains:
                row["domain"] = domain_from_path(relative_path, pattern)
                if use_metadata:
                    row["_sample_id"] = Path(relative_path).stem
                else:
                    row["_token_string"] = values[3]
                    row["_source_line_number"] = index + 1
                row["feedback"] = False
                rows.append(row)
                continue
            if use_metadata:
                sample_id = Path(relative_path).stem
                if sample_id not in metadata:
                    raise ValueError(f"元数据缺少样本：{sample_id}")
                raw_target_text = metadata[sample_id]
                raw_characters += len(raw_target_text)
                target_text, target_tokens = encode_text(
                    raw_target_text, token_to_id, args.oov_policy, dropped_counter
                )
                if not target_tokens:
                    raise ValueError(f"样本 {sample_id} 的标签规范化后为空")
                target_characters += len(target_text)
                decoded = "".join(id_to_token[token] for token in target_tokens)
                if decoded != target_text:
                    raise RuntimeError(
                        f"目标 token 反解与规范化文本不一致：{sample_id}"
                    )
                row["raw_target_text"] = raw_target_text
                row["target_text"] = target_text
                row["target_tokens"] = target_tokens
            else:
                token_string = values[3]
                row["target_tokens"] = [
                    int(token) for token in token_string.split()
                ]
                if id_to_token is not None and any(
                    token not in id_to_token for token in row["target_tokens"]
                ):
                    raise ValueError(
                        f"第 {index + 1} 行包含目标词表之外的 token ID"
                    )
            row["domain"] = domain_from_path(relative_path, pattern)
            row["feedback"] = False
            rows.append(row)

    if revisit_domains:
        shuffled_rows = ordered_rows(
            rows,
            "domain-block",
            args.shuffle_within_domain,
            False,
            rng,
        )
        grouped = defaultdict(list)
        for row in shuffled_rows:
            grouped[row["domain"]].append(row)
        first, second, third = revisit_domains
        missing_domains = [
            domain for domain in revisit_domains if domain not in grouped
        ]
        if missing_domains:
            raise ValueError(f"回访域不存在：{', '.join(missing_domains)}")
        selected_rows = [
            row
            for domain in revisit_domains
            for row in grouped[domain]
        ]
        uid_counts = Counter(row["uid"] for row in selected_rows)
        duplicate_uids = sorted(uid for uid, count in uid_counts.items() if count > 1)
        if duplicate_uids:
            raise ValueError(f"回访流 UID 重复：{duplicate_uids[0]}")
        first_rows = grouped[first]
        if len(first_rows) < 2:
            raise ValueError("首个回访域至少需要 2 条样本")
        split_index = (len(first_rows) + 1) // 2
        segment_lengths = OrderedDict(
            [
                ("A1", split_index),
                ("B", len(grouped[second])),
                ("C", len(grouped[third])),
                ("A2", len(first_rows) - split_index),
            ]
        )
        rows = (
            first_rows[:split_index]
            + grouped[second]
            + grouped[third]
            + first_rows[split_index:]
        )
        materialized_rows = []
        for row in rows:
            materialized = OrderedDict(
                [("uid", row["uid"]), ("video", row["video"])]
            )
            if use_metadata:
                sample_id = row["_sample_id"]
                if sample_id not in metadata:
                    raise ValueError(f"元数据缺少样本：{sample_id}")
                raw_target_text = metadata[sample_id]
                raw_characters += len(raw_target_text)
                target_text, target_tokens = encode_text(
                    raw_target_text,
                    token_to_id,
                    args.oov_policy,
                    dropped_counter,
                )
                if not target_tokens:
                    raise ValueError(f"样本 {sample_id} 的标签规范化后为空")
                target_characters += len(target_text)
                decoded = "".join(id_to_token[token] for token in target_tokens)
                if decoded != target_text:
                    raise RuntimeError(
                        f"目标 token 反解与规范化文本不一致：{sample_id}"
                    )
                materialized["raw_target_text"] = raw_target_text
                materialized["target_text"] = target_text
                materialized["target_tokens"] = target_tokens
            else:
                materialized["target_tokens"] = [
                    int(token) for token in row["_token_string"].split()
                ]
                if any(
                    token not in id_to_token
                    for token in materialized["target_tokens"]
                ):
                    raise ValueError(
                        f"第 {row['_source_line_number']} 行包含目标词表之外的 token ID"
                    )
            materialized["domain"] = row["domain"]
            materialized["feedback"] = False
            materialized_rows.append(materialized)
        rows = materialized_rows
    else:
        rows = ordered_rows(
            rows,
            args.order,
            args.shuffle_within_domain,
            args.shuffle_domains,
            rng,
        )
    if args.feedback_every > 0:
        for index, row in enumerate(rows, start=1):
            row["feedback"] = index % args.feedback_every == 0

    output = Path(args.output)
    sidecar_path = output.with_name(output.name + ".meta.json")
    sidecar = None
    if use_metadata:
        sidecar = build_metadata_sidecar(
            args, len(rows), raw_characters, target_characters, dropped_counter
        )
    elif args.target_vocab:
        sidecar = build_passthrough_sidecar(args, len(rows))
    if sidecar is not None and revisit_domains:
        sidecar["revisit_protocol"] = OrderedDict(
            [
                ("domain_sequence", [first, second, third, first]),
                ("segment_lengths", segment_lengths),
            ]
        )
    try:
        _write_lines_atomic(
            output, (json.dumps(row, ensure_ascii=False) for row in rows)
        )
        if sidecar is not None:
            _write_json_atomic(sidecar_path, sidecar)
        else:
            sidecar_path.unlink(missing_ok=True)
    except Exception:
        output.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise
    print(f"已写入 {len(rows)} 条流式样本：{output}")
    return rows


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
