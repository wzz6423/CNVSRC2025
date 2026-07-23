#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


SPEAKERS = ("128", "047", "202")
EXPECTED_COUNTS = {"128": 239, "047": 231, "202": 230}
EXPECTED_SEGMENTS = {"A1": 120, "B": 231, "C": 230, "A2": 119}
FORBIDDEN_SPEAKERS = {"013"}
KNOWN_OTHER_SPLITS = {
    "dev2": {"015", "098", "133"},
    "dev3": {"120", "176", "183"},
    "holdout2_frozen": {"046", "001", "093"},
}
SPEAKER_PATTERN = re.compile(r"(?:^|/)(?P<speaker>[0-9]{3})_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建并严格审计冻结的 RSP-VSR dev4 流清单"
    )
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--text-metadata-csv", required=True, type=Path)
    parser.add_argument("--target-vocab", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speaker_from_path(relative_path):
    match = SPEAKER_PATTERN.search(relative_path)
    if match is None:
        raise ValueError(f"无法从路径提取 speaker：{relative_path}")
    return match.group("speaker")


def write_csv_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_source_rows(train_csv):
    selected = []
    counts = Counter()
    with train_csv.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"训练 CSV 第 {line_number} 行少于 4 列")
            speaker = speaker_from_path(row[1])
            if speaker in SPEAKERS:
                selected.append(row)
                counts[speaker] += 1
    if set(SPEAKERS) & FORBIDDEN_SPEAKERS:
        raise RuntimeError("冻结 speaker 集包含禁止候选 013")
    for split_name, split_speakers in KNOWN_OTHER_SPLITS.items():
        overlap = set(SPEAKERS) & split_speakers
        if overlap:
            raise RuntimeError(f"dev4 与 {split_name} speaker 重叠：{sorted(overlap)}")
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"dev4 speaker 数量漂移：expected={EXPECTED_COUNTS}, actual={dict(counts)}"
        )
    if any(speaker_from_path(row[1]) in FORBIDDEN_SPEAKERS for row in selected):
        raise RuntimeError("dev4 source 意外包含禁止候选 013")
    return selected


def source_uids(rows):
    return {
        f"{row[0]}:{Path(row[1]).with_suffix('').as_posix()}"
        for row in rows
    }


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_manifest(args, source_rows):
    manifest_rows = read_jsonl(args.manifest_output)
    if len(manifest_rows) != 700:
        raise ValueError(f"dev4 manifest 应为 700 行，实际 {len(manifest_rows)}")
    uids = [row["uid"] for row in manifest_rows]
    if len(set(uids)) != 700:
        raise ValueError("dev4 manifest UID 不唯一")
    if set(uids) != source_uids(source_rows):
        raise ValueError("dev4 manifest UID 集与 source CSV 不一致")
    if any(row.get("feedback") is not False for row in manifest_rows):
        raise ValueError("dev4 manifest 原始 feedback 必须全部为 false")

    expected_domains = (
        ["128"] * EXPECTED_SEGMENTS["A1"]
        + ["047"] * EXPECTED_SEGMENTS["B"]
        + ["202"] * EXPECTED_SEGMENTS["C"]
        + ["128"] * EXPECTED_SEGMENTS["A2"]
    )
    actual_domains = [row["domain"] for row in manifest_rows]
    if actual_domains != expected_domains:
        raise ValueError("dev4 manifest 不符合冻结的 A-B-C-A 顺序")
    for row in manifest_rows:
        path_speaker = speaker_from_path(row["video"])
        if row["domain"] != path_speaker:
            raise ValueError(f"domain/path speaker 不一致：{row['uid']}")
        if path_speaker in FORBIDDEN_SPEAKERS:
            raise RuntimeError("dev4 manifest 意外包含禁止候选 013")

    sidecar_path = Path(f"{args.manifest_output}.meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    revisit = sidecar.get("revisit_protocol") or {}
    if sidecar.get("samples") != 700:
        raise ValueError("dev4 sidecar samples 不等于 700")
    if revisit.get("domain_sequence") != ["128", "047", "202", "128"]:
        raise ValueError("dev4 sidecar domain_sequence 漂移")
    if revisit.get("segment_lengths") != EXPECTED_SEGMENTS:
        raise ValueError("dev4 sidecar segment_lengths 漂移")
    if sidecar.get("source_csv_sha256") != sha256(args.source_output):
        raise ValueError("dev4 sidecar source CSV SHA-256 不一致")
    if sidecar.get("text_metadata_csv_sha256") != sha256(
        args.text_metadata_csv
    ):
        raise ValueError("dev4 sidecar text metadata SHA-256 不一致")
    if sidecar.get("target_vocab_sha256") != sha256(args.target_vocab):
        raise ValueError("dev4 sidecar target vocab SHA-256 不一致")

    return {
        "schema_version": 1,
        "status": "AUDIT_OK",
        "code_commit": args.code_commit,
        "seed": args.seed,
        "speakers": list(SPEAKERS),
        "speaker_counts": EXPECTED_COUNTS,
        "forbidden_speakers": sorted(FORBIDDEN_SPEAKERS),
        "samples": 700,
        "feedback_all_false": True,
        "uid_unique": True,
        "domain_sequence": ["128", "047", "202", "128"],
        "segment_lengths": EXPECTED_SEGMENTS,
        "artifacts": {
            "source_csv": {
                "path": str(args.source_output),
                "sha256": sha256(args.source_output),
            },
            "manifest": {
                "path": str(args.manifest_output),
                "sha256": sha256(args.manifest_output),
            },
            "manifest_sidecar": {
                "path": str(sidecar_path),
                "sha256": sha256(sidecar_path),
            },
            "train_csv_sha256": sha256(args.train_csv),
            "text_metadata_csv_sha256": sha256(args.text_metadata_csv),
            "target_vocab_sha256": sha256(args.target_vocab),
        },
    }


def main():
    args = parse_args()
    prepare_script = args.code_root / "scripts" / "prepare_stream_manifest.py"
    if not prepare_script.is_file():
        raise FileNotFoundError(f"找不到现有 manifest 生成器：{prepare_script}")
    source_rows = select_source_rows(args.train_csv)
    write_csv_atomic(args.source_output, source_rows)
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--csv",
            str(args.source_output),
            "--output",
            str(args.manifest_output),
            "--domain-regex",
            r"(?:^|/)(?P<domain>[0-9]{3})_",
            "--order",
            "domain-block",
            "--shuffle-within-domain",
            "--revisit-domains",
            ",".join(SPEAKERS),
            "--seed",
            str(args.seed),
            "--text-metadata-csv",
            str(args.text_metadata_csv),
            "--target-vocab",
            str(args.target_vocab),
            "--oov-policy",
            "drop",
        ],
        check=True,
        cwd=args.code_root,
    )
    audit = audit_manifest(args, source_rows)
    write_json_atomic(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
