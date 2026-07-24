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


EXPECTED_COMMIT = "b9d6d499dfe88703cf45a661bf4463880cf0a632"
SPEAKERS = ("188", "011", "036")
EXPECTED_COUNTS = {"188": 213, "011": 206, "036": 206}
EXPECTED_SEGMENTS = {"A1": 107, "B": 206, "C": 206, "A2": 106}
EXPECTED_SAMPLES = 625
EXCLUDED_SPEAKERS = {
    "001",
    "013",
    "015",
    "045",
    "046",
    "047",
    "071",
    "093",
    "098",
    "120",
    "126",
    "128",
    "133",
    "176",
    "183",
    "202",
}
SPEAKER_PATTERN = re.compile(r"(?:^|/)(?P<speaker>[0-9]{3})_")


def parse_args():
    parser = argparse.ArgumentParser(description="构建并审计冻结的 RSP-VSR dev7 流")
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--text-metadata-csv", required=True, type=Path)
    parser.add_argument("--target-vocab", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--code-commit", default=EXPECTED_COMMIT)
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
    all_counts = Counter()
    selected = []
    with train_csv.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"训练 CSV 第 {line_number} 行少于 4 列")
            speaker = speaker_from_path(row[1])
            all_counts[speaker] += 1
            if speaker in SPEAKERS:
                selected.append(row)

    ranked_eligible = sorted(
        (
            (speaker, count)
            for speaker, count in all_counts.items()
            if speaker not in EXCLUDED_SPEAKERS
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if tuple(speaker for speaker, _ in ranked_eligible[:3]) != SPEAKERS:
        raise ValueError(f"dev7 确定性选择漂移：{ranked_eligible[:3]}")
    selected_counts = Counter(speaker_from_path(row[1]) for row in selected)
    if dict(selected_counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"dev7 speaker 数量漂移：expected={EXPECTED_COUNTS}, "
            f"actual={dict(selected_counts)}"
        )
    return selected, ranked_eligible[:3]


def source_uids(rows):
    return {
        f"{row[0]}:{Path(row[1]).with_suffix('').as_posix()}"
        for row in rows
    }


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_manifest(args, source_rows, ranked_selection):
    manifest_rows = read_jsonl(args.manifest_output)
    if len(manifest_rows) != EXPECTED_SAMPLES:
        raise ValueError(
            f"dev7 manifest 应为 {EXPECTED_SAMPLES} 行，实际 {len(manifest_rows)}"
        )
    uids = [row["uid"] for row in manifest_rows]
    if len(set(uids)) != EXPECTED_SAMPLES or set(uids) != source_uids(source_rows):
        raise ValueError("dev7 manifest UID 不唯一或与 source CSV 不一致")
    if any(row.get("feedback") is not False for row in manifest_rows):
        raise ValueError("dev7 manifest 原始 feedback 必须全部为 false")

    expected_domains = (
        [SPEAKERS[0]] * EXPECTED_SEGMENTS["A1"]
        + [SPEAKERS[1]] * EXPECTED_SEGMENTS["B"]
        + [SPEAKERS[2]] * EXPECTED_SEGMENTS["C"]
        + [SPEAKERS[0]] * EXPECTED_SEGMENTS["A2"]
    )
    if [row["domain"] for row in manifest_rows] != expected_domains:
        raise ValueError("dev7 manifest 不符合冻结的 A-B-C-A 顺序")
    for row in manifest_rows:
        speaker = speaker_from_path(row["video"])
        if row["domain"] != speaker or speaker in EXCLUDED_SPEAKERS:
            raise ValueError(f"dev7 domain/path/exclusion 审计失败：{row['uid']}")

    sidecar_path = Path(f"{args.manifest_output}.meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    revisit = sidecar.get("revisit_protocol") or {}
    expected_domain_sequence = [SPEAKERS[0], SPEAKERS[1], SPEAKERS[2], SPEAKERS[0]]
    if sidecar.get("samples") != EXPECTED_SAMPLES:
        raise ValueError("dev7 sidecar samples 不等于 625")
    if revisit.get("domain_sequence") != expected_domain_sequence:
        raise ValueError("dev7 sidecar domain_sequence 漂移")
    if revisit.get("segment_lengths") != EXPECTED_SEGMENTS:
        raise ValueError("dev7 sidecar segment_lengths 漂移")
    if sidecar.get("source_csv_sha256") != sha256(args.source_output):
        raise ValueError("dev7 sidecar source CSV SHA-256 不一致")
    if sidecar.get("text_metadata_csv_sha256") != sha256(args.text_metadata_csv):
        raise ValueError("dev7 sidecar text metadata SHA-256 不一致")
    if sidecar.get("target_vocab_sha256") != sha256(args.target_vocab):
        raise ValueError("dev7 sidecar target vocab SHA-256 不一致")

    return {
        "schema_version": 1,
        "status": "AUDIT_OK",
        "code_commit": args.code_commit,
        "seed": args.seed,
        "selection_rule": "largest eligible train counts, speaker id tie-break",
        "ranked_selection": [list(item) for item in ranked_selection],
        "excluded_speakers": sorted(EXCLUDED_SPEAKERS),
        "speakers": list(SPEAKERS),
        "speaker_counts": EXPECTED_COUNTS,
        "samples": EXPECTED_SAMPLES,
        "feedback_all_false": True,
        "uid_unique": True,
        "domain_sequence": expected_domain_sequence,
        "segment_lengths": EXPECTED_SEGMENTS,
        "planned_periodic_feedback_queries": 62,
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
    if args.code_commit != EXPECTED_COMMIT:
        raise ValueError("dev7 code commit 与0是冻结提交")
    actual_commit = subprocess.run(
        ["git", "-C", str(args.code_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != EXPECTED_COMMIT:
        raise ValueError(f"dev7 code root commit 漂移：{actual_commit}")
    prepare_script = args.code_root / "scripts" / "prepare_stream_manifest.py"
    if not prepare_script.is_file():
        raise FileNotFoundError(f"找不到现有 manifest 生成器：{prepare_script}")
    source_rows, ranked_selection = select_source_rows(args.train_csv)
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
    audit = audit_manifest(args, source_rows, ranked_selection)
    write_json_atomic(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
