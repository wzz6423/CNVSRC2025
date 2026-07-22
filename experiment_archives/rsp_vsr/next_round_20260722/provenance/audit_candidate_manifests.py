import csv
import hashlib
import json
from pathlib import Path

from continual_adapt import _stream_state


MANIFEST_DIR = Path("/hy-tmp/datasets/manifests")
TRAIN_CSV = Path(
    "/hy-tmp/cn_dataset/chinese_lips/chinese_lips/labels/train.csv"
)
TEXT_METADATA = Path(
    "/hy-tmp/cn_dataset/chinese_lips/chinese_lips/meta_train.csv"
)
TARGET_VOCAB = Path(
    "/hy-tmp/rsp-vsr-next-353c47d/VSR/datamodule/char_units.txt"
)
BASE_CHECKPOINT = Path(
    "/hy-tmp/rsp-vsr/VSR/pretrained_models/model_avg_cncvs_2_3_cnvsrc.pth"
)

DEV2_SOURCE = (
    MANIFEST_DIR / "chinese_lips_trainpool_dev2_015_098_133_seed42_source.csv"
)
DEV2 = (
    MANIFEST_DIR / "chinese_lips_trainpool_dev2_revisit_015_098_133_seed42.jsonl"
)
HOLDOUT2_SOURCE = (
    MANIFEST_DIR / "chinese_lips_trainpool_holdout2_046_001_093_seed42_source.csv"
)
HOLDOUT2 = (
    MANIFEST_DIR
    / "chinese_lips_trainpool_holdout2_revisit_046_001_093_seed42.jsonl"
)
VALIDATION = MANIFEST_DIR / "chinese_lips_val_model_vocab_seed42.jsonl"
TEST = MANIFEST_DIR / "chinese_lips_test_model_vocab_seed42.jsonl"

EXPECTED_HASHES = {
    TRAIN_CSV: "a1d7a5ba8da02ab3e0284365ab9f1bde00d76657604b5453b41db6a7891c22ab",
    TEXT_METADATA: "bc0150e082657698019a976b3a280c23f0640df38603f658870f548fe4268bac",
    TARGET_VOCAB: "635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0",
    BASE_CHECKPOINT: "577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c",
    DEV2_SOURCE: "70483a27e8a1e4f23c4aeccbea9df52c7c9e8e8045233e4a45908bf7141cbd54",
    HOLDOUT2_SOURCE: "16d0d172c240b29eb1b9a73b26676a91c29978cbd0b0ebff7b7dd59d9e4d378c",
    DEV2: "7defae9074e78d3893edc41bde1654b709cc59efaddb183531fecce20778e590",
    Path(f"{DEV2}.meta.json"): "6053164ebfba99b8c47dda620287f605cc452467ca83ae9e1158b3cd7c132c77",
    HOLDOUT2: "29b4005205e18c8a2b1fa643d0fc9fb35bbf6266692e12a4c73ee0ad42a07651",
    Path(f"{HOLDOUT2}.meta.json"): "0a15e1f2bd0ec6284b9b1df1b924a748a46356be1682c9975c126577f93edc17",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_uids(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            f"{row[0]}:{Path(row[1]).with_suffix('').as_posix()}"
            for row in csv.reader(handle)
            if row
        }


def audit_manifest(path, source, segment_lengths, domains):
    rows = read_jsonl(path)
    expected_samples = sum(segment_lengths)
    uids = {row["uid"] for row in rows}
    assert len(rows) == expected_samples
    assert len(uids) == expected_samples
    assert uids == csv_uids(source)
    assert all(row["feedback"] is False for row in rows)

    expected_order = (
        [domains[0]] * segment_lengths[0]
        + [domains[1]] * segment_lengths[1]
        + [domains[2]] * segment_lengths[2]
        + [domains[0]] * segment_lengths[3]
    )
    assert [row["domain"] for row in rows] == expected_order

    sidecar = json.loads(
        Path(f"{path}.meta.json").read_text(encoding="utf-8")
    )
    revisit = sidecar["revisit_protocol"]
    assert sidecar["samples"] == expected_samples
    assert revisit["domain_sequence"] == [
        domains[0],
        domains[1],
        domains[2],
        domains[0],
    ]
    assert list(revisit["segment_lengths"].values()) == segment_lengths
    runtime_state = _stream_state(path, require_metadata=True)
    return uids, runtime_state


def main():
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        assert actual == expected, f"SHA mismatch for {path}: {actual}"

    dev2_uids, dev2_state = audit_manifest(
        DEV2, DEV2_SOURCE, [154, 253, 243, 154], ["015", "098", "133"]
    )
    holdout2_uids, holdout2_state = audit_manifest(
        HOLDOUT2,
        HOLDOUT2_SOURCE,
        [136, 242, 264, 136],
        ["046", "001", "093"],
    )
    validation_uids = {row["uid"] for row in read_jsonl(VALIDATION)}
    test_uids = {row["uid"] for row in read_jsonl(TEST)}

    assert not dev2_uids & holdout2_uids
    assert not dev2_uids & validation_uids
    assert not dev2_uids & test_uids
    assert not holdout2_uids & validation_uids
    assert not holdout2_uids & test_uids

    print(
        json.dumps(
            {
                "status": "AUDIT_OK",
                "dev2_samples": len(dev2_uids),
                "holdout2_samples": len(holdout2_uids),
                "feedback_all_false": True,
                "cross_split_uid_overlap": 0,
                "dev2_stream_state": dev2_state,
                "holdout2_stream_state": holdout2_state,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
