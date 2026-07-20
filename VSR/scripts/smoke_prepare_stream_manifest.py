import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_stream_manifest import (
    _file_sha256,
    encode_text,
    load_target_vocab,
    load_text_metadata,
    run,
)


def make_args(csv_path, output, **overrides):
    values = dict(
        csv=str(csv_path),
        output=str(output),
        domain_regex=None,
        order="original",
        shuffle_within_domain=False,
        shuffle_domains=False,
        feedback_every=0,
        seed=42,
        text_metadata_csv=None,
        metadata_id_column="ID",
        metadata_text_column="TEXT",
        target_vocab=None,
        oov_policy="error",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def assert_raises(exc, needle, function, *function_args, **function_kwargs):
    try:
        function(*function_args, **function_kwargs)
    except exc as error:
        assert needle in str(error), (needle, str(error))
    else:
        raise AssertionError(f"预期抛出 {exc.__name__}（{needle}）")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_fixtures(directory):
    source_csv = directory / "source.csv"
    # 第四列是源词表 ID，元数据模式必须忽略它。
    source_csv.write_text(
        "train,speaker/aaa.mp4,30,999 999 999 999\n"
        "train,speaker/bbb.mp4,42,777 777 777\n",
        encoding="utf-8",
    )
    metadata_csv = directory / "metadata.csv"
    # 带 BOM 的 utf-8-sig 表头，验证 DictReader 正确剥离 BOM。
    metadata_csv.write_text(
        "ID,TEXT\naaa,你好世界\nbbb,你好猫猫狗\n",
        encoding="utf-8-sig",
    )
    vocab = directory / "target_vocab.txt"
    vocab.write_text("你 1\n好 2\n世 3\n界 4\n", encoding="utf-8")
    return source_csv, metadata_csv, vocab


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        source_csv, metadata_csv, vocab = write_fixtures(directory)

        # 单元级：目标词表与元数据加载。
        token_to_id, id_to_token = load_target_vocab(vocab)
        assert token_to_id == {"你": 1, "好": 2, "世": 3, "界": 4}
        assert id_to_token[4] == "界"
        metadata = load_text_metadata(metadata_csv, "ID", "TEXT")
        assert set(metadata) == {"aaa", "bbb"}, "utf-8-sig BOM 未被正确剥离"

        # 单元级：drop 策略统计与 error 策略拒绝。
        dropped = Counter()
        text, tokens = encode_text("你好猫猫狗", token_to_id, "drop", dropped)
        assert text == "你好" and tokens == [1, 2]
        assert dropped == Counter({"猫": 2, "狗": 1})
        assert_raises(
            ValueError, "不在目标词表", encode_text, "你猫", token_to_id, "error", Counter()
        )

        # 集成：drop 模式端到端映射与 sidecar。
        drop_output = directory / "drop.jsonl"
        rows = run(
            make_args(
                source_csv,
                drop_output,
                text_metadata_csv=str(metadata_csv),
                target_vocab=str(vocab),
                oov_policy="drop",
            )
        )
        disk_rows = read_jsonl(drop_output)
        assert disk_rows == rows, "落盘 JSONL 与返回结果不一致"
        by_uid = {row["uid"]: row for row in disk_rows}
        aaa = by_uid["train:speaker/aaa"]
        assert aaa["raw_target_text"] == "你好世界"
        assert aaa["target_text"] == "你好世界"
        assert aaa["target_tokens"] == [1, 2, 3, 4]
        bbb = by_uid["train:speaker/bbb"]
        assert bbb["raw_target_text"] == "你好猫猫狗"
        assert bbb["target_text"] == "你好"  # 忽略源第四列，OOV 已丢弃
        assert bbb["target_tokens"] == [1, 2]
        for row in disk_rows:  # target_tokens 反解严格等于 target_text
            decoded = "".join(id_to_token[token] for token in row["target_tokens"])
            assert decoded == row["target_text"]

        sidecar_path = drop_output.with_name(drop_output.name + ".meta.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["schema_version"] == 1
        assert sidecar["label_mode"] == "metadata_reencoded"
        assert sidecar["samples"] == 2 == len(disk_rows)
        assert sidecar["oov_policy"] == "drop"
        assert sidecar["source_csv_sha256"] == _file_sha256(source_csv)
        assert sidecar["text_metadata_csv_sha256"] == _file_sha256(metadata_csv)
        assert sidecar["target_vocab_sha256"] == _file_sha256(vocab)
        assert sidecar["raw_characters"] == 9
        assert sidecar["target_characters"] == 6
        assert sidecar["dropped_characters"] == 3
        assert abs(sidecar["dropped_rate"] - 3 / 9) < 1e-9
        assert sidecar["distinct_dropped_characters"] == 2
        # 按频次降序：猫(2) 在 狗(1) 之前（JSON 反序列化为 dict，比较键序）。
        assert list(sidecar["dropped_character_counts"].items()) == [
            ("猫", 2),
            ("狗", 1),
        ]

        # 集成：error 策略遇 OOV 失败且不留半文件。
        error_output = directory / "error.jsonl"
        assert_raises(
            ValueError,
            "不在目标词表",
            run,
            make_args(
                source_csv,
                error_output,
                text_metadata_csv=str(metadata_csv),
                target_vocab=str(vocab),
                oov_policy="error",
            ),
        )
        assert not error_output.exists(), "失败时不应遗留输出文件"
        assert not error_output.with_name(error_output.name + ".meta.json").exists()

        # 集成：缺失样本元数据必须报错。
        partial_metadata = directory / "partial.csv"
        partial_metadata.write_text("ID,TEXT\naaa,你好世界\n", encoding="utf-8-sig")
        missing_output = directory / "missing.jsonl"
        assert_raises(
            ValueError,
            "元数据缺少样本",
            run,
            make_args(
                source_csv,
                missing_output,
                text_metadata_csv=str(partial_metadata),
                target_vocab=str(vocab),
            ),
        )
        assert not missing_output.exists()

        # 集成：旧模式（无元数据）四列 token 直通完全兼容。
        legacy_output = directory / "legacy.jsonl"
        legacy_rows = run(make_args(source_csv, legacy_output))
        assert legacy_rows[0] == {
            "uid": "train:speaker/aaa",
            "video": str(Path("train") / "speaker/aaa.mp4"),
            "target_tokens": [999, 999, 999, 999],
            "domain": "speaker",
            "feedback": False,
        }
        assert "raw_target_text" not in legacy_rows[0]
        assert not legacy_output.with_name(legacy_output.name + ".meta.json").exists()

        verified_legacy_source = directory / "verified_legacy.csv"
        verified_legacy_source.write_text(
            "train,speaker/aaa.mp4,30,1 2 3 4\n", encoding="utf-8"
        )
        verified_legacy_output = directory / "verified_legacy.jsonl"
        verified_legacy_rows = run(
            make_args(
                verified_legacy_source,
                verified_legacy_output,
                target_vocab=str(vocab),
            )
        )
        assert verified_legacy_rows[0]["target_tokens"] == [1, 2, 3, 4]
        verified_legacy_sidecar = json.loads(
            verified_legacy_output.with_name(
                verified_legacy_output.name + ".meta.json"
            ).read_text(encoding="utf-8")
        )
        assert verified_legacy_sidecar["label_mode"] == "token_passthrough"
        assert verified_legacy_sidecar["target_vocab_sha256"] == _file_sha256(vocab)

        reused_output = directory / "reused.jsonl"
        run(
            make_args(
                source_csv,
                reused_output,
                text_metadata_csv=str(metadata_csv),
                target_vocab=str(vocab),
                oov_policy="drop",
            )
        )
        reused_sidecar = reused_output.with_name(reused_output.name + ".meta.json")
        assert reused_sidecar.is_file()
        run(make_args(source_csv, reused_output))
        assert not reused_sidecar.exists(), "旧模式不应保留之前的词表 sidecar"

        # 校验：元数据模式强制要求 target vocab。
        assert_raises(
            ValueError,
            "必须提供 --target-vocab",
            run,
            make_args(
                source_csv,
                directory / "novocab.jsonl",
                text_metadata_csv=str(metadata_csv),
            ),
        )

        # 校验：重复 ID / 缺列 / 词表非法。
        dup_metadata = directory / "dup.csv"
        dup_metadata.write_text(
            "ID,TEXT\naaa,你好\naaa,世界\n", encoding="utf-8-sig"
        )
        assert_raises(
            ValueError, "元数据 ID 重复", load_text_metadata, dup_metadata, "ID", "TEXT"
        )
        assert_raises(
            ValueError, "缺少列", load_text_metadata, metadata_csv, "ID", "MISSING"
        )
        dup_vocab = directory / "dup_vocab.txt"
        dup_vocab.write_text("你 1\n好 1\n", encoding="utf-8")
        assert_raises(ValueError, "ID 重复", load_target_vocab, dup_vocab)
        bad_id_vocab = directory / "bad_vocab.txt"
        bad_id_vocab.write_text("你 0\n", encoding="utf-8")
        assert_raises(ValueError, "正整数", load_target_vocab, bad_id_vocab)
        non_contiguous_vocab = directory / "non_contiguous_vocab.txt"
        non_contiguous_vocab.write_text("你 2\n", encoding="utf-8")
        assert_raises(
            ValueError, "连续", load_target_vocab, non_contiguous_vocab
        )

        all_oov_metadata = directory / "all_oov.csv"
        all_oov_metadata.write_text(
            "ID,TEXT\naaa,猫\nbbb,狗\n", encoding="utf-8-sig"
        )
        empty_output = directory / "empty_target.jsonl"
        assert_raises(
            ValueError,
            "规范化后为空",
            run,
            make_args(
                source_csv,
                empty_output,
                text_metadata_csv=str(all_oov_metadata),
                target_vocab=str(vocab),
                oov_policy="drop",
            ),
        )
        assert not empty_output.exists()

    print("RSP-VSR prepare_stream_manifest smoke 通过")


if __name__ == "__main__":
    main()
