import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.prepare_stream_manifest as manifest_script
from continual_adapt import _stream_state
from datamodule.transforms import DICT_PATH
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
        revisit_domains=None,
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


def write_revisit_fixtures(directory, name="revisit", a_count=4):
    source_csv = directory / f"{name}_source.csv"
    records = [
        ("validation", f"A/a{index}.mp4", f"a{index}", "你猫")
        for index in range(a_count)
    ]
    records.extend(
        [
            ("validation", "B/b0.mp4", "b0", "好狗"),
            ("validation", "B/b1.mp4", "b1", "好狗"),
            ("validation", "C/c0.mp4", "c0", "世界"),
            ("validation", "C/c1.mp4", "c1", "世界"),
            ("validation", "D/d0.mp4", "d0", "猫猫猫"),
        ]
    )
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for dataset, relative_path, _sample_id, _text in records:
            writer.writerow([dataset, relative_path, 1, 1])

    metadata_csv = directory / f"{name}_metadata.csv"
    with metadata_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "TEXT"])
        writer.writeheader()
        for _dataset, _relative_path, sample_id, text in records:
            writer.writerow({"ID": sample_id, "TEXT": text})

    vocab = directory / f"{name}_vocab.txt"
    vocab.write_text("你 1\n好 2\n世 3\n界 4\n", encoding="utf-8")
    return source_csv, metadata_csv, vocab


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        source_csv, metadata_csv, vocab = write_fixtures(directory)

        # RED/GREEN：回访模式只保留指定域，并按 A1-B-C-A2 排列。
        revisit_source, revisit_metadata, revisit_vocab = write_revisit_fixtures(
            directory
        )
        revisit_output = directory / "revisit_even.jsonl"
        revisit_rows = run(
            make_args(
                revisit_source,
                revisit_output,
                order="domain-block",
                shuffle_within_domain=True,
                feedback_every=2,
                seed=17,
                revisit_domains=("A", "B", "C"),
                text_metadata_csv=str(revisit_metadata),
                target_vocab=str(revisit_vocab),
                oov_policy="drop",
            )
        )
        assert [row["domain"] for row in revisit_rows] == [
            "A",
            "A",
            "B",
            "B",
            "C",
            "C",
            "A",
            "A",
        ]
        assert len({row["uid"] for row in revisit_rows}) == len(revisit_rows)
        assert [row["feedback"] for row in revisit_rows] == [
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
        ]
        revisit_sidecar = json.loads(
            Path(f"{revisit_output}.meta.json").read_text(encoding="utf-8")
        )
        assert revisit_sidecar["samples"] == 8
        assert revisit_sidecar["raw_characters"] == 16
        assert revisit_sidecar["target_characters"] == 10
        assert revisit_sidecar["dropped_characters"] == 6
        assert (
            revisit_sidecar["raw_characters"]
            == revisit_sidecar["target_characters"]
            + revisit_sidecar["dropped_characters"]
        )
        assert revisit_sidecar["dropped_character_counts"] == {"猫": 4, "狗": 2}
        assert revisit_sidecar["revisit_protocol"] == {
            "domain_sequence": ["A", "B", "C", "A"],
            "segment_lengths": {"A1": 2, "B": 2, "C": 2, "A2": 2},
        }

        # CLI 使用 CSV 解析，允许引号内逗号，但严格拒绝数量/空值/重复域。
        assert hasattr(manifest_script, "parse_revisit_domains")
        parse_revisit_domains = manifest_script.parse_revisit_domains
        assert parse_revisit_domains("A,B,C") == ("A", "B", "C")
        assert parse_revisit_domains('"A,part",B,C') == ("A,part", "B", "C")
        assert_raises(
            argparse.ArgumentTypeError,
            "恰好 3 个",
            parse_revisit_domains,
            "A,B",
        )
        assert_raises(
            argparse.ArgumentTypeError,
            "不能为空",
            parse_revisit_domains,
            "A,,C",
        )
        assert_raises(
            argparse.ArgumentTypeError,
            "互不相同",
            parse_revisit_domains,
            "A,A,C",
        )
        assert_raises(
            argparse.ArgumentTypeError,
            "CSV 格式无效",
            parse_revisit_domains,
            'A,"B,C',
        )

        # test 风格奇数 A：A1 按 ceil(n/2) 分配，且同 seed 完全可复现。
        odd_source, odd_metadata, odd_vocab = write_revisit_fixtures(
            directory, name="revisit_odd", a_count=5
        )
        odd_args = dict(
            order="domain-block",
            shuffle_within_domain=True,
            seed=123,
            revisit_domains=("A", "B", "C"),
            text_metadata_csv=str(odd_metadata),
            target_vocab=str(odd_vocab),
            oov_policy="drop",
        )
        odd_output = directory / "revisit_odd.jsonl"
        odd_rows = run(make_args(odd_source, odd_output, **odd_args))
        repeated_output = directory / "revisit_odd_repeated.jsonl"
        repeated_rows = run(make_args(odd_source, repeated_output, **odd_args))
        assert [row["uid"] for row in odd_rows] == [
            row["uid"] for row in repeated_rows
        ]
        assert [row["domain"] for row in odd_rows] == [
            "A",
            "A",
            "A",
            "B",
            "B",
            "C",
            "C",
            "A",
            "A",
        ]
        odd_sidecar = json.loads(
            Path(f"{odd_output}.meta.json").read_text(encoding="utf-8")
        )
        assert odd_sidecar["revisit_protocol"]["segment_lengths"] == {
            "A1": 3,
            "B": 2,
            "C": 2,
            "A2": 2,
        }

        # 回访子流必须先复用全域 shuffle 的 RNG 消耗，再选 A/B/C。
        shuffle_source = directory / "revisit_shuffle_source.csv"
        shuffle_records = []
        for domain, count in (("excluded", 4), ("A", 5), ("B", 3), ("C", 3)):
            for index in range(count):
                shuffle_records.append(
                    ["validation", f"{domain}/{domain}{index}.mp4", 1, 1]
                )
        with shuffle_source.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(shuffle_records)
        full_rows = run(
            make_args(
                shuffle_source,
                directory / "revisit_shuffle_full.jsonl",
                order="domain-block",
                shuffle_within_domain=True,
                seed=91,
                target_vocab=str(revisit_vocab),
            )
        )
        full_groups = {
            domain: [row for row in full_rows if row["domain"] == domain]
            for domain in ("A", "B", "C")
        }
        full_a_split = (len(full_groups["A"]) + 1) // 2
        expected_revisit_uids = [
            row["uid"]
            for row in (
                full_groups["A"][:full_a_split]
                + full_groups["B"]
                + full_groups["C"]
                + full_groups["A"][full_a_split:]
            )
        ]
        shuffled_revisit_rows = run(
            make_args(
                shuffle_source,
                directory / "revisit_shuffle_subset.jsonl",
                order="domain-block",
                shuffle_within_domain=True,
                seed=91,
                revisit_domains=("A", "B", "C"),
                target_vocab=str(revisit_vocab),
            )
        )
        assert [row["uid"] for row in shuffled_revisit_rows] == expected_revisit_uids

        invalid_base = dict(
            order="domain-block",
            revisit_domains=("A", "B", "C"),
            target_vocab=str(revisit_vocab),
        )
        assert_raises(
            ValueError,
            "回访域不存在",
            run,
            make_args(
                revisit_source,
                directory / "missing_domain.jsonl",
                **{**invalid_base, "revisit_domains": ("A", "B", "missing")},
            ),
        )
        short_first_domain_source = directory / "short_first_domain.csv"
        with short_first_domain_source.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            csv.writer(handle).writerows(
                [
                    ["validation", "A/a0.mp4", 1, 1],
                    ["validation", "B/b0.mp4", 1, 1],
                    ["validation", "C/c0.mp4", 1, 1],
                ]
            )
        assert_raises(
            ValueError,
            "首个回访域至少需要 2 条样本",
            run,
            make_args(
                short_first_domain_source,
                directory / "short_first_domain.jsonl",
                **invalid_base,
            ),
        )
        assert_raises(
            ValueError,
            "要求 --order domain-block",
            run,
            make_args(
                revisit_source,
                directory / "invalid_order.jsonl",
                **{**invalid_base, "order": "original"},
            ),
        )
        assert_raises(
            ValueError,
            "不能同时使用 --shuffle-domains",
            run,
            make_args(
                revisit_source,
                directory / "invalid_shuffle.jsonl",
                **{**invalid_base, "shuffle_domains": True},
            ),
        )
        assert_raises(
            ValueError,
            "必须提供 --target-vocab",
            run,
            make_args(
                revisit_source,
                directory / "revisit_without_sidecar.jsonl",
                order="domain-block",
                revisit_domains=("A", "B", "C"),
            ),
        )

        duplicate_source = directory / "duplicate_uid.csv"
        with duplicate_source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(
                [
                    ["validation", "A/a0.mp4", 1, 1],
                    ["validation", "A/a0.mp4", 1, 1],
                    ["validation", "B/b0.mp4", 1, 1],
                    ["validation", "C/c0.mp4", 1, 1],
                ]
            )
        assert_raises(
            ValueError,
            "UID 重复",
            run,
            make_args(
                duplicate_source,
                directory / "duplicate_uid.jsonl",
                order="domain-block",
                revisit_domains=("A", "B", "C"),
                target_vocab=str(revisit_vocab),
            ),
        )

        # 生成的 passthrough sidecar 必须能被真实运行时清单校验接受。
        runtime_output = directory / "revisit_runtime.jsonl"
        runtime_rows = run(
            make_args(
                revisit_source,
                runtime_output,
                order="domain-block",
                revisit_domains=("A", "B", "C"),
                target_vocab=str(DICT_PATH),
            )
        )
        runtime_state = _stream_state(runtime_output)
        assert len(runtime_rows) == 8
        assert set(runtime_state) == {
            "manifest_sha256",
            "manifest_metadata_sha256",
            "target_vocab_sha256",
        }

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
        expected_legacy_row = {
            "uid": "train:speaker/aaa",
            "video": str(Path("train") / "speaker/aaa.mp4"),
            "target_tokens": [999, 999, 999, 999],
            "domain": "speaker",
            "feedback": False,
        }
        assert legacy_rows[0] == expected_legacy_row
        assert legacy_output.read_text(encoding="utf-8").splitlines()[0] == (
            json.dumps(expected_legacy_row, ensure_ascii=False)
        ), "旧模式 JSON 键顺序不得变化，否则 manifest SHA 会漂移"
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
