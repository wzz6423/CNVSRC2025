#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_COMMIT = "d123cf3cca7b900be0c8baa6538fd6237081be14"
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
INPUT_SPECS = {
    "dev6": {
        "samples": 356,
        "sha256": "2570f297331281b6dec5389546bad75de36d35de51c86bdc65d32903c1035922",
    },
    "dev7": {
        "samples": 625,
        "sha256": "a7eff26328a8ba136e575efa5c775a17e941743297a74014adb5f4303c6f3c69",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="审计 Dev8 约束式小语言模型重排")
    parser.add_argument("--name", choices=sorted(INPUT_SPECS), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_evaluator(code_root):
    scripts = code_root / "VSR" / "scripts"
    sys.path.insert(0, str(scripts))
    import evaluate_llm_nbest_selector as evaluator

    return evaluator


def check_metadata(args, evaluator, spec):
    metadata_path = args.output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["input_sha256"] == spec["sha256"]
    assert metadata["expected_samples"] == spec["samples"]
    assert metadata["top_k"] == 10
    assert metadata["model"]["id"] == EXPECTED_MODEL_ID
    assert metadata["model"]["revision"] == EXPECTED_MODEL_REVISION
    assert metadata["prompt"]["version"] == evaluator.PROMPT_VERSION
    assert metadata["code_commit"] == EXPECTED_COMMIT
    assert metadata["generation"] == {
        "batch_size": 8,
        "do_sample": False,
        "max_new_tokens": 8,
        "seed": 42,
    }
    assert metadata["bootstrap_samples"] == 10000
    return metadata


def check_records(args, evaluator, source_rows, records):
    assert len(records) <= len(source_rows)
    tokenizer = None
    if records:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True, trust_remote_code=False
        )
    for index, record in enumerate(records):
        source = source_rows[index]
        assert record["index"] == index
        assert record["uid"] == source["uid"]
        assert record["domain"] == source.get("domain")
        assert 1 <= int(record["selected_rank"]) <= 10
        assert bool(record["selection_changed"]) == (
            record["selected_rank"] != 1
        )
        parsed_rank, parser = evaluator.parse_rank(record["response"], 10)
        assert record["parser"] == parser
        assert bool(record["parsed"]) == (parsed_rank is not None)
        expected_rank = parsed_rank if parsed_rank is not None else 1
        assert record["selected_rank"] == expected_rank

        nbest = source["decoder_nbest"]
        assert record["baseline_transcript"] == nbest[0]["transcript"]
        assert record["selected_transcript"] == nbest[expected_rank - 1]["transcript"]
        assert record["candidate_transcripts_sha256"] == evaluator.payload_sha256(
            [candidate["transcript"] for candidate in nbest]
        )
        edits = [
            evaluator.edit_distance(candidate["transcript"], source["target"])
            for candidate in nbest
        ]
        assert record["candidate_edits"] == edits
        assert record["baseline_edits"] == edits[0]
        assert record["selected_edits"] == edits[expected_rank - 1]
        assert record["oracle_edits"] == min(edits)
        assert record["oracle_ranks"] == [
            rank
            for rank, value in enumerate(edits, start=1)
            if value == min(edits)
        ]
        assert record["characters"] == len(source["target"])

        messages = evaluator.build_messages(
            [candidate["transcript"] for candidate in nbest]
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert record["prompt_sha256"] == hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()


def check_final(args, evaluator, records, metadata, spec):
    assert len(records) == spec["samples"]
    assert metadata is not None and metadata["status"] == "COMPLETE"
    summary_path = args.output_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recomputed = evaluator.summarize(records, bootstrap_samples=10000, seed=42)
    for key, value in recomputed.items():
        assert summary[key] == value, key
    results_path = args.output_dir / "selections.jsonl"
    assert summary["results_sha256"] == sha256(results_path)
    assert metadata["results_sha256"] == summary["results_sha256"]
    assert metadata["summary_sha256"] == sha256(summary_path)


def main():
    args = parse_args()
    spec = INPUT_SPECS[args.name]
    assert sha256(args.input) == spec["sha256"]
    evaluator = load_evaluator(args.code_root)
    source_rows = read_jsonl(args.input)
    evaluator.validate_source_rows(source_rows, spec["samples"], top_k=10)
    metadata = check_metadata(args, evaluator, spec)
    records = read_jsonl(args.output_dir / "selections.jsonl")
    check_records(args, evaluator, source_rows, records)
    if args.final:
        check_final(args, evaluator, records, metadata, spec)
    print(
        json.dumps(
            {
                "status": "AUDIT_OK",
                "name": args.name,
                "rows": len(records),
                "final": args.final,
                "metadata": metadata is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
