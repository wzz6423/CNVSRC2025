#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path


PROMPT_VERSION = "dev8-constrained-choice-v1"
SYSTEM_PROMPT = (
    "你是中文句子判别器。下面的候选来自同一段无声唇语视频，"
    "可能只有少量字不同。结合语法、语义、常识和句子完整性，"
    "选出最可能正确的一条。不得改写、补充或合并候选，"
    "只输出一个阿拉伯数字作为候选序号。"
)
RANK_FIELD_PATTERN = re.compile(
    r'(?:"?rank"?|序号|候选)\s*[:：]?\s*(10|[1-9])', re.IGNORECASE
)
STANDALONE_RANK_PATTERN = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用固定小型语言模型从已有 VSR N-best 中选择候选"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalize_transcript(value):
    if not isinstance(value, str):
        raise ValueError("N-best transcript 必须是字符串")
    return " ".join(value.split())


def validate_source_rows(rows, expected_samples, top_k):
    if len(rows) != expected_samples:
        raise ValueError(
            f"输入应为 {expected_samples} 行，实际为 {len(rows)} 行"
        )
    seen_uids = set()
    for position, row in enumerate(rows):
        if row.get("index") != position:
            raise ValueError(f"输入 index 不连续：{position}")
        uid = row.get("uid")
        if not isinstance(uid, str) or not uid or uid in seen_uids:
            raise ValueError(f"输入 UID 无效或重复：{uid}")
        seen_uids.add(uid)
        if not isinstance(row.get("target"), str):
            raise ValueError(f"输入 target 无效：{uid}")
        nbest = row.get("decoder_nbest")
        if not isinstance(nbest, list) or len(nbest) != top_k:
            raise ValueError(f"{uid} 不是完整 top-{top_k}")
        if [candidate.get("rank") for candidate in nbest] != list(
            range(1, top_k + 1)
        ):
            raise ValueError(f"{uid} N-best rank 不连续")
        transcripts = [
            normalize_transcript(candidate.get("transcript"))
            for candidate in nbest
        ]
        if row.get("transcript") != transcripts[0]:
            raise ValueError(f"{uid} 原始输出不是 N-best rank-1")


def build_messages(candidates):
    candidate_lines = []
    for rank, candidate in enumerate(candidates, start=1):
        transcript = normalize_transcript(candidate)
        candidate_lines.append(f"候选 {rank}：{transcript}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(candidate_lines) + "\n请只输出候选序号。",
        },
    ]


def parse_rank(response, candidate_count):
    value = response.strip()
    if value.isdigit() and 1 <= int(value) <= candidate_count:
        return int(value), "exact_integer"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("rank"), int)
        and 1 <= parsed["rank"] <= candidate_count
    ):
        return parsed["rank"], "json_rank"
    field_matches = {
        int(match)
        for match in RANK_FIELD_PATTERN.findall(value)
        if 1 <= int(match) <= candidate_count
    }
    if len(field_matches) == 1:
        return field_matches.pop(), "rank_field"
    matches = {
        int(match)
        for match in STANDALONE_RANK_PATTERN.findall(value)
        if 1 <= int(match) <= candidate_count
    }
    if len(matches) == 1:
        return matches.pop(), "single_number"
    return None, "unparsed"


def model_artifacts(model_dir):
    records = []
    for path in sorted(Path(model_dir).rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        records.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not records:
        raise ValueError(f"模型目录为空：{model_dir}")
    return records


def percentile(sorted_values, probability):
    if not sorted_values:
        raise ValueError("不能对空列表计算分位数")
    index = int(round((len(sorted_values) - 1) * probability))
    return float(sorted_values[index])


def paired_bootstrap(records, samples, seed):
    if samples < 1:
        raise ValueError("bootstrap_samples 必须大于 0")
    rng = random.Random(seed)
    count = len(records)
    deltas = []
    for _ in range(samples):
        baseline_edits = 0
        selected_edits = 0
        characters = 0
        for _ in range(count):
            record = records[rng.randrange(count)]
            baseline_edits += int(record["baseline_edits"])
            selected_edits += int(record["selected_edits"])
            characters += int(record["characters"])
        deltas.append(
            (selected_edits - baseline_edits) / characters if characters else 0.0
        )
    deltas.sort()
    return {
        "samples": samples,
        "seed": seed,
        "ci95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
    }


def summarize(records, bootstrap_samples, seed):
    baseline_edits = sum(int(record["baseline_edits"]) for record in records)
    selected_edits = sum(int(record["selected_edits"]) for record in records)
    oracle_edits = sum(int(record["oracle_edits"]) for record in records)
    characters = sum(int(record["characters"]) for record in records)
    changed = sum(bool(record["selection_changed"]) for record in records)
    improved = sum(
        record["selected_edits"] < record["baseline_edits"]
        for record in records
    )
    worsened = sum(
        record["selected_edits"] > record["baseline_edits"]
        for record in records
    )
    parsed = sum(bool(record["parsed"]) for record in records)
    delta = (selected_edits - baseline_edits) / characters if characters else None
    bootstrap = paired_bootstrap(records, bootstrap_samples, seed)
    gate = {
        "parse_complete": parsed == len(records),
        "material_delta": delta is not None and delta <= -0.003,
        "ci_below_zero": bootstrap["ci95"][1] < 0,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "samples": len(records),
        "characters": characters,
        "baseline": {
            "edits": baseline_edits,
            "cer": baseline_edits / characters if characters else None,
        },
        "selected": {
            "edits": selected_edits,
            "cer": selected_edits / characters if characters else None,
        },
        "oracle": {
            "edits": oracle_edits,
            "cer": oracle_edits / characters if characters else None,
        },
        "selected_minus_baseline_cer": delta,
        "paired_bootstrap": bootstrap,
        "parsing": {
            "parsed": parsed,
            "failed": len(records) - parsed,
            "rate": parsed / len(records) if records else 0.0,
        },
        "selection": {
            "changed": changed,
            "unchanged": len(records) - changed,
            "improved": improved,
            "worsened": worsened,
            "equal": len(records) - improved - worsened,
            "changed_rate": changed / len(records) if records else 0.0,
            "improvement_precision": improved / changed if changed else 0.0,
        },
        "phase0_gate": gate,
    }


def load_runtime(model_dir, device, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def generate_responses(torch, tokenizer, model, device, messages, max_new_tokens):
    prompts = [
        tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(
        generated[:, input_width:], skip_special_tokens=True
    ), prompts


def expected_metadata(args, source_sha256, artifacts):
    prompt_contract = {
        "version": PROMPT_VERSION,
        "system": SYSTEM_PROMPT,
        "output_contract": "one rank from 1..top_k; no rewrite",
    }
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "input": str(args.input),
        "input_sha256": source_sha256,
        "expected_samples": args.expected_samples,
        "top_k": args.top_k,
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "directory": str(args.model_dir),
            "artifacts": artifacts,
            "artifacts_sha256": payload_sha256(artifacts),
        },
        "prompt": {
            **prompt_contract,
            "sha256": payload_sha256(prompt_contract),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "device": args.device,
        "code_commit": args.code_commit,
        "bootstrap_samples": args.bootstrap_samples,
    }


def validate_resume(records, source_rows):
    if len(records) > len(source_rows):
        raise ValueError("已有结果行数超过输入")
    for index, record in enumerate(records):
        if record.get("index") != index:
            raise ValueError(f"已有结果 index 不连续：{index}")
        if record.get("uid") != source_rows[index]["uid"]:
            raise ValueError(f"已有结果 UID 与输入不一致：{index}")


def main():
    args = parse_args()
    if args.expected_samples < 1 or args.top_k < 2:
        raise ValueError("expected_samples 和 top_k 必须为正数，top_k 至少为 2")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch_size 和 max_new_tokens 必须大于 0")
    source_rows = read_jsonl(args.input)
    validate_source_rows(source_rows, args.expected_samples, args.top_k)
    artifacts = model_artifacts(args.model_dir)
    metadata = expected_metadata(args, sha256(args.input), artifacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "run_metadata.json"
    results_path = args.output_dir / "selections.jsonl"
    summary_path = args.output_dir / "summary.json"
    if metadata_path.is_file():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_metadata.pop("results_sha256", None)
        saved_metadata.pop("summary_sha256", None)
        saved_metadata["status"] = "RUNNING"
        if saved_metadata != metadata:
            raise ValueError("恢复运行的 metadata 与当前配置不一致")
    else:
        write_json_atomic(metadata_path, metadata)

    records = read_jsonl(results_path) if results_path.is_file() else []
    validate_resume(records, source_rows)
    if len(records) < len(source_rows):
        torch, tokenizer, model = load_runtime(
            args.model_dir, args.device, args.seed
        )
        mode = "a" if records else "w"
        with results_path.open(mode, encoding="utf-8") as output:
            for start in range(len(records), len(source_rows), args.batch_size):
                batch_rows = source_rows[start : start + args.batch_size]
                batch_messages = [
                    build_messages(
                        [candidate["transcript"] for candidate in row["decoder_nbest"]]
                    )
                    for row in batch_rows
                ]
                started = time.perf_counter()
                responses, prompts = generate_responses(
                    torch,
                    tokenizer,
                    model,
                    args.device,
                    batch_messages,
                    args.max_new_tokens,
                )
                elapsed = time.perf_counter() - started
                for offset, (row, response, prompt) in enumerate(
                    zip(batch_rows, responses, prompts)
                ):
                    index = start + offset
                    selected_rank, parser = parse_rank(response, args.top_k)
                    parsed = selected_rank is not None
                    if selected_rank is None:
                        selected_rank = 1
                    nbest = row["decoder_nbest"]
                    selected = nbest[selected_rank - 1]
                    target = row["target"]
                    candidate_edits = [
                        edit_distance(candidate["transcript"], target)
                        for candidate in nbest
                    ]
                    oracle_edits = min(candidate_edits)
                    record = {
                        "index": index,
                        "uid": row["uid"],
                        "domain": row.get("domain"),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "response": response,
                        "parser": parser,
                        "parsed": parsed,
                        "selected_rank": selected_rank,
                        "selection_changed": selected_rank != 1,
                        "baseline_transcript": nbest[0]["transcript"],
                        "selected_transcript": selected["transcript"],
                        "candidate_transcripts_sha256": payload_sha256(
                            [candidate["transcript"] for candidate in nbest]
                        ),
                        "candidate_edits": candidate_edits,
                        "baseline_edits": candidate_edits[0],
                        "selected_edits": candidate_edits[selected_rank - 1],
                        "oracle_edits": oracle_edits,
                        "oracle_ranks": [
                            rank
                            for rank, edits in enumerate(candidate_edits, start=1)
                            if edits == oracle_edits
                        ],
                        "characters": len(target),
                        "batch_inference_seconds": elapsed,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"已完成 {min(start + len(batch_rows), len(source_rows))}/"
                    f"{len(source_rows)}",
                    flush=True,
                )
        records = read_jsonl(results_path)
        validate_resume(records, source_rows)

    summary = summarize(records, args.bootstrap_samples, args.seed)
    summary.update(
        {
            "input_sha256": metadata["input_sha256"],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "model_artifacts_sha256": metadata["model"]["artifacts_sha256"],
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": metadata["prompt"]["sha256"],
            "code_commit": args.code_commit,
            "results_sha256": sha256(results_path),
        }
    )
    write_json_atomic(summary_path, summary)
    metadata["status"] = "COMPLETE"
    metadata["results_sha256"] = summary["results_sha256"]
    metadata["summary_sha256"] = sha256(summary_path)
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
