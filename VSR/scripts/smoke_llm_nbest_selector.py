#!/usr/bin/env python3
from evaluate_llm_nbest_selector import (
    PROMPT_VERSION,
    build_messages,
    paired_bootstrap,
    parse_rank,
    payload_sha256,
    summarize,
)


def main():
    candidates = [f"候选文本{i}" for i in range(1, 11)]
    messages = build_messages(candidates)
    serialized = str(messages)
    assert PROMPT_VERSION == "dev8-constrained-choice-v1"
    assert "不得改写" in serialized
    assert "绝密参考答案" not in serialized
    assert all(candidate in serialized for candidate in candidates)
    assert payload_sha256(messages) == payload_sha256(messages)

    assert parse_rank("7", 10) == (7, "exact_integer")
    assert parse_rank('{"rank": 3}', 10) == (3, "json_rank")
    assert parse_rank("候选：10", 10) == (10, "rank_field")
    assert parse_rank("我选择第 4 条", 10) == (4, "single_number")
    assert parse_rank("候选 2 或候选 3", 10) == (None, "unparsed")
    assert parse_rank("无法判断", 10) == (None, "unparsed")
    assert parse_rank("11", 10) == (None, "unparsed")

    records = [
        {
            "baseline_edits": 5,
            "selected_edits": 4,
            "oracle_edits": 3,
            "characters": 10,
            "selection_changed": True,
            "parsed": True,
        },
        {
            "baseline_edits": 3,
            "selected_edits": 3,
            "oracle_edits": 2,
            "characters": 10,
            "selection_changed": False,
            "parsed": True,
        },
    ]
    bootstrap = paired_bootstrap(records, samples=100, seed=42)
    assert bootstrap["ci95"][1] <= 0.0
    summary = summarize(records, bootstrap_samples=100, seed=42)
    assert summary["baseline"]["cer"] == 0.4
    assert summary["selected"]["cer"] == 0.35
    assert summary["selected_minus_baseline_cer"] == -0.05
    assert summary["selection"]["changed"] == 1
    assert summary["parsing"]["failed"] == 0
    print("LLM_NBEST_SELECTOR_SMOKE_OK")


if __name__ == "__main__":
    main()
