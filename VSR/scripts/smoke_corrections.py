#!/usr/bin/env python3
"""Smoke tests for target-conditioned CTC forward-backward occupancy."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.corrections import (
    CorrectionDiagnostics,
    TargetConditionedOccupancy,
    ctc_forward_backward_occupancy,
    localize_feedback_correction,
    localize_target_conditioned_occupancy,
    randomized_error_support,
)


def _make_log_probs(frame_token_rows, vocabulary_size, blank_id=0, peak=6.0):
    """Build peaked log-probs from per-frame soft token weights.

    frame_token_rows: list[list[tuple[token, weight]]]
    """
    time_steps = len(frame_token_rows)
    logits = torch.full((1, time_steps, vocabulary_size), -peak)
    for time_index, pairs in enumerate(frame_token_rows):
        for token, weight in pairs:
            logits[0, time_index, int(token)] = float(weight)
        # Ensure blank has a defined baseline when not listed.
        if all(int(token) != blank_id for token, _ in pairs):
            logits[0, time_index, blank_id] = -peak
    return torch.log_softmax(logits, dim=-1)


def _logsumexp(values):
    if not values:
        return float("-inf")
    max_value = max(values)
    if max_value == float("-inf"):
        return float("-inf")
    total = 0.0
    for value in values:
        total += math.exp(value - max_value)
    return max_value + math.log(total)


def exhaustive_token_occupancy_and_ll(log_probs, target_tokens, blank_id=0):
    """Enumerate all valid CTC alignments on a tiny lattice."""
    target = tuple(int(token) for token in target_tokens)
    blank = int(blank_id)
    time_steps = int(log_probs.size(1))
    target_length = len(target)
    state_count = 2 * target_length + 1
    labels = [
        blank if state_index % 2 == 0 else target[state_index // 2]
        for state_index in range(state_count)
    ]
    emission = [
        [float(log_probs[0, time_index, labels[state_index]].item()) for state_index in range(state_count)]
        for time_index in range(time_steps)
    ]

    path_log_probs = []
    occupancy_mass = [
        [0.0 for _ in range(target_length)] for _ in range(time_steps)
    ]

    def transitions_to(state_index):
        predecessors = [state_index]
        if state_index > 0:
            predecessors.append(state_index - 1)
        if (
            state_index > 1
            and labels[state_index] != blank
            and labels[state_index] != labels[state_index - 2]
        ):
            predecessors.append(state_index - 2)
        return predecessors

    def dfs(time_index, state_index, log_path, state_trace):
        if time_index == time_steps - 1:
            if state_index not in {state_count - 1, state_count - 2}:
                return
            path_log_probs.append(log_path)
            probability = math.exp(log_path)
            for frame_index, path_state in enumerate(state_trace):
                if path_state % 2 == 1:
                    token_index = path_state // 2
                    occupancy_mass[frame_index][token_index] += probability
            return

        next_time = time_index + 1
        for next_state in range(state_count):
            if state_index not in transitions_to(next_state):
                continue
            dfs(
                next_time,
                next_state,
                log_path + emission[next_time][next_state],
                state_trace + [next_state],
            )

    for start_state in (0, 1):
        if start_state >= state_count:
            continue
        dfs(
            0,
            start_state,
            emission[0][start_state],
            [start_state],
        )

    log_likelihood = _logsumexp(path_log_probs)
    if log_likelihood == float("-inf"):
        raise RuntimeError("穷举未找到可行路径")
    total_probability = math.exp(log_likelihood)
    occupancy = torch.zeros(1, time_steps, target_length)
    for time_index in range(time_steps):
        for token_index in range(target_length):
            occupancy[0, time_index, token_index] = (
                occupancy_mass[time_index][token_index] / total_probability
            )
    return occupancy, log_likelihood


def assert_close(actual, expected, atol=1e-5, label="value"):
    if isinstance(actual, torch.Tensor):
        actual_value = float(actual.item()) if actual.numel() == 1 else actual
    else:
        actual_value = actual
    if isinstance(actual_value, torch.Tensor):
        if not torch.allclose(actual_value, expected, atol=atol, rtol=0):
            raise AssertionError(
                f"{label} 不匹配：max_abs="
                f"{(actual_value - expected).abs().max().item()}"
            )
    elif abs(actual_value - expected) > atol:
        raise AssertionError(f"{label} 不匹配：{actual_value} vs {expected}")


def test_forward_backward_matches_enumeration():
    cases = [
        {
            "name": "single_token",
            "target": (1,),
            "rows": [
                [(0, 2.0), (1, 4.0)],
                [(0, 1.0), (1, 5.0)],
                [(0, 3.0), (1, 2.0)],
            ],
            "vocabulary_size": 3,
        },
        {
            "name": "two_distinct_tokens",
            "target": (1, 2),
            "rows": [
                [(0, 3.0), (1, 4.0), (2, 0.0)],
                [(0, 1.0), (1, 4.0), (2, 2.0)],
                [(0, 2.0), (1, 1.0), (2, 4.0)],
                [(0, 3.0), (1, 0.0), (2, 3.0)],
            ],
            "vocabulary_size": 4,
        },
        {
            "name": "repeated_tokens",
            "target": (1, 1),
            "rows": [
                [(0, 2.0), (1, 4.0)],
                [(0, 3.0), (1, 2.0)],
                [(0, 1.0), (1, 5.0)],
                [(0, 2.0), (1, 3.0)],
            ],
            "vocabulary_size": 3,
        },
    ]
    for case in cases:
        log_probs = _make_log_probs(
            case["rows"], case["vocabulary_size"], blank_id=0, peak=5.0
        )
        occupancy, log_likelihood = ctc_forward_backward_occupancy(
            log_probs, case["target"], blank_id=0
        )
        reference_occupancy, reference_ll = exhaustive_token_occupancy_and_ll(
            log_probs, case["target"], blank_id=0
        )
        assert_close(
            log_likelihood,
            reference_ll,
            atol=1e-4,
            label=f"{case['name']}.log_likelihood",
        )
        assert_close(
            occupancy,
            reference_occupancy,
            atol=1e-4,
            label=f"{case['name']}.occupancy",
        )
        # Occupancy is a stop-gradient quantity derived from detached emissions.
        assert not occupancy.requires_grad
        assert occupancy.device == log_probs.device
        # Each frame's token-state mass is in [0, 1].
        assert torch.all(occupancy >= -1e-6)
        assert torch.all(occupancy <= 1.0 + 1e-5)


def test_repeated_token_forbids_skip():
    # With identical consecutive tokens, T == L is insufficient; need blank separation.
    logits = torch.full((1, 2, 3), -8.0)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 1] = 8.0
    log_probs = torch.log_softmax(logits, dim=-1)
    try:
        ctc_forward_backward_occupancy(log_probs, [1, 1], blank_id=0)
        raise AssertionError("重复 token 在帧数不足时应失败")
    except ValueError as error:
        assert "帧数不足" in str(error)

    # With an intervening blank frame, a valid path exists and occupancy is positive.
    logits = torch.full((1, 3, 3), -8.0)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 0] = 8.0
    logits[0, 2, 1] = 8.0
    log_probs = torch.log_softmax(logits, dim=-1)
    occupancy, log_likelihood = ctc_forward_backward_occupancy(
        log_probs, [1, 1], blank_id=0
    )
    reference_occupancy, reference_ll = exhaustive_token_occupancy_and_ll(
        log_probs, [1, 1], blank_id=0
    )
    assert_close(log_likelihood, reference_ll, atol=1e-4, label="repeat.ll")
    assert_close(occupancy, reference_occupancy, atol=1e-4, label="repeat.occ")
    assert occupancy[0, 0, 0] > 0.5
    assert occupancy[0, 2, 1] > 0.5


def test_edit_mapping_substitution_deletion_insertion():
    # Greedy path collapses to [1, 2, 3]; target is [1, 4, 3, 5]
    # match 1, substitution 2->4, match 3, deletion of 5.
    alignment_logits = torch.full((1, 8, 6), -8.0)
    for frame_index, token in enumerate((0, 1, 1, 0, 2, 2, 0, 3)):
        alignment_logits[0, frame_index, token] = 8.0
    log_probs = torch.log_softmax(alignment_logits, dim=-1)
    target = [1, 4, 3, 5]

    # Legacy greedy API remains compatible.
    diagnostics, correct_frame_mask = localize_feedback_correction(log_probs, target)
    assert isinstance(diagnostics, CorrectionDiagnostics)
    assert diagnostics.to_dict() == {
        "predicted_tokens": 3,
        "target_tokens": 4,
        "matched_tokens": 2,
        "substituted_tokens": 1,
        "missing_target_tokens": 1,
        "extra_prediction_tokens": 0,
        "token_error_rate": 0.5,
        "matched_frame_rate": 0.375,
    }
    assert correct_frame_mask[0].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]

    result = localize_target_conditioned_occupancy(log_probs, target)
    assert isinstance(result, TargetConditionedOccupancy)
    assert result.matched_target_indices == (0, 2)
    assert result.substitution_target_indices == (1,)
    assert result.deletion_target_indices == (3,)
    assert result.extra_prediction_frame_mask[0].tolist() == [False] * 8
    assert result.token_occupancy.shape == (1, 8, 4)
    assert result.matched_target_occupancy.shape == (1, 8)
    assert result.error_target_occupancy.shape == (1, 8)
    # Matched / error occupancy are sums over the corresponding token states.
    assert torch.allclose(
        result.matched_target_occupancy,
        result.token_occupancy[:, :, [0, 2]].sum(dim=-1),
        atol=1e-6,
    )
    assert torch.allclose(
        result.error_target_occupancy,
        result.token_occupancy[:, :, [1, 3]].sum(dim=-1),
        atol=1e-6,
    )

    # Insertion: greedy has an extra token that target lacks.
    insert_logits = torch.full((1, 6, 5), -8.0)
    for frame_index, token in enumerate((0, 1, 1, 2, 2, 0)):
        insert_logits[0, frame_index, token] = 8.0
    insert_log_probs = torch.log_softmax(insert_logits, dim=-1)
    insert_result = localize_target_conditioned_occupancy(insert_log_probs, [1])
    assert insert_result.matched_target_indices == (0,)
    assert insert_result.substitution_target_indices == ()
    assert insert_result.deletion_target_indices == ()
    # Greedy segments: token1 frames [1,2], token2 frames [3,4]; extra is token2.
    assert insert_result.extra_prediction_frame_mask[0].tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_no_grad_and_device_preservation():
    logits = torch.full((1, 4, 4), -5.0, requires_grad=True)
    for frame_index, token in enumerate((1, 0, 2, 0)):
        logits.data[0, frame_index, token] = 5.0
    log_probs = torch.log_softmax(logits, dim=-1)
    occupancy, log_likelihood = ctc_forward_backward_occupancy(
        log_probs, [1, 2], blank_id=0
    )
    assert occupancy.device == log_probs.device
    assert log_likelihood.device == log_probs.device
    assert not occupancy.requires_grad
    # Occupancy itself is detached; summing it must not create a graph into logits.
    try:
        occupancy.sum().backward()
        raise AssertionError("occupancy 不应暴露对参数的梯度")
    except RuntimeError:
        pass
    result = localize_target_conditioned_occupancy(log_probs, [1, 2])
    assert result.token_occupancy.device == log_probs.device
    assert result.extra_prediction_frame_mask.device == log_probs.device


def test_randomized_support_is_deterministic_and_mass_matched():
    logits = torch.full((1, 6, 5), -8.0)
    for frame_index, token in enumerate((0, 1, 1, 2, 2, 0)):
        logits[0, frame_index, token] = 8.0
    localization = localize_target_conditioned_occupancy(
        torch.log_softmax(logits, dim=-1), [1]
    )
    occupancy_a, insertion_a = randomized_error_support(
        localization, "sample-42", seed=7
    )
    occupancy_b, insertion_b = randomized_error_support(
        localization, "sample-42", seed=7
    )
    assert torch.equal(occupancy_a, occupancy_b)
    assert torch.equal(insertion_a, insertion_b)
    assert torch.allclose(
        occupancy_a.sum(dim=1), localization.token_occupancy.sum(dim=1)
    )
    assert insertion_a.sum() == localization.extra_prediction_frame_mask.sum()
    assert not torch.equal(insertion_a, localization.extra_prediction_frame_mask)


def test_invalid_inputs():
    valid = torch.log_softmax(torch.randn(1, 4, 4), dim=-1)

    def expect_value_error(callable_fn, text_fragment):
        try:
            callable_fn()
            raise AssertionError(f"期望 ValueError 含有：{text_fragment}")
        except ValueError as error:
            if text_fragment not in str(error):
                raise AssertionError(
                    f"错误信息未包含 {text_fragment!r}：{error}"
                ) from error

    expect_value_error(
        lambda: ctc_forward_backward_occupancy(torch.randn(2, 4, 4), [1]),
        "[1, T, V]",
    )
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(
            torch.zeros(1, 4, 4, dtype=torch.long), [1]
        ),
        "浮点",
    )
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(valid, []),
        "target 不能为空",
    )
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(valid, [0]),
        "blank",
    )
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(valid, [9]),
        "越界",
    )
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(valid[:, :1, :], [1, 2]),
        "帧数不足",
    )
    non_finite = valid.clone()
    non_finite[0, 0, 0] = float("nan")
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(non_finite, [1]),
        "有限",
    )

    # Unreachable non-blank token: blank mass only, token log-prob is -inf.
    # -inf is not finite, so validation rejects before forward-backward.
    unreachable = torch.full((1, 2, 3), float("-inf"))
    unreachable[:, :, 0] = 0.0
    expect_value_error(
        lambda: ctc_forward_backward_occupancy(unreachable, [1], blank_id=0),
        "有限",
    )


def test_legacy_diagnostics_schema_unchanged():
    keys = set(CorrectionDiagnostics(
        predicted_tokens=0,
        target_tokens=0,
        matched_tokens=0,
        substituted_tokens=0,
        missing_target_tokens=0,
        extra_prediction_tokens=0,
        token_error_rate=0.0,
        matched_frame_rate=0.0,
    ).to_dict())
    assert keys == {
        "predicted_tokens",
        "target_tokens",
        "matched_tokens",
        "substituted_tokens",
        "missing_target_tokens",
        "extra_prediction_tokens",
        "token_error_rate",
        "matched_frame_rate",
    }


def main():
    tests = [
        ("forward_backward_matches_enumeration", test_forward_backward_matches_enumeration),
        ("repeated_token_forbids_skip", test_repeated_token_forbids_skip),
        ("edit_mapping_substitution_deletion_insertion", test_edit_mapping_substitution_deletion_insertion),
        ("no_grad_and_device_preservation", test_no_grad_and_device_preservation),
        (
            "randomized_support_is_deterministic_and_mass_matched",
            test_randomized_support_is_deterministic_and_mass_matched,
        ),
        ("invalid_inputs", test_invalid_inputs),
        ("legacy_diagnostics_schema_unchanged", test_legacy_diagnostics_schema_unchanged),
    ]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 - smoke harness reports all failures
            failures.append((name, error))
            print(f"FAIL {name}: {error}")
    print(
        f"\n{len(tests) - len(failures)} passed, {len(failures)} failed, "
        f"{len(tests)} total"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
