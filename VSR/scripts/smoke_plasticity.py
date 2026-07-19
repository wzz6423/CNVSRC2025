import json
import random
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continual_adapt import _prepare_result_file, _resolve_device, _write_json_atomic
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from plasticity.adapters import ExpertBank, sequence_signature
from plasticity.checkpoint import (
    capture_rng_state,
    load_adaptation_checkpoint,
    restore_rng_state,
    save_adaptation_checkpoint,
)
from plasticity.engine import ContinualAdaptationEngine
from plasticity.metrics import StreamMetrics
from plasticity.objectives import ctc_sequence_loss, ctc_target_error
from plasticity.reliability import ReliabilityDecision, ReliabilityGate
from scripts.prepare_stream_manifest import ordered_rows


class MockEncoder(nn.Module):
    def __init__(self, channels, feature_dim):
        super().__init__()
        self.projection = nn.Linear(channels, feature_dim, bias=False)

    def forward(self, video, _mask):
        pooled = video.mean(dim=(-1, -2))
        return self.projection(pooled), None


class MockCTC(nn.Module):
    def __init__(self, feature_dim, vocabulary_size):
        super().__init__()
        self.ctc_lo = nn.Linear(feature_dim, vocabulary_size)
        with torch.no_grad():
            self.ctc_lo.bias.zero_()
            self.ctc_lo.bias[1] = 2.0

    def log_softmax(self, features):
        return torch.log_softmax(self.ctc_lo(features), dim=-1)


class MockModel(nn.Module):
    def __init__(self, channels=1, feature_dim=8, vocabulary_size=5):
        super().__init__()
        self.encoder = MockEncoder(channels, feature_dim)
        self.ctc = MockCTC(feature_dim, vocabulary_size)


class PostUpdateCollapseGate:
    def __init__(self):
        self.calls = 0

    def evaluate(
        self, _clean_log_probs, _augmented_log_probs, decoder_tokens=None
    ):
        self.calls += 1
        accepted = self.calls < 3
        tokens = [1] if accepted else []
        return tokens, ReliabilityDecision(
            accepted=accepted,
            score=1.0,
            confidence=1.0,
            view_consistency=1.0,
            view_token_agreement=1.0,
            decoder_agreement=1.0 if decoder_tokens is not None else None,
            emission_rate=0.5 if accepted else 0.0,
            pseudo_token_count=len(tokens),
            reasons=() if accepted else ("invalid_emission_rate",),
        )


class CountingDecoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, _features):
        self.calls += 1
        return "1", [1]


def main():
    torch.manual_seed(7)
    assert _resolve_device("auto").type in {"cpu", "cuda", "mps"}
    model = MockModel()
    bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=3,
        route_threshold=0.95,
        growth_patience=2,
        pending_similarity=0.9,
        signature_motion_order=2,
    )
    gate = ReliabilityGate(
        min_score=0.0,
        min_confidence=0.0,
        min_view_agreement=0.0,
        min_emission_rate=0.0,
        max_emission_rate=1.0,
    )
    decoder = CountingDecoder()
    engine = ContinualAdaptationEngine(
        model,
        bank,
        gate,
        device="cpu",
        decoder=decoder,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    video = torch.randn(12, 1, 8, 8)
    engine.process(video)
    adapter_before = {
        name: value.detach().clone()
        for name, value in bank.experts[0].state_dict().items()
    }
    outcome = engine.process(video)
    assert outcome.update.status == "accepted"
    assert decoder.calls == 4
    assert bank.expert_count == 1
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert any(
        not torch.equal(adapter_before[name], value)
        for name, value in bank.experts[0].state_dict().items()
    )

    slow = torch.tensor([0.0, 0.0, 1.0, 1.0]).view(1, 4, 1)
    alternating = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 4, 1)
    static_slow = sequence_signature(slow, motion_order=0)
    static_alternating = sequence_signature(alternating, motion_order=0)
    motion_slow = sequence_signature(slow, motion_order=2)
    motion_alternating = sequence_signature(alternating, motion_order=2)
    assert static_slow.numel() == 2
    assert motion_slow.numel() == 4
    assert torch.allclose(static_slow, static_alternating)
    assert not torch.allclose(motion_slow, motion_alternating)

    routing_bank = ExpertBank(
        feature_dim=4,
        bottleneck_dim=2,
        max_experts=2,
        route_threshold=0.9,
        growth_patience=2,
        pending_similarity=0.9,
    )
    first = torch.zeros(8)
    first[0] = 1.0
    shifted = torch.zeros(8)
    shifted[1] = 1.0
    routing_bank.route(first)
    quarantine = routing_bank.route(shifted)
    assert quarantine.quarantined
    decision = routing_bank.route(shifted)
    assert decision.created
    assert routing_bank.expert_count == 2
    third_domain = torch.zeros(8)
    third_domain[2] = 1.0
    at_capacity = routing_bank.route(third_domain)
    assert at_capacity.quarantined
    assert not at_capacity.created

    feedback_bank = ExpertBank(
        feature_dim=4,
        bottleneck_dim=2,
        max_experts=2,
        route_threshold=0.9,
        growth_patience=6,
        pending_similarity=0.9,
        signature_motion_order=2,
    )
    feedback_first = torch.zeros(16)
    feedback_first[0] = 1.0
    feedback_shifted = torch.zeros(16)
    feedback_shifted[1] = 1.0
    feedback_bank.route(feedback_first)
    feedback_growth = feedback_bank.route(
        feedback_shifted, confirm_shift=True
    )
    assert feedback_growth.created
    assert not feedback_growth.quarantined

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "adaptation_state.pt"
        metrics = StreamMetrics()
        metrics.update("a", "b", "domain-a", 0.5, "accepted")
        rng_state = capture_rng_state()
        save_adaptation_checkpoint(
            checkpoint_path,
            feedback_bank,
            {},
            2,
            metrics_state=metrics.state_dict(),
            rng_state=rng_state,
            stream_state={"manifest_sha256": "test"},
        )
        restored_bank = ExpertBank(
            feature_dim=4,
            bottleneck_dim=2,
            max_experts=2,
            route_threshold=0.9,
            growth_patience=6,
            pending_similarity=0.9,
            signature_motion_order=2,
        )
        restored = load_adaptation_checkpoint(
            checkpoint_path, restored_bank, "cpu"
        )
        assert restored["processed_samples"] == 2
        assert restored["metrics_state"] == metrics.state_dict()
        assert restored_bank.expert_count == feedback_bank.expert_count
        assert all(
            torch.equal(value, restored_bank.state_dict()[name])
            for name, value in feedback_bank.state_dict().items()
        )

        random.seed(123)
        torch.manual_seed(123)
        saved_rng = capture_rng_state()
        expected_python = random.random()
        expected_torch = torch.rand(3)
        restore_rng_state(saved_rng)
        assert random.random() == expected_python
        assert torch.equal(torch.rand(3), expected_torch)

        result_path = Path(temporary_directory) / "stream_results.jsonl"
        result_path.write_text("0\n1\n2\n", encoding="utf-8")
        assert _prepare_result_file(result_path, 2) == "a"
        assert result_path.read_text(encoding="utf-8") == "0\n1\n"

        summary_path = Path(temporary_directory) / "summary.json"
        _write_json_atomic(summary_path, {"samples": 2, "mode": "static"})
        assert json.loads(summary_path.read_text(encoding="utf-8"))["samples"] == 2
        assert not (Path(temporary_directory) / ".summary.json.tmp").exists()

    rollback_model = MockModel()
    rollback_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    rollback_bank.add_expert(torch.randn(16))
    rollback_engine = ContinualAdaptationEngine(
        rollback_model,
        rollback_bank,
        gate,
        device="cpu",
        decoder=None,
        learning_rate=0.1,
        weight_decay=0.0,
        rollback_enabled=True,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=0.0,
        max_target_loss_increase=100.0,
    )
    rollback_before = {
        name: value.detach().clone()
        for name, value in rollback_bank.experts[0].state_dict().items()
    }
    rollback_outcome = rollback_engine.process(video)
    assert rollback_outcome.update.status == "rolled_back"
    assert all(
        torch.equal(rollback_before[name], value)
        for name, value in rollback_bank.experts[0].state_dict().items()
    )

    collapse_model = MockModel()
    collapse_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    collapse_bank.add_expert(torch.randn(16))
    collapse_gate = PostUpdateCollapseGate()
    collapse_engine = ContinualAdaptationEngine(
        collapse_model,
        collapse_bank,
        collapse_gate,
        device="cpu",
        decoder=None,
        learning_rate=0.1,
        weight_decay=0.0,
        rollback_enabled=True,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    collapse_before = {
        name: value.detach().clone()
        for name, value in collapse_bank.experts[0].state_dict().items()
    }
    collapse_outcome = collapse_engine.process(video)
    assert collapse_outcome.update.status == "rolled_back"
    assert "post_update_unreliable" in collapse_outcome.update.reasons
    assert all(
        torch.equal(collapse_before[name], value)
        for name, value in collapse_bank.experts[0].state_dict().items()
    )

    disabled_model = MockModel()
    disabled_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    disabled_gate = PostUpdateCollapseGate()
    disabled_engine = ContinualAdaptationEngine(
        disabled_model,
        disabled_bank,
        disabled_gate,
        device="cpu",
        decoder=None,
        reliability_enabled=False,
        rollback_enabled=True,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    disabled_outcome = disabled_engine.process(video)
    assert disabled_outcome.update.status == "accepted"

    optimizer_states = engine.optimizer_state_dict()
    assert optimizer_states
    restored_optimizer_model = MockModel()
    restored_optimizer_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=3,
        route_threshold=0.95,
        growth_patience=2,
        pending_similarity=0.9,
        signature_motion_order=2,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "optimizer_state.pt"
        save_adaptation_checkpoint(
            checkpoint_path,
            bank,
            {},
            2,
            optimizer_states=optimizer_states,
        )
        checkpoint = load_adaptation_checkpoint(
            checkpoint_path, restored_optimizer_bank, "cpu"
        )
        restored_optimizer_engine = ContinualAdaptationEngine(
            restored_optimizer_model,
            restored_optimizer_bank,
            gate,
            device="cpu",
            decoder=None,
            rollback_enabled=False,
            view_noise_std=0.0,
            temporal_mask_ratio=0.0,
            max_anchor_kl=100.0,
            max_target_loss_increase=100.0,
        )
        restored_optimizer_engine.load_optimizer_state_dict(
            checkpoint["optimizer_states"]
        )
        assert restored_optimizer_engine.optimizer_state_dict()

    rows = [
        {"domain": domain, "value": value}
        for domain in ("a", "b", "c", "d")
        for value in range(2)
    ]
    first_order = ordered_rows(rows, "domain-block", False, True, random.Random(42))
    repeated_order = ordered_rows(
        rows, "domain-block", False, True, random.Random(42)
    )
    assert first_order == repeated_order
    assert [row["domain"] for row in first_order[::2]] != ["a", "b", "c", "d"]

    short_log_probs = torch.log_softmax(torch.randn(1, 2, 5), dim=-1)
    assert ctc_target_error(short_log_probs, [1, 1]) == "insufficient_ctc_frames"
    invalid_log_probs = short_log_probs.clone()
    invalid_log_probs[0, 0, 0] = float("nan")
    _, invalid_reliability = gate.evaluate(invalid_log_probs, short_log_probs)
    assert not invalid_reliability.accepted
    assert invalid_reliability.score == 0.0
    if torch.backends.mps.is_available():
        mps_logits = torch.randn(1, 8, 5, device="mps", requires_grad=True)
        mps_loss = ctc_sequence_loss(
            torch.log_softmax(mps_logits, dim=-1), [1, 2, 3]
        )
        mps_loss.backward()
        assert mps_loss.device.type == "mps"
        assert torch.isfinite(mps_logits.grad).all()
        prefix_scorer = CTCPrefixScoreTH(
            torch.log_softmax(torch.randn(1, 8, 5, device="mps"), dim=-1),
            torch.tensor([8]),
            blank=0,
            eos=4,
        )
        prefix_scores, _ = prefix_scorer(
            torch.tensor([[4]], device="mps"),
            None,
            scoring_ids=torch.tensor([[1, 2]], device="mps"),
        )
        assert prefix_scores.device.type == "mps"
        assert torch.isfinite(prefix_scores[:, 1:3]).all()
    print("RSP-VSR plasticity smoke 通过")


if __name__ == "__main__":
    main()
