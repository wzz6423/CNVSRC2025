import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasticity.adapters import ExpertBank, sequence_signature
from plasticity.checkpoint import (
    load_adaptation_checkpoint,
    save_adaptation_checkpoint,
)
from plasticity.engine import ContinualAdaptationEngine
from plasticity.objectives import ctc_target_error
from plasticity.reliability import ReliabilityDecision, ReliabilityGate


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


def main():
    torch.manual_seed(7)
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
    engine = ContinualAdaptationEngine(
        model,
        bank,
        gate,
        device="cpu",
        decoder=None,
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
        save_adaptation_checkpoint(checkpoint_path, feedback_bank, {}, 2)
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
        assert restored_bank.expert_count == feedback_bank.expert_count
        assert all(
            torch.equal(value, restored_bank.state_dict()[name])
            for name, value in feedback_bank.state_dict().items()
        )

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

    short_log_probs = torch.log_softmax(torch.randn(1, 2, 5), dim=-1)
    assert ctc_target_error(short_log_probs, [1, 1]) == "insufficient_ctc_frames"
    invalid_log_probs = short_log_probs.clone()
    invalid_log_probs[0, 0, 0] = float("nan")
    _, invalid_reliability = gate.evaluate(invalid_log_probs, short_log_probs)
    assert not invalid_reliability.accepted
    assert invalid_reliability.score == 0.0
    print("RSP-VSR plasticity smoke 通过")


if __name__ == "__main__":
    main()
