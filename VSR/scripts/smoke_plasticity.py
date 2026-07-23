import copy
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continual_adapt import (
    _effective_stream_samples,
    _file_sha256,
    _prepare_result_file,
    _resource_summary,
    _resolve_device,
    _restore_feedback_policy,
    _stream_state,
    _validate_stream_state,
    _write_json_atomic,
)
from datamodule.transforms import DICT_PATH
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from plasticity.adapters import ExpertBank, sequence_signature
from plasticity.artifacts import (
    append_metrics_history,
    prepare_metrics_history,
    retain_best_checkpoints,
    reset_best_checkpoints,
)
from plasticity.checkpoint import (
    capture_rng_state,
    load_adaptation_checkpoint,
    restore_rng_state,
    save_adaptation_checkpoint,
)
from plasticity.corrections import localize_feedback_correction
from plasticity.engine import ContinualAdaptationEngine
from plasticity.feedback import FeedbackQueryPolicy
from plasticity.metrics import StreamMetrics
from plasticity.objectives import (
    ctc_posterior_entropy,
    ctc_sequence_loss,
    ctc_target_error,
    occupancy_weighted_blank_loss,
    occupancy_weighted_posterior_kl,
    occupancy_weighted_token_loss,
    posterior_kl,
)
from plasticity.parameter_adaptation import (
    LoRALinear,
    configure_attention_lora_adaptation,
    named_lora_parameters,
    parameter_state,
)
from plasticity.reliability import ReliabilityDecision, ReliabilityGate
from plasticity.routing_diagnostics import summarize_route_records
from plasticity.stream import iter_stream_manifest
from scripts.prepare_stream_manifest import domain_from_path, ordered_rows


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


class MockBatchNormEncoder(nn.Module):
    def __init__(self, channels=2, feature_dim=8):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(channels)
        self.projection = nn.Linear(channels, feature_dim, bias=False)

    def forward(self, video, _mask):
        batch, frames, channels, height, width = video.shape
        normalized = self.batch_norm(
            video.reshape(batch * frames, channels, height, width)
        )
        pooled = normalized.mean(dim=(-1, -2)).reshape(batch, frames, channels)
        return self.projection(pooled), None


class MockBatchNormModel(nn.Module):
    def __init__(self, channels=2, feature_dim=8, vocabulary_size=5):
        super().__init__()
        self.encoder = MockBatchNormEncoder(channels, feature_dim)
        self.ctc = MockCTC(feature_dim, vocabulary_size)


class MockSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear_q = nn.Linear(feature_dim, feature_dim)
        self.linear_k = nn.Linear(feature_dim, feature_dim)
        self.linear_v = nn.Linear(feature_dim, feature_dim)
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        combined = (
            self.linear_q(features)
            + self.linear_k(features)
            + self.linear_v(features)
        ) / 3.0
        return self.linear_out(torch.tanh(combined))


class MockAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.self_attn = MockSelfAttention(feature_dim)

    def forward(self, features):
        return features + self.self_attn(features)


class MockLoRAEncoder(nn.Module):
    def __init__(self, channels=1, feature_dim=8, layers=2):
        super().__init__()
        self.projection = nn.Linear(channels, feature_dim, bias=False)
        self.encoders = nn.ModuleList(
            MockAttentionLayer(feature_dim) for _ in range(layers)
        )

    def forward(self, video, _mask):
        features = self.projection(video.mean(dim=(-1, -2)))
        for layer in self.encoders:
            features = layer(features)
        return features, None


class MockLoRAModel(nn.Module):
    def __init__(self, channels=1, feature_dim=8, vocabulary_size=5):
        super().__init__()
        self.encoder = MockLoRAEncoder(channels, feature_dim)
        self.decoder = nn.ModuleList((MockAttentionLayer(feature_dim),))
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


class EmptyReliabilityGate:
    def evaluate(self, _clean_log_probs, _augmented_log_probs, decoder_tokens=None):
        return [], ReliabilityDecision(
            accepted=False,
            score=0.0,
            confidence=0.0,
            view_consistency=0.0,
            view_token_agreement=0.0,
            decoder_agreement=0.0 if decoder_tokens is not None else None,
            emission_rate=0.0,
            pseudo_token_count=0,
            reasons=("empty_pseudo_label",),
        )


def main():
    torch.manual_seed(7)
    scores = [0.9, 0.8, 0.7, 0.6, 0.5] * 5
    periodic = FeedbackQueryPolicy("periodic", 5, len(scores), random_seed=7)
    periodic_queries = [
        index
        for index, score in enumerate(scores)
        if periodic.decide(index, score).queried
    ]
    assert periodic_queries == [4, 9, 14, 19, 24]
    assert periodic.summary()["policy_queries"] == 5

    random_policy = FeedbackQueryPolicy("random", 5, len(scores), random_seed=7)
    repeated_random = FeedbackQueryPolicy("random", 5, len(scores), random_seed=7)
    random_queries = [
        index
        for index, score in enumerate(scores)
        if random_policy.decide(index, score).queried
    ]
    repeated_queries = [
        index
        for index, score in enumerate(scores)
        if repeated_random.decide(index, score).queried
    ]
    assert random_queries == repeated_queries
    assert len(random_queries) == 5
    assert [index // 5 for index in random_queries] == list(range(5))

    uncertainty_scores = [
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
    ]
    uncertainty = FeedbackQueryPolicy(
        "uncertainty", 5, len(uncertainty_scores), random_seed=7
    )
    uncertainty_decisions = [
        uncertainty.decide(index, score)
        for index, score in enumerate(uncertainty_scores)
    ]
    assert [
        index
        for index, decision in enumerate(uncertainty_decisions)
        if decision.queried
    ] == [4, 5, 14]
    assert uncertainty_decisions[5].reason == (
        "uncertainty_below_history_quantile"
    )
    assert uncertainty_decisions[14].reason == "uncertainty_window_fallback"

    restored_uncertainty = FeedbackQueryPolicy(
        "uncertainty", 5, len(uncertainty_scores), random_seed=7
    )
    for index, score in enumerate(uncertainty_scores[:8]):
        assert restored_uncertainty.decide(index, score) == (
            uncertainty_decisions[index]
        )
    for index, score in enumerate(uncertainty_scores[8:], start=8):
        assert restored_uncertainty.decide(index, score) == (
            uncertainty_decisions[index]
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        result_path = Path(temporary_directory) / "stream_results.jsonl"
        with result_path.open("w", encoding="utf-8") as handle:
            for index, decision in enumerate(uncertainty_decisions[:8]):
                handle.write(
                    json.dumps(
                        {
                            "reliability": {
                                "score": uncertainty_scores[index]
                            },
                            "feedback_query": decision.to_dict(),
                        }
                    )
                    + "\n"
                )
        replayed_uncertainty = FeedbackQueryPolicy(
            "uncertainty", 5, len(uncertainty_scores), random_seed=7
        )
        _restore_feedback_policy(replayed_uncertainty, result_path, 8)
        for index, score in enumerate(uncertainty_scores[8:], start=8):
            assert replayed_uncertainty.decide(index, score) == (
                uncertainty_decisions[index]
            )

        manifest_path = Path(temporary_directory) / "stream.jsonl"
        manifest_path.write_text("{}\n{}\n{}\n", encoding="utf-8")
        assert _effective_stream_samples(manifest_path, None) == 3
        assert _effective_stream_samples(manifest_path, 2) == 2

    with_manifest = FeedbackQueryPolicy("periodic", 5, 6, random_seed=7)
    manifest_decision = with_manifest.decide(
        0, 0.9, manifest_requested=True
    )
    assert manifest_decision.queried
    assert manifest_decision.source == "manifest"
    assert with_manifest.summary()["manifest_queries"] == 1

    resource_model = MockModel()
    resource_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    resource_bank.ensure_experts(1)
    resource_engine = ContinualAdaptationEngine(
        resource_model,
        resource_bank,
        ReliabilityGate(min_score=0.0),
        device="cpu",
        decoder=None,
        enabled=False,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "adaptation_state.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        resources = _resource_summary(
            resource_engine,
            {
                "samples": 5,
                "timing": {"total_process_seconds": 2.0},
            },
            checkpoint_path,
            temporary_directory,
            torch.device("cpu"),
        )
    assert resources["base_model_parameters"] > 0
    assert resources["adapter_bank_parameters"] > 0
    assert resources["updatable_parameters"] == 0
    assert resources["samples_per_process_second"] == 2.5
    assert resources["latest_checkpoint_bytes"] == len(b"checkpoint")
    assert resources["retained_checkpoint_files"] == 1
    assert resources["peak_gpu_memory_allocated_bytes"] is None

    alignment_logits = torch.full((1, 8, 6), -8.0)
    for frame_index, token in enumerate((0, 1, 1, 0, 2, 2, 0, 3)):
        alignment_logits[0, frame_index, token] = 8.0
    alignment_log_probs = torch.log_softmax(alignment_logits, dim=-1)
    correction, correct_frame_mask = localize_feedback_correction(
        alignment_log_probs, [1, 4, 3, 5]
    )
    assert correction.to_dict() == {
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
    teacher = torch.log_softmax(torch.randn(1, 3, 5), dim=-1)
    student = torch.log_softmax(torch.randn(1, 3, 5), dim=-1).requires_grad_()
    empty_kl = posterior_kl(
        teacher, student, frame_mask=torch.zeros((1, 3), dtype=torch.bool)
    )
    empty_kl.backward()
    assert empty_kl.item() == 0.0
    assert student.grad is not None
    blank_logits = torch.full((1, 3, 4), -4.0)
    blank_logits[:, :, 0] = 4.0
    blank_log_probs = torch.log_softmax(blank_logits, dim=-1)
    assert ctc_posterior_entropy(blank_log_probs).item() > 0
    assert (
        ctc_posterior_entropy(
            blank_log_probs, frame_selection="nonblank"
        ).item()
        == 0.0
    )
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
        feedback_correct_span_kl_enabled=True,
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
    feedback_outcome = engine.process(video, feedback_tokens=[1])
    assert feedback_outcome.update.status == "accepted"
    assert feedback_outcome.update.correction is not None
    assert feedback_outcome.to_dict()["update"]["correction"]["target_tokens"] == 1

    tent_model = MockModel()
    tent_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    tent_engine = ContinualAdaptationEngine(
        tent_model,
        tent_bank,
        EmptyReliabilityGate(),
        device="cpu",
        decoder=None,
        adaptation_objective="tent_adapter",
        entropy_frame_selection="nonblank",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    tent_outcome = tent_engine.process(video)
    assert tent_outcome.ctc_tokens == ()
    assert tent_outcome.update.status == "accepted"
    assert tent_outcome.update.supervision == "entropy"
    assert tent_outcome.update.loss_after <= tent_outcome.update.loss_before
    assert all(not parameter.requires_grad for parameter in tent_model.parameters())

    bn_video = torch.randn(12, 2, 8, 8)
    bn_model = MockBatchNormModel()
    bn_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    bn_engine = ContinualAdaptationEngine(
        bn_model,
        bn_bank,
        gate,
        device="cpu",
        decoder=None,
        adaptation_objective="bn_tent",
        parameter_update_mode="batch_norm",
        entropy_frame_selection="nonblank",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    bn_names = [name for name, _ in bn_engine._named_adaptation_parameters]
    assert bn_names == [
        "encoder.batch_norm.weight",
        "encoder.batch_norm.bias",
    ]
    assert bn_model.encoder.batch_norm.running_mean is None
    assert bn_model.encoder.batch_norm.running_var is None
    bn_before = parameter_state(bn_engine._named_adaptation_parameters)
    bn_prediction = bn_engine.predict(bn_video)
    assert torch.equal(bn_prediction.video, bn_video)
    bn_outcome = bn_engine.adapt(bn_prediction)
    assert bn_outcome.update.status == "accepted"
    assert bn_outcome.update.supervision == "entropy"
    bn_after = parameter_state(bn_engine._named_adaptation_parameters)
    assert any(not torch.equal(bn_before[name], bn_after[name]) for name in bn_before)
    assert all(
        parameter.requires_grad == (name in bn_names)
        for name, parameter in bn_model.named_parameters()
    )

    eta_model = MockBatchNormModel()
    eta_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    eta_engine = ContinualAdaptationEngine(
        eta_model,
        eta_bank,
        gate,
        device="cpu",
        decoder=None,
        adaptation_objective="eta",
        parameter_update_mode="batch_norm",
        entropy_frame_selection="nonblank",
        eta_entropy_margin=10.0,
        eta_redundancy_margin=0.05,
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    first_eta = eta_engine.process(bn_video)
    second_eta = eta_engine.process(bn_video)
    assert first_eta.update.status == "accepted"
    assert second_eta.update.status == "skipped"
    assert second_eta.update.reasons == ("eta_redundant",)
    eta_summary = eta_engine.adaptation_summary()
    assert eta_summary["eta_reliable_samples"] == 2
    assert eta_summary["eta_nonredundant_samples"] == 1
    assert eta_summary["eta_reference_initialized"]

    lora_model = MockLoRAModel()
    with torch.no_grad():
        lora_features_before, _ = lora_model.encoder(video.unsqueeze(0), None)
    lora_modules, lora_parameters = configure_attention_lora_adaptation(
        lora_model,
        rank=1,
        alpha=1.0,
    )
    with torch.no_grad():
        lora_features_after, _ = lora_model.encoder(video.unsqueeze(0), None)
    assert torch.equal(lora_features_before, lora_features_after)
    assert len(lora_modules) == 8
    assert len(lora_parameters) == 16
    assert all(
        isinstance(lora_model.get_submodule(name), LoRALinear)
        for name in lora_modules
    )
    initial_lora_model = copy.deepcopy(lora_model)
    lora_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    lora_engine = ContinualAdaptationEngine(
        lora_model,
        lora_bank,
        gate,
        device="cpu",
        decoder=None,
        adaptation_objective="online_lora",
        parameter_update_mode="lora",
        parameter_module_names=lora_modules,
        non_feedback_updates_enabled=False,
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    no_feedback_lora = lora_engine.process(video)
    assert no_feedback_lora.update.status == "skipped"
    assert no_feedback_lora.update.reasons == (
        "non_feedback_updates_disabled",
    )
    lora_before = parameter_state(named_lora_parameters(lora_model))
    lora_prediction = lora_engine.predict(video)
    lora_outcome = lora_engine.adapt(lora_prediction, feedback_tokens=[2])
    assert lora_outcome.update.status == "accepted"
    assert lora_outcome.update.supervision == "feedback"
    lora_after = parameter_state(named_lora_parameters(lora_model))
    assert any(
        not torch.equal(lora_before[name], lora_after[name]) for name in lora_before
    )
    assert lora_engine.adaptation_summary()["parameter_count"] == 128
    assert all(
        parameter.requires_grad == (".lora_" in name)
        for name, parameter in lora_model.named_parameters()
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        lora_checkpoint_path = Path(temporary_directory) / "lora_state.pt"
        save_adaptation_checkpoint(
            lora_checkpoint_path,
            lora_bank,
            {},
            2,
            optimizer_states=lora_engine.optimizer_state_dict(),
            adaptation_state=lora_engine.adaptation_state_dict(),
        )
        restored_lora_bank = ExpertBank(
            feature_dim=8,
            bottleneck_dim=4,
            max_experts=1,
            allow_growth=False,
        )
        restored_lora_checkpoint = load_adaptation_checkpoint(
            lora_checkpoint_path,
            restored_lora_bank,
            "cpu",
        )
        restored_lora_engine = ContinualAdaptationEngine(
            initial_lora_model,
            restored_lora_bank,
            gate,
            device="cpu",
            decoder=None,
            adaptation_objective="online_lora",
            parameter_update_mode="lora",
            parameter_module_names=lora_modules,
            non_feedback_updates_enabled=False,
            learning_rate=0.01,
            rollback_enabled=False,
            view_noise_std=0.0,
            temporal_mask_ratio=0.0,
        )
        restored_lora_engine.load_adaptation_state_dict(
            restored_lora_checkpoint["adaptation_state"]
        )
        restored_lora_engine.load_optimizer_state_dict(
            restored_lora_checkpoint["optimizer_states"]
        )
        restored_lora_state = parameter_state(
            named_lora_parameters(initial_lora_model)
        )
        assert all(
            torch.equal(lora_after[name], restored_lora_state[name])
            for name in lora_after
        )
        assert restored_lora_engine.optimizer_state_dict()["parameters"]["state"]

    nonfinite_model = MockBatchNormModel()
    nonfinite_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    nonfinite_engine = ContinualAdaptationEngine(
        nonfinite_model,
        nonfinite_bank,
        gate,
        device="cpu",
        decoder=None,
        adaptation_objective="bn_tent",
        parameter_update_mode="batch_norm",
        entropy_frame_selection="nonblank",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    nonfinite_prediction = nonfinite_engine.predict(bn_video)
    nonfinite_before = parameter_state(
        nonfinite_engine._named_adaptation_parameters
    )
    gradient_hook = nonfinite_engine.adaptation_parameters()[0].register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    nonfinite_outcome = nonfinite_engine.adapt(nonfinite_prediction)
    gradient_hook.remove()
    assert nonfinite_outcome.update.status == "failed"
    assert nonfinite_outcome.update.reasons == ("non_finite_gradient",)
    nonfinite_after = parameter_state(
        nonfinite_engine._named_adaptation_parameters
    )
    assert all(
        torch.equal(nonfinite_before[name], nonfinite_after[name])
        for name in nonfinite_before
    )
    assert nonfinite_engine.optimizer_state_dict()["parameters"]["state"] == {}

    local_model = MockModel()
    local_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    local_engine = ContinualAdaptationEngine(
        local_model,
        local_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_local",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    local_prediction = local_engine.predict(video)
    local_adapter_before = {
        name: value.detach().clone()
        for name, value in local_bank.experts[0].state_dict().items()
    }
    local_outcome = local_engine.adapt(
        local_prediction,
        feedback_tokens=[2],
        sample_key="local-feedback",
    )
    assert local_outcome.transcript == local_prediction.transcript
    assert any(
        not torch.equal(local_adapter_before[name], value)
        for name, value in local_bank.experts[0].state_dict().items()
    )
    assert local_outcome.update.status == "accepted"
    assert local_outcome.adaptation_expert_index == local_outcome.route.expert_index
    assert local_outcome.update.correction.substituted_tokens == 1
    assert local_outcome.update.localization["strategy"] == "ctc_error_local"
    assert not local_outcome.update.localization["randomized_support"]
    assert local_outcome.update.localization["error_target_tokens"] == 1
    assert local_outcome.update.objective_after <= local_outcome.update.objective_before

    equivalence_model = MockModel()
    equivalence_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    process_engine = ContinualAdaptationEngine(
        equivalence_model,
        equivalence_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    split_engine = ContinualAdaptationEngine(
        copy.deepcopy(equivalence_model),
        copy.deepcopy(equivalence_bank),
        copy.deepcopy(gate),
        device="cpu",
        decoder=CountingDecoder(),
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    torch.manual_seed(101)
    process_outcome = process_engine.process(video, feedback_tokens=[2])
    torch.manual_seed(101)
    split_prediction = split_engine.predict(video)
    split_outcome = split_engine.adapt(split_prediction, feedback_tokens=[2])
    assert process_outcome.to_dict() == split_outcome.to_dict()
    assert all(
        torch.equal(
            equivalence_bank.experts[0].state_dict()[name],
            split_engine.expert_bank.experts[0].state_dict()[name],
        )
        for name in equivalence_bank.experts[0].state_dict()
    )

    local_rollback_model = MockModel()
    local_rollback_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    local_rollback_engine = ContinualAdaptationEngine(
        local_rollback_model,
        local_rollback_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_local",
        learning_rate=0.1,
        rollback_enabled=True,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=0.0,
        max_target_loss_increase=100.0,
    )
    local_rollback_prediction = local_rollback_engine.predict(video)
    local_rollback_before = {
        name: value.detach().clone()
        for name, value in local_rollback_bank.experts[0].state_dict().items()
    }
    local_rollback_outcome = local_rollback_engine.adapt(
        local_rollback_prediction,
        feedback_tokens=[2],
        sample_key="local-rollback",
    )
    assert local_rollback_outcome.update.status == "rolled_back"
    assert all(
        torch.equal(local_rollback_before[name], value)
        for name, value in local_rollback_bank.experts[0].state_dict().items()
    )
    assert local_rollback_engine.optimizer_state_dict()["0"]["state"] == {}

    no_error_model = MockModel()
    no_error_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    no_error_engine = ContinualAdaptationEngine(
        no_error_model,
        no_error_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_local",
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    no_error_outcome = no_error_engine.process(
        video, feedback_tokens=[1], sample_key="already-correct"
    )
    assert no_error_outcome.update.status == "skipped"
    assert no_error_outcome.update.reasons == ("feedback_already_correct",)

    random_model = MockModel()
    random_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    random_engine = ContinualAdaptationEngine(
        random_model,
        random_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_local_random",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
    )
    random_outcome = random_engine.process(
        video, feedback_tokens=[2], sample_key="random-control"
    )
    assert random_outcome.update.status == "accepted"
    assert random_outcome.update.localization["randomized_support"]

    # --- feedback-only gate ---
    fo_model = MockModel()
    fo_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    fo_engine = ContinualAdaptationEngine(
        fo_model,
        fo_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        non_feedback_updates_enabled=False,
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    fo_pred = fo_engine.predict(video)
    fo_adapter_before = {
        name: value.detach().clone()
        for name, value in fo_bank.experts[0].state_dict().items()
    }
    fo_skip = fo_engine.adapt(fo_pred, feedback_tokens=None)
    assert fo_skip.update.status == "skipped"
    assert fo_skip.update.reasons == ("non_feedback_updates_disabled",)
    assert all(
        torch.equal(fo_adapter_before[name], value)
        for name, value in fo_bank.experts[0].state_dict().items()
    )
    assert fo_engine.optimizer_state_dict() == {}
    assert fo_bank.accepted_updates.tolist() == [0]
    fo_fb = fo_engine.adapt(fo_pred, feedback_tokens=[2], sample_key="fo-fb")
    assert fo_fb.update.status == "accepted"
    assert any(
        not torch.equal(fo_adapter_before[name], value)
        for name, value in fo_bank.experts[0].state_dict().items()
    )
    assert fo_bank.accepted_updates.tolist()[0] >= 1

    # --- hybrid vs full_sequence from identical init ---
    def _fresh_pair(strategy):
        model = MockModel()
        bank = ExpertBank(
            feature_dim=8,
            bottleneck_dim=4,
            max_experts=1,
            allow_growth=False,
        )
        eng = ContinualAdaptationEngine(
            model,
            bank,
            gate,
            device="cpu",
            decoder=CountingDecoder(),
            feedback_update_strategy=strategy,
            learning_rate=0.01,
            rollback_enabled=False,
            view_noise_std=0.0,
            temporal_mask_ratio=0.0,
            max_anchor_kl=100.0,
            max_target_loss_increase=100.0,
        )
        return eng, bank

    torch.manual_seed(202)
    full_eng, full_bank = _fresh_pair("full_sequence")
    torch.manual_seed(202)
    hybrid_eng, hybrid_bank = _fresh_pair("ctc_error_hybrid")
    # 同步初始 adapter：两侧各自 predict 创建 expert 后拷贝相同权重
    torch.manual_seed(303)
    full_pred = full_eng.predict(video)
    torch.manual_seed(303)
    hybrid_pred = hybrid_eng.predict(video)
    hybrid_bank.experts[0].load_state_dict(full_bank.experts[0].state_dict())
    full_out = full_eng.adapt(
        full_pred, feedback_tokens=[2], sample_key="hyb-vs-full"
    )
    hybrid_out = hybrid_eng.adapt(
        hybrid_pred, feedback_tokens=[2], sample_key="hyb-vs-full"
    )

    assert full_out.update.status == "accepted"
    assert hybrid_out.update.status == "accepted"
    assert hybrid_out.update.localization["strategy"] == "ctc_error_hybrid"
    assert not hybrid_out.update.localization["randomized_support"]
    assert hybrid_out.update.objective_before is not None
    assert any(
        not torch.equal(
            full_bank.experts[0].state_dict()[name],
            hybrid_bank.experts[0].state_dict()[name],
        )
        for name in full_bank.experts[0].state_dict()
    )

    # hybrid still updates when prediction already matches feedback
    hyb_correct_model = MockModel()
    hyb_correct_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    hyb_correct_engine = ContinualAdaptationEngine(
        hyb_correct_model,
        hyb_correct_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_hybrid",
        learning_rate=0.01,
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    hyb_correct_pred = hyb_correct_engine.predict(video)
    hyb_correct_before = {
        name: value.detach().clone()
        for name, value in hyb_correct_bank.experts[0].state_dict().items()
    }
    hyb_correct_out = hyb_correct_engine.adapt(
        hyb_correct_pred, feedback_tokens=[1], sample_key="hyb-already-correct"
    )
    assert hyb_correct_out.update.status == "accepted"
    assert hyb_correct_out.update.localization["error_target_tokens"] == 0
    assert any(
        not torch.equal(hyb_correct_before[name], value)
        for name, value in hyb_correct_bank.experts[0].state_dict().items()
    )

    # hybrid rollback restores adapter + AdamW
    hyb_rb_model = MockModel()
    hyb_rb_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    hyb_rb_engine = ContinualAdaptationEngine(
        hyb_rb_model,
        hyb_rb_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_hybrid",
        learning_rate=0.1,
        rollback_enabled=True,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=0.0,
        max_target_loss_increase=100.0,
    )
    hyb_rb_pred = hyb_rb_engine.predict(video)
    hyb_rb_before = {
        name: value.detach().clone()
        for name, value in hyb_rb_bank.experts[0].state_dict().items()
    }
    hyb_rb_out = hyb_rb_engine.adapt(
        hyb_rb_pred, feedback_tokens=[2], sample_key="hyb-rollback"
    )
    assert hyb_rb_out.update.status == "rolled_back"
    assert all(
        torch.equal(hyb_rb_before[name], value)
        for name, value in hyb_rb_bank.experts[0].state_dict().items()
    )
    assert hyb_rb_engine.optimizer_state_dict()["0"]["state"] == {}

    # hybrid process == predict->adapt
    hyb_eq_model = MockModel()
    hyb_eq_bank = ExpertBank(
        feature_dim=8,
        bottleneck_dim=4,
        max_experts=1,
        allow_growth=False,
    )
    hyb_process_engine = ContinualAdaptationEngine(
        hyb_eq_model,
        hyb_eq_bank,
        gate,
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_hybrid",
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    hyb_split_engine = ContinualAdaptationEngine(
        copy.deepcopy(hyb_eq_model),
        copy.deepcopy(hyb_eq_bank),
        copy.deepcopy(gate),
        device="cpu",
        decoder=CountingDecoder(),
        feedback_update_strategy="ctc_error_hybrid",
        rollback_enabled=False,
        view_noise_std=0.0,
        temporal_mask_ratio=0.0,
        max_anchor_kl=100.0,
        max_target_loss_increase=100.0,
    )
    torch.manual_seed(404)
    hyb_process_out = hyb_process_engine.process(
        video, feedback_tokens=[2], sample_key="hyb-eq"
    )
    torch.manual_seed(404)
    hyb_split_pred = hyb_split_engine.predict(video)
    hyb_split_out = hyb_split_engine.adapt(
        hyb_split_pred, feedback_tokens=[2], sample_key="hyb-eq"
    )
    assert hyb_process_out.to_dict() == hyb_split_out.to_dict()
    assert all(
        torch.equal(
            hyb_eq_bank.experts[0].state_dict()[name],
            hyb_split_engine.expert_bank.experts[0].state_dict()[name],
        )
        for name in hyb_eq_bank.experts[0].state_dict()
    )

    # random hybrid: deterministic + differs from true occupancy hybrid
    def _hybrid_state(strategy, seed_init, seed_run, sample_key):
        torch.manual_seed(seed_init)
        model = MockModel()
        bank = ExpertBank(
            feature_dim=8,
            bottleneck_dim=4,
            max_experts=1,
            allow_growth=False,
        )
        eng = ContinualAdaptationEngine(
            model,
            bank,
            gate,
            device="cpu",
            decoder=CountingDecoder(),
            feedback_update_strategy=strategy,
            feedback_random_control_seed=11,
            learning_rate=0.01,
            rollback_enabled=False,
            view_noise_std=0.0,
            temporal_mask_ratio=0.0,
            max_anchor_kl=100.0,
            max_target_loss_increase=100.0,
        )
        torch.manual_seed(seed_run)
        out = eng.process(video, feedback_tokens=[2], sample_key=sample_key)
        state = {
            name: value.detach().clone()
            for name, value in bank.experts[0].state_dict().items()
        }
        return out, state

    out_r1, state_r1 = _hybrid_state(
        "ctc_error_hybrid_random", 501, 502, "rand-hyb"
    )
    out_r2, state_r2 = _hybrid_state(
        "ctc_error_hybrid_random", 501, 502, "rand-hyb"
    )
    assert out_r1.update.status == "accepted"
    assert out_r1.update.localization["randomized_support"]
    assert out_r1.update.localization["strategy"] == "ctc_error_hybrid_random"
    assert all(torch.equal(state_r1[n], state_r2[n]) for n in state_r1)
    out_true, state_true = _hybrid_state(
        "ctc_error_hybrid", 501, 502, "rand-hyb"
    )
    assert out_true.update.status == "accepted"
    assert not out_true.update.localization["randomized_support"]
    assert any(not torch.equal(state_r1[n], state_true[n]) for n in state_r1)

    local_log_probs = torch.log_softmax(torch.randn(1, 4, 5), dim=-1)
    local_occupancy = torch.zeros(1, 4, 2)
    local_occupancy[0, 1, 0] = 0.75
    local_occupancy[0, 2, 0] = 0.25
    token_loss = occupancy_weighted_token_loss(
        local_log_probs, [1, 2], local_occupancy, [0]
    )
    blank_loss = occupancy_weighted_blank_loss(
        local_log_probs, local_occupancy[:, :, 0], blank_id=0
    )
    weighted_kl = occupancy_weighted_posterior_kl(
        local_log_probs,
        local_log_probs,
        local_occupancy[:, :, 0],
    )
    assert torch.isfinite(token_loss)
    assert torch.isfinite(blank_loss)
    assert weighted_kl.item() == 0.0

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
    feedback_route = feedback_bank.route(feedback_shifted)
    assert feedback_route.quarantined
    assert feedback_bank.expert_count == 1
    feedback_expert = feedback_bank.confirm_pending_shift(feedback_route)
    assert feedback_expert == 1
    assert feedback_bank.expert_count == 2
    assert feedback_bank.route_counts.tolist() == [2, 0]

    route_records = [
        {
            "domain": "a",
            "route": {
                "expert_index": 0,
                "created": True,
                "similarity": 1.0,
                "quarantined": False,
            },
        },
        {
            "domain": "a",
            "route": {
                "expert_index": 0,
                "created": False,
                "similarity": 0.95,
                "quarantined": False,
            },
        },
        {
            "domain": "b",
            "route": {
                "expert_index": 0,
                "created": False,
                "similarity": 0.85,
                "quarantined": True,
            },
        },
        {
            "domain": "b",
            "route": {
                "expert_index": 1,
                "created": True,
                "similarity": 0.8,
                "quarantined": False,
            },
        },
        {
            "domain": "a",
            "route": {
                "expert_index": 0,
                "created": False,
                "similarity": 0.92,
                "quarantined": False,
            },
        },
    ]
    route_summary = summarize_route_records(route_records, threshold=0.9)
    assert route_summary["samples"] == 5
    assert route_summary["similarity"]["quantiles"]["p50"] == 0.92
    assert route_summary["similarity"]["below_threshold_count"] == 2
    assert route_summary["created_count"] == 2
    assert route_summary["quarantined_count"] == 1
    assert route_summary["route_counts"] == {"0": 4, "1": 1}
    assert route_summary["route_share"] == {"0": 0.8, "1": 0.2}
    assert route_summary["domains"]["a"]["route_purity"] == 1.0
    assert route_summary["domains"]["a"]["reuse_count"] == 2
    assert route_summary["domains"]["b"]["route_purity"] == 0.5
    assert route_summary["domains"]["b"]["reuse_count"] == 1
    assert route_summary["experts"]["0"]["domain_purity"] == 0.75
    assert route_summary["domain_route_consistency"] == 0.8
    assert route_summary["clustering_purity"] == 0.8
    assert "domain_route_purity" not in route_summary

    revisit_specs = (
        ("a", 1, True, False),
        ("a", 0, False, True),
        ("b", 2, True, False),
        ("b", 2, False, False),
        ("c", 1, False, False),
        ("c", 2, False, True),
        ("a", 0, False, False),
        ("a", 3, True, False),
    )
    revisit_route_records = [
        {
            "domain": domain,
            "route": {
                "expert_index": expert_index,
                "created": created,
                "similarity": 0.95,
                "quarantined": quarantined,
            },
        }
        for domain, expert_index, created, quarantined in revisit_specs
    ]
    revisit_route_summary = summarize_route_records(
        revisit_route_records, threshold=0.9, segment_lengths=(2, 2, 2, 2)
    )
    assert revisit_route_summary["segment_lengths"] == {
        "A1": 2,
        "B": 2,
        "C": 2,
        "A2": 2,
    }
    assert revisit_route_summary["segments"]["A1"] == {
        "samples": 2,
        "route_counts": {"0": 1, "1": 1},
        "created_count": 1,
        "quarantined_count": 1,
    }
    assert revisit_route_summary["segments"]["B"]["route_counts"] == {"2": 2}
    assert revisit_route_summary["segments"]["C"]["created_count"] == 0
    assert revisit_route_summary["segments"]["C"]["quarantined_count"] == 1
    assert revisit_route_summary["segments"]["A2"]["route_counts"] == {
        "0": 1,
        "3": 1,
    }
    assert revisit_route_summary["returning_A"] == {
        "a1_dominant_expert": 0,
        "a1_dominant_share": 0.5,
        "a2_routes_to_a1_dominant_expert": 1,
        "a2_reuse_rate": 0.5,
    }
    for invalid_lengths, expected_error in (
        ((2, 2, 2), "四个正整数"),
        ((2, 2, 2, 0), "四个正整数"),
        ((2, 2, 2, 1), "恰好包含 7 条，实际为 8 条"),
    ):
        try:
            summarize_route_records(
                revisit_route_records,
                threshold=0.9,
                segment_lengths=invalid_lengths,
            )
        except ValueError as error:
            assert expected_error in str(error)
        else:
            raise AssertionError("非法回访段长必须拒绝")

    collapsed_route_summary = summarize_route_records(
        [
            {
                "domain": domain,
                "route": {
                    "expert_index": 0,
                    "created": domain == "a",
                    "similarity": 1.0 if domain == "a" else 0.95,
                    "quarantined": False,
                },
            }
            for domain in ("a", "b")
        ],
        threshold=0.9,
    )
    assert collapsed_route_summary["domain_route_consistency"] == 1.0
    assert collapsed_route_summary["clustering_purity"] == 0.5
    try:
        summarize_route_records(
            [{"domain": "a", "route": {"expert_index": 0}}], threshold=0.9
        )
    except ValueError as error:
        assert "第 1 条记录缺少字段 route.similarity" in str(error)
    else:
        raise AssertionError("路由记录缺少必需字段时必须拒绝")

    with tempfile.TemporaryDirectory() as temporary_directory:
        route_path = Path(temporary_directory) / "routes.jsonl"
        route_path.write_text(
            "\n".join(json.dumps(row) for row in route_records) + "\n",
            encoding="utf-8",
        )
        route_cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_route_records.py"),
                "--input",
                str(route_path),
                "--threshold",
                "0.9",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(route_cli.stdout) == route_summary

        revisit_route_path = Path(temporary_directory) / "revisit_routes.jsonl"
        revisit_route_path.write_text(
            "\n".join(json.dumps(row) for row in revisit_route_records) + "\n",
            encoding="utf-8",
        )
        revisit_route_cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_route_records.py"),
                "--input",
                str(revisit_route_path),
                "--revisit-segment-lengths",
                "2,2,2,2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(revisit_route_cli.stdout) == revisit_route_summary

        invalid_segments_cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_route_records.py"),
                "--input",
                str(revisit_route_path),
                "--revisit-segment-lengths",
                "2,2,2,0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid_segments_cli.returncode != 0
        assert "回访段长必须是正整数" in invalid_segments_cli.stderr

        invalid_route_path = Path(temporary_directory) / "invalid_routes.jsonl"
        invalid_route_path.write_text(
            json.dumps({"domain": "a", "route": {"expert_index": 0}}) + "\n",
            encoding="utf-8",
        )
        invalid_route_cli = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_route_records.py"),
                "--input",
                str(invalid_route_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid_route_cli.returncode != 0
        assert (
            "第 1 条记录缺少字段 route.similarity"
            in invalid_route_cli.stderr
        )

        checkpoint_path = Path(temporary_directory) / "adaptation_state.pt"
        metrics = StreamMetrics()
        metrics.update(
            "a",
            "b",
            "domain-a",
            0.5,
            "accepted",
            video_load_seconds=0.1,
            process_seconds=0.2,
            total_seconds=0.3,
        )
        timing = metrics.summary()["timing"]
        assert timing["mean_video_load_seconds"] == 0.1
        assert timing["mean_process_seconds"] == 0.2
        assert timing["mean_total_seconds"] == 0.3
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

        history_path = Path(temporary_directory) / "metrics_history.jsonl"
        append_metrics_history(history_path, {"processed_samples": 25, "cer": 0.6})
        append_metrics_history(history_path, {"processed_samples": 50, "cer": 0.5})
        append_metrics_history(history_path, {"processed_samples": 75, "cer": 0.4})
        assert prepare_metrics_history(history_path, 50) == 50
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["processed_samples"] for row in history] == [25, 50]

        best_directory = Path(temporary_directory) / "best_checkpoints"
        expected_contents = {}
        for step, score in ((100, 0.5), (200, 0.4), (300, 0.6), (400, 0.3)):
            checkpoint_path.write_text(f"checkpoint-{step}", encoding="utf-8")
            expected_contents[step] = checkpoint_path.read_text(encoding="utf-8")
            retain_best_checkpoints(
                checkpoint_path,
                best_directory,
                score=score,
                processed_samples=step,
                keep=3,
                metric_name="cer",
            )
        index = json.loads(
            (best_directory / "index.json").read_text(encoding="utf-8")
        )
        assert [entry["processed_samples"] for entry in index["checkpoints"]] == [
            400,
            200,
            100,
        ]
        assert [entry["score"] for entry in index["checkpoints"]] == [0.3, 0.4, 0.5]
        retained = sorted(best_directory.glob("*.pt"))
        assert len(retained) == 3
        assert {
            path.read_text(encoding="utf-8") for path in retained
        } == {expected_contents[step] for step in (100, 200, 400)}

        checkpoint_path.write_text("checkpoint-400-rewritten", encoding="utf-8")
        retain_best_checkpoints(
            checkpoint_path,
            best_directory,
            score=0.35,
            processed_samples=400,
            keep=3,
            metric_name="cer",
        )
        index = json.loads(
            (best_directory / "index.json").read_text(encoding="utf-8")
        )
        assert len(index["checkpoints"]) == 3
        assert sum(
            entry["processed_samples"] == 400 for entry in index["checkpoints"]
        ) == 1
        reset_best_checkpoints(best_directory)
        assert not (best_directory / "index.json").exists()
        assert not list(best_directory.glob("*.pt"))
        for step, score in ((100, 0.5), (200, 0.8), (300, 0.6)):
            checkpoint_path.write_text(f"checkpoint-{step}", encoding="utf-8")
            retain_best_checkpoints(
                checkpoint_path,
                best_directory,
                score=score,
                processed_samples=step,
                keep=2,
                metric_name="mean_reliability",
                mode="max",
            )
        index = json.loads(
            (best_directory / "index.json").read_text(encoding="utf-8")
        )
        assert [entry["processed_samples"] for entry in index["checkpoints"]] == [
            200,
            300,
        ]
        assert len(list(best_directory.glob("*.pt"))) == 2
        assert checkpoint_path.is_file()
        reset_best_checkpoints(best_directory)

        class FakeTextTransform:
            token_list = ["<blank>", "你", "好", "<eos>"]

            def post_process(self, token_ids):
                return "".join(self.token_list[token] for token in token_ids)

        manifest_path = Path(temporary_directory) / "stream.jsonl"
        stream_row = {
            "uid": "sample-1",
            "video": "sample.mp4",
            "target_tokens": [1, 2],
            "target_text": "你好",
            "domain": "speaker-1",
            "feedback": True,
        }
        manifest_path.write_text(
            json.dumps(stream_row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        items = list(
            iter_stream_manifest(manifest_path, temporary_directory, FakeTextTransform())
        )
        assert items[0].target_text == "你好"
        stream_row["target_text"] = "好你"
        manifest_path.write_text(
            json.dumps(stream_row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            list(
                iter_stream_manifest(
                    manifest_path, temporary_directory, FakeTextTransform()
                )
            )
        except ValueError as error:
            assert "反解" in str(error)
        else:
            raise AssertionError("target_text 与 token 反解不一致时必须拒绝清单")
        stream_row["target_text"] = "你好"
        stream_row["target_tokens"] = [0, 2]
        manifest_path.write_text(
            json.dumps(stream_row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            list(
                iter_stream_manifest(
                    manifest_path, temporary_directory, FakeTextTransform()
                )
            )
        except ValueError as error:
            assert "目标 token" in str(error)
        else:
            raise AssertionError("目标 token 包含 blank 时必须拒绝清单")

        sidecar_path = Path(f"{manifest_path}.meta.json")
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "label_mode": "token_passthrough",
                    "samples": 2,
                    "source_csv_sha256": "0" * 64,
                    "target_vocab_sha256": _file_sha256(DICT_PATH),
                }
            ),
            encoding="utf-8",
        )
        try:
            _stream_state(manifest_path)
        except ValueError as error:
            assert "样本数" in str(error)
        else:
            raise AssertionError("sidecar 样本数与 JSONL 不一致时必须拒绝清单")

        provenance_manifest = Path(temporary_directory) / "provenance.jsonl"
        provenance_manifest.write_text(
            json.dumps(
                {
                    "uid": "sample-1",
                    "video": "sample.mp4",
                    "target_tokens": [1],
                    "target_text": "你",
                    "domain": "speaker-1",
                    "feedback": False,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance_sidecar = Path(f"{provenance_manifest}.meta.json")
        provenance_sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "label_mode": "token_passthrough",
                    "samples": 1,
                    "source_csv_sha256": "0" * 64,
                    "target_vocab_sha256": _file_sha256(DICT_PATH),
                }
            ),
            encoding="utf-8",
        )
        saved_stream_state = _stream_state(provenance_manifest)
        provenance_sidecar.unlink()
        try:
            _stream_state(provenance_manifest)
        except ValueError as error:
            assert "sidecar" in str(error)
        else:
            raise AssertionError("正式流缺少 sidecar 时必须拒绝运行")
        current_stream_state = _stream_state(
            provenance_manifest, require_metadata=False
        )
        try:
            _validate_stream_state(saved_stream_state, current_stream_state)
        except ValueError as error:
            assert "不一致" in str(error)
        else:
            raise AssertionError("删除 sidecar 后必须拒绝恢复")

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
    speaker_pattern = re.compile(r"(?:^|/)(?P<domain>[0-9]+)_")
    assert domain_from_path("processed_test/078_sample.mp4", speaker_pattern) == "078"
    try:
        domain_from_path("processed_test/no_speaker.mp4", speaker_pattern)
    except ValueError:
        pass
    else:
        raise AssertionError("域正则不匹配时应该拒绝生成清单")

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
