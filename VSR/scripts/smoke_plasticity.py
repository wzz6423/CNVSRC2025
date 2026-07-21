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
    _file_sha256,
    _prepare_result_file,
    _resolve_device,
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
from plasticity.engine import ContinualAdaptationEngine
from plasticity.metrics import StreamMetrics
from plasticity.objectives import ctc_sequence_loss, ctc_target_error
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
