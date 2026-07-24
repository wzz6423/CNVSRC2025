import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from espnet.nets.beam_search import Hypothesis
from plasticity.artifacts import _write_jsonl_atomic
from plasticity.decoding import BeamDecoder
from scripts.analyze_nbest_evidence import analyze_nbest_evidence
from scripts.audit_nbest_replay import check_nbest, check_stream
from scripts.export_nbest_evidence import compact_nbest_evidence, read_jsonl


def _candidate(rank, transcript, score):
    tokens = [ord(character) for character in transcript]
    return {
        "rank": rank,
        "transcript": transcript,
        "tokens": tokens,
        "score": score,
        "normalized_score": score / max(len(tokens), 1),
        "scores": {"decoder": score},
    }


def main():
    decoder = BeamDecoder.__new__(BeamDecoder)
    decoder.model = SimpleNamespace(sos=3, eos=3, blank=0)
    decoder.token_list = ["<blank>", "你", "好", "<eos>"]
    decoder.nbest_size = 2
    decoder.search = lambda _features: [
        Hypothesis(
            yseq=torch.tensor([3, 1, 2, 3]),
            score=-1.0,
            scores={"decoder": -0.8, "ctc": -0.2},
            states={},
        ),
        Hypothesis(
            yseq=torch.tensor([3, 1, 1, 3]),
            score=-2.0,
            scores={"decoder": -1.5, "ctc": -0.5},
            states={},
        ),
    ]
    transcript, tokens, nbest = decoder.decode_with_nbest(torch.zeros(2, 4))
    assert transcript == "你好"
    assert tokens == [1, 2]
    assert [candidate["rank"] for candidate in nbest] == [1, 2]
    assert nbest[0]["scores"] == {"decoder": -0.8, "ctc": -0.2}
    assert decoder(torch.zeros(2, 4)) == ("你好", [1, 2])

    decoder.nbest_size = 0
    transcript, tokens, nbest = decoder.decode_with_nbest(torch.zeros(2, 4))
    assert (transcript, tokens, nbest) == ("你好", [1, 2], ())

    decoder.nbest_size = 2
    decoder.search = lambda _features: [
        Hypothesis(
            yseq=torch.tensor([3, 1, 2, 3]),
            score=float("nan"),
            scores={"decoder": -0.8},
            states={},
        )
    ]
    try:
        decoder.decode_with_nbest(torch.zeros(2, 4))
    except ValueError as error:
        assert "非有限" in str(error)
    else:
        raise AssertionError("非有限 decoder 分数必须被拒绝")

    stream_records = [
        {
            "index": 0,
            "uid": "u0",
            "domain": "A",
            "target": "甲乙",
            "transcript": "甲丙",
            "decoder_tokens": [ord("甲"), ord("丙")],
            "decoder_nbest": [
                _candidate(1, "甲丙", -1.0),
                _candidate(2, "甲乙", -2.0),
            ],
        },
        {
            "index": 1,
            "uid": "u1",
            "domain": "B",
            "target": "丁戊",
            "transcript": "丁戊",
            "decoder_tokens": [ord("丁"), ord("戊")],
            "decoder_nbest": [
                _candidate(1, "丁戊", -1.0),
                _candidate(2, "丁己", -2.0),
            ],
        },
        {
            "index": 2,
            "uid": "u2",
            "domain": "C",
            "target": "己庚",
            "transcript": "己",
            "decoder_tokens": [ord("己")],
            "decoder_nbest": [
                _candidate(1, "己", -1.0),
                _candidate(2, "己庚", -2.0),
            ],
        },
    ]
    for row in stream_records:
        row.update(
            {
                "ctc_tokens": row["decoder_tokens"],
                "feedback_used": False,
                "feedback_query": {"queried": False},
                "adaptation_expert_index": 0,
                "update": {
                    "status": "skipped",
                    "supervision": "pseudo",
                    "reasons": ["test"],
                },
            }
        )
    compact = compact_nbest_evidence(stream_records, top_k=2)
    for row in stream_records:
        check_nbest(row, top_k=2)
    edits, characters = check_stream(
        stream_records,
        stream_records,
        [{"uid": row["uid"]} for row in stream_records],
        top_k=2,
        expected_samples=len(stream_records),
        final=True,
    )
    assert edits == 2
    assert characters == 6
    analysis = analyze_nbest_evidence(
        compact,
        top_k=2,
        min_oracle_headroom=0.02,
        min_substitution_coverage=0.55,
    )
    assert analysis["one_best"]["edits"] == 2
    assert analysis["nbest_oracle"]["edits"] == 0
    assert analysis["error_position_coverage"]["substitution_coverage_at_k"] == 1.0
    assert analysis["error_position_coverage"]["deletion_coverage_at_k"] == 1.0
    assert analysis["identity_risk"]["incorrect_alternative_rate"] == 1.0
    assert analysis["phase0a_gate"]["decision"] == "BEAM_GO"

    invalid = [dict(stream_records[0])]
    invalid[0]["decoder_nbest"] = [
        _candidate(1, "错误", -1.0),
        _candidate(2, "甲乙", -2.0),
    ]
    try:
        compact_nbest_evidence(invalid, top_k=2)
    except ValueError as error:
        assert "rank-1" in str(error)
    else:
        raise AssertionError("rank-1 不一致时必须拒绝导出")

    invalid_tokens = [dict(stream_records[0])]
    invalid_tokens[0]["decoder_tokens"] = [99]
    try:
        compact_nbest_evidence(invalid_tokens, top_k=2)
    except ValueError as error:
        assert "decoder_tokens" in str(error)
    else:
        raise AssertionError("rank-1 token 不一致时必须拒绝导出")

    incomplete = analyze_nbest_evidence(
        [dict(compact[0], nbest=compact[0]["nbest"][:1])],
        top_k=2,
        min_oracle_headroom=0.0,
        min_substitution_coverage=0.0,
    )
    assert incomplete["phase0a_gate"]["complete_top_k_pass"] is False
    assert incomplete["phase0a_gate"]["decision"] == "NO_GO"

    with tempfile.TemporaryDirectory() as temporary_directory:
        stream_path = Path(temporary_directory) / "stream_results.jsonl"
        evidence_path = Path(temporary_directory) / "evidence.jsonl"
        analysis_path = Path(temporary_directory) / "analysis.json"
        _write_jsonl_atomic(stream_path, stream_records)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_nbest_evidence.py"),
                "--stream-results",
                str(stream_path),
                "--output",
                str(evidence_path),
                "--top-k",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_nbest_evidence.py"),
                "--evidence",
                str(evidence_path),
                "--output",
                str(analysis_path),
                "--top-k",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        restored = read_jsonl(evidence_path)
        assert restored == compact
        restored_analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert restored_analysis["phase0a_gate"]["decision"] == "BEAM_GO"
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            json.loads(line)

    print("RSP-VSR N-best evidence smoke 通过")


if __name__ == "__main__":
    main()
