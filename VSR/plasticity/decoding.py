import math

import torch

from espnet.asr.asr_utils import add_results_to_json
from lightning import get_beam_search_decoder


class BeamDecoder:
    def __init__(
        self,
        model,
        token_list,
        beam_size=12,
        ctc_weight=0.3,
        nbest_size=0,
    ):
        self.model = model
        self.token_list = token_list
        self.nbest_size = int(nbest_size)
        if self.nbest_size < 0:
            raise ValueError("nbest_size 不能为负数")
        if self.nbest_size > int(beam_size):
            raise ValueError("nbest_size 不能大于 beam_size")
        self.search = get_beam_search_decoder(
            model,
            token_list,
            ctc_weight=float(ctc_weight),
            beam_size=int(beam_size),
        )

    def _format_hypothesis(self, hypothesis, rank):
        value = hypothesis.asdict()
        transcript = add_results_to_json([value], self.token_list)
        transcript = transcript.replace("▁", " ").strip().replace("<eos>", "")
        yseq = hypothesis.yseq
        tokens = yseq.tolist() if hasattr(yseq, "tolist") else list(yseq)
        tokens = [
            int(token)
            for token in tokens
            if int(token) not in {self.model.sos, self.model.eos, self.model.blank}
        ]
        score = float(value["score"])
        scores = {
            str(name): float(component)
            for name, component in value["scores"].items()
        }
        if not math.isfinite(score) or any(
            not math.isfinite(component) for component in scores.values()
        ):
            raise ValueError("decoder N-best 包含非有限分数")
        return {
            "rank": int(rank),
            "transcript": transcript,
            "tokens": tokens,
            "score": score,
            "normalized_score": score / max(len(tokens), 1),
            "scores": scores,
        }

    @torch.inference_mode()
    def decode_with_nbest(self, encoder_features):
        hypotheses = self.search(encoder_features)
        if not hypotheses:
            return "", [], ()
        limit = max(1, self.nbest_size)
        records = tuple(
            self._format_hypothesis(hypothesis, rank)
            for rank, hypothesis in enumerate(hypotheses[:limit], start=1)
        )
        exported = records if self.nbest_size else ()
        return records[0]["transcript"], list(records[0]["tokens"]), exported

    @torch.inference_mode()
    def __call__(self, encoder_features):
        transcript, tokens, _ = self.decode_with_nbest(encoder_features)
        return transcript, tokens
