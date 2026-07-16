import torch

from espnet.asr.asr_utils import add_results_to_json
from lightning import get_beam_search_decoder


class BeamDecoder:
    def __init__(self, model, token_list, beam_size=12, ctc_weight=0.3):
        self.model = model
        self.token_list = token_list
        self.search = get_beam_search_decoder(
            model,
            token_list,
            ctc_weight=float(ctc_weight),
            beam_size=int(beam_size),
        )

    @torch.inference_mode()
    def __call__(self, encoder_features):
        hypotheses = self.search(encoder_features)
        if not hypotheses:
            return "", []
        best = hypotheses[0]
        value = best.asdict()
        transcript = add_results_to_json([value], self.token_list)
        transcript = transcript.replace("▁", " ").strip().replace("<eos>", "")
        tokens = best.yseq.tolist() if hasattr(best.yseq, "tolist") else list(best.yseq)
        tokens = [
            int(token)
            for token in tokens
            if int(token) not in {self.model.sos, self.model.eos, self.model.blank}
        ]
        return transcript, tokens
