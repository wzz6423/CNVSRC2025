# Dev8 Phase-0: Constrained Small-LM N-best Selection

Status: frozen before model inference.

## Question

Dev6 established a non-zero N-best oracle gap, while Dev7 showed that directly
optimizing a local beam-margin surrogate did not yield a significant CER gain.
This phase asks whether a compact Chinese language model can recover part of the
existing oracle gap by selecting one original beam candidate without generating
or editing text.

## Frozen Inputs

- Primary development stream: Dev7 replay, 625 rows, SHA-256
  `a7eff26328a8ba136e575efa5c775a17e941743297a74014adb5f4303c6f3c69`.
- Replication development prefix: Dev6 replay, 356 rows, SHA-256
  `2570f297331281b6dec5389546bad75de36d35de51c86bdc65d32903c1035922`.
- Each row must contain exactly ten finite, ordered decoder candidates. The
  original rank-1 transcript is the matched baseline.
- Targets are used only after selection for evaluation. They are never included
  in the model prompt.
- Holdout2 remains frozen and unread.

## Frozen Selector

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- License: Apache-2.0.
- Runtime: isolated environment with `transformers==4.46.3`; the VSR training
  environment is not modified.
- Prompt: `dev8-constrained-choice-v1` in
  `VSR/scripts/evaluate_llm_nbest_selector.py`.
- Decoding: greedy, `do_sample=false`, at most eight new tokens.
- The model may return only a rank from 1 through 10. It cannot rewrite,
  synthesize, or merge candidates. An unparsable answer falls back to rank 1
  and fails the integrity gate.
- There is one model, one prompt, and no interpolation weight or parameter
  search.

## Pre-registered Decision Gate

Phase-0 is `GO` only if all of the following hold on the 625-row primary Dev7
stream:

1. all rows, UIDs, and ten-candidate slates pass integrity checks;
2. every response parses to one valid rank;
3. selected-minus-rank-1 CER is at most `-0.003`;
4. the 10,000-sample paired-bootstrap 95% CI is entirely below zero.

The 356-row Dev6 prefix is a directional replication and must not contradict
the primary result. Any primary-gate failure is `NO_GO`: do not start a new
visual stream, add model sizes/prompts, tune on these targets, or read holdout2.
Passing Phase-0 freezes this selector for a later disjoint Dev8 causal run; it
does not authorize holdout2.

## Audit Boundary

The evaluator records the input SHA, exact model artifact manifest, revision,
prompt hash, raw response, parser path, selected rank, candidate edit counts,
paired bootstrap, code commit, and resumable per-row JSONL. Dev6 and Dev7 run on
separate GPUs only to reduce wall-clock time; they use the same frozen selector.
