# Analysis outputs

Final paired-bootstrap intervals, pre-registered gate decisions, and resource
summaries are written here only after each source run passes row-count, UID,
hash, history-continuity, checkpoint, and error-log validation.

- `dev2_update_source_analysis.json`: verified development-only comparison of
  combined replay and feedback-only updates on the 804-clip target-dev2 stream.
  The paired overall and revisit intervals both cross zero, so this artifact
  does not establish a significant incremental benefit from pseudo updates.
- `dev2_hybrid_analysis.json`: verified development-only comparison of full
  replay, CTC-error hybrid, and randomized-support hybrid. Error-conditioned
  support significantly beats the randomized control, but does not improve
  full replay; the pre-registered candidate gate is therefore `NO-GO` and the
  frozen holdout remains unread.
- `dev2_static_corrected_analysis.json`: verified development-only analysis of
  static, full replay, feedback-only, CTC-error hybrid, and randomized support.
  Full replay improves overall CER and static-corrected revisit forgetting over
  static, while the hybrid remains non-superior to full replay; the failed
  hybrid gate is not reopened.
- `strict_nofeedback_3seed_analysis.json`: verified strict-test replication of
  no-feedback single-adapter adaptation for seeds 7, 42, and 123. Mean CER is
  0.6771309119763154 with population standard deviation 0.0017823867025910881.
- `dev3_active_feedback_analysis.json`: verified development-only six-arm
  comparison on the 758-clip target-dev3 stream. All adaptive arms improve CER
  over static, but uncertainty does not significantly outperform periodic or
  random querying and does not enrich the queried true-error rate over random.
- `dev3_gate_decision.json`: machine-readable evaluation of the five
  pre-registered active-query conditions. Only the matched 75-query budget
  passes; the final decision is `NO_GO`, so holdout2 remains frozen and unread.
  SHA-256: `03aad43ab95ded3e9c822f23c2c58a7b350079033c549ad65385505dcb13fefd`.
- `dev4_strong_baselines_analysis.json`: verified development-only five-arm
  comparison on the 700-clip target-dev4 stream. BN-TENT-VSR and ETA-VSR are
  significantly worse than static. Under the matched 70-correction budget,
  online LoRA improves over static but is significantly worse than combined
  replay in both CER and static-corrected revisit forgetting. SHA-256:
  `3fbd637e537c66590ae93a566fe6b2bfc60cc1008dccdd08ab9a6316c94742bd`.
- `dev4_decision_resources.json`: machine-readable dev4 integrity, resource,
  and promotion decision summary. The replay adapter remains the development
  incumbent; all new dev4 baselines are `NO_GO`, so holdout2 remains frozen and
  unread, with no additional seed or parameter sweep.
- `dev5_feature_film_analysis.json`: verified development-only static, replay,
  and Feature-FiLM comparison on the 681-clip target-dev5 stream. Feature-FiLM
  significantly improves over static, but is significantly worse than replay
  in overall CER and static-corrected revisit forgetting. SHA-256:
  `f7faa3f0da3e214cd15bceba92ff0a93d73ebb69ba5e05e57e6b285e58c8afc2`.
- `dev5_decision_resources.json`: machine-readable dev5 integrity, resource,
  and promotion decision summary. Feature-FiLM is a development `NO_GO`; the
  replay adapter remains the incumbent and holdout2 remains frozen and unread,
  with no additional seed or parameter sweep. SHA-256:
  `2e8fce06f0f38fab98e79b37f497ccaf59005624a807446111e7484a197fa07c`.
- `dev6_nbest_phase0a_early_stop.json`: development-only beam-10 prefix
  analysis at 356/681 samples. Oracle headroom passes at 0.03039 CER, but
  substitution coverage is 0.07871. The full replay reference proves that even
  perfect coverage of all remaining substitutions could reach at most 0.45162,
  below the pre-registered 0.55 threshold. The resulting `EARLY_NO_GO` stops
  direct small-language-model repair from the existing beam candidates without
  reading holdout2. SHA-256:
  `6d14ed43cf0e23e497c8692bab3579fa8a838dc90b2d5cb5faa6e36367daf52c`.
- `dev6_nbest_early_stop_evidence.jsonl`: compact top-10 hypotheses for the
  strictly audited 356-sample prefix. This is intentionally partial evidence,
  not a completed stream result. SHA-256:
  `0b7cfbd4bcfa8175983776cb95ae1cc4b5b77144a69288ddcad12c872107b029`.
- `dev7_counterfactual_margin_analysis.json`: verified development-only replay
  versus counterfactual beam-margin comparison on 625 samples. Candidate minus
  replay is -0.0018779 CER, 95% CI [-0.0046971, +0.0010184], and therefore
  misses both materiality and significance conditions. SHA-256:
  `1a26a147bb1ea83d192c141f7fdb2e4d58c35176f1b4a30a1cc49ddbaf5bba1b`.
- `dev7_decision_resources.json`: machine-readable integrity, local-mechanism,
  resource, and gate summary. The violation reduction is directional but only
  1.929%, so the component is `NO_GO`; holdout2 remains frozen and unread.
  SHA-256:
  `11b30c143774757bc9edf03ad2e4cd4eec7d7a7e690a34ca5b11c2d898d79bf8`.
