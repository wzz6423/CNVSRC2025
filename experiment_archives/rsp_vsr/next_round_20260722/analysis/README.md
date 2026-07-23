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
