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
