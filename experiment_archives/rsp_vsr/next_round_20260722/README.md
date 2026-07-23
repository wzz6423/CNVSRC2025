# RSP-VSR next-round archive (2026-07-22)

This directory mirrors the evidence needed to reproduce and audit the next
RSP-VSR experiment round. Server JSON and log files remain byte-for-byte
unchanged. Large adaptation checkpoints are retained locally but ignored by
Git; their final hashes and sizes are recorded under `provenance/`.

## Study tracks

1. Strict no-feedback three-seed replication on the fixed 3,908-clip test
   stream. Seeds 7, 42, and 123 completed from code commit `1d99eb8`; their
   verified mean CER and population standard deviation are recorded under
   `analysis/`.
2. Feedback-only ablation on a new hash-locked target-development split drawn
   from previously unused Chinese-LiPS train-pool speakers. Non-feedback samples
   perform prediction only; scheduled ground-truth corrections may update the
   single adapter.
3. CTC-error hybrid candidate on the same target-development split. Full-sequence
   feedback replay remains the primary objective and target-conditioned CTC
   error occupancy is an auxiliary term.
4. Budget-matched active feedback on a disjoint target-development split with
   speakers 120/176/183. Periodic, random, and online uncertainty queries each
   request one correction per complete ten-sample window; holdout2 remains
   unread unless the pre-registered development gate passes.
5. Strong matched baselines on a further disjoint development split with
   speakers 128/047/202. The U0 track compares static, BN-TENT-VSR, and ETA-VSR;
   the F10 track compares combined replay with online LoRA under the same 70
   correction positions and similar parameter counts.

Tracks 2--5 are development-only. The train pool was not used by the documented
source checkpoint, but this reuse is a protocol change and is not described as
the official validation split. They must not be described as test results. A
disjoint train-pool holdout is already hash-locked and remains untouched until
a pre-registered development gate selects a fixed candidate.

All six target-dev3 arms completed and passed strict integrity acceptance at
758 samples. The active-query gate is `NO_GO`: uncertainty did not meet the
material CER threshold against periodic, did not significantly beat random,
did not enrich queried true errors, and did not establish non-degraded
static-corrected forgetting. No target-holdout2 result was read, and no active
query seed expansion or policy sweep was run.

All five target-dev4 arms completed and passed final integrity acceptance at
700 samples, with 28 history rows, three checkpoints, attempt 1, and zero
errors. BN-TENT-VSR and ETA-VSR are significantly worse than static. Online
LoRA improves over static but is significantly worse than combined replay in
both CER and static-corrected revisit forgetting. The dev4 decision is therefore
to retain replay as the development incumbent and reject all three new baseline
configurations; holdout2 remains frozen and unread, with no extra seed or sweep.

## Layout

- `provenance/study_manifest.json`: immutable study inputs, server paths, code
  revisions, and run states.
- `runs/<run_name>/`: unmodified runtime metadata, logs, per-sample results,
  metric history, summaries, and adaptation checkpoints.
- `analysis/`: paired bootstrap, decision, and resource summaries generated
  only after a run passes integrity checks.
- `analysis/dev3_gate_decision.json`: five-condition target-dev3 gate outcome
  and the explicit frozen/unread holdout state.
- `analysis/dev4_strong_baselines_analysis.json`: strict five-arm dev4 analysis
  with 10,000 paired bootstrap replicates.
- `analysis/dev4_decision_resources.json`: dev4 integrity, resource, and
  promotion decision summary.
- `provenance/dev3_audit.json`: hash, disjointness, video-existence, and
  deterministic-regeneration audit for target-dev3.
- `provenance/dev4_audit.json`: source, manifest, stream-order, vocabulary, and
  fixed-commit audit for target-dev4.
- `provenance/audit_dev4_runs.py`: prefix/final run validator for all five
  dev4 arms.
- `provenance/dev4_artifacts.sha256`: immutable hashes for dev4 analyses,
  provenance inputs/tools, and accepted run artifacts; process IDs and lock
  files are intentionally excluded. Manifest SHA-256:
  `5ae2548161efc00186a34d1a430b7915a8d68fffd8ef4f9668241748250966c2`.
- `provenance/launch_dev3_wave.sh`: exact three-wave server launcher; requires
  `DEV3_CODE_COMMIT` and refuses occupied GPUs or pre-existing run directories.
- `provenance/run_artifacts.sha256`: content hashes for archived run metadata,
  logs, streams, histories, summaries, and retained checkpoints.

The 1.1-GiB source checkpoint and raw videos are not duplicated here. Their
server path and SHA-256 are recorded instead.
