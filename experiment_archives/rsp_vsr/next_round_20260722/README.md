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

The latter two tracks are development candidates. The train pool was not used
by the documented source checkpoint, but this reuse is a protocol change and is
not described as the official validation split. They must not be described as
test results. A disjoint train-pool holdout is already hash-locked and remains
untouched until the pre-registered development gate selects a fixed candidate.

All six target-dev3 arms completed and passed strict integrity acceptance at
758 samples. The active-query gate is `NO_GO`: uncertainty did not meet the
material CER threshold against periodic, did not significantly beat random,
did not enrich queried true errors, and did not establish non-degraded
static-corrected forgetting. No target-holdout2 result was read, and no active
query seed expansion or policy sweep was run.

## Layout

- `provenance/study_manifest.json`: immutable study inputs, server paths, code
  revisions, and run states.
- `runs/<run_name>/`: unmodified runtime metadata, logs, per-sample results,
  metric history, summaries, and adaptation checkpoints.
- `analysis/`: paired bootstrap, decision, and resource summaries generated
  only after a run passes integrity checks.
- `analysis/dev3_gate_decision.json`: five-condition target-dev3 gate outcome
  and the explicit frozen/unread holdout state.
- `provenance/dev3_audit.json`: hash, disjointness, video-existence, and
  deterministic-regeneration audit for target-dev3.
- `provenance/launch_dev3_wave.sh`: exact three-wave server launcher; requires
  `DEV3_CODE_COMMIT` and refuses occupied GPUs or pre-existing run directories.
- `provenance/run_artifacts.sha256`: content hashes for archived run metadata,
  logs, streams, histories, summaries, and retained checkpoints.

The 1.1-GiB source checkpoint and raw videos are not duplicated here. Their
server path and SHA-256 are recorded instead.
