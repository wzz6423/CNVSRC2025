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
6. Feature-FiLM on a fifth disjoint development split with speakers 071/126/045.
   Static and the 75,265-parameter replay incumbent are matched against one
   identity-initialized 1,536-parameter visual feature scale/bias candidate under
   the same 68 correction positions and pseudo-update protocol.
7. A development-only beam-10 recoverability preflight on the consumed dev5
   replay stream. It tests whether the frozen decoder candidates contain enough
   target characters to justify a grounded small-language-model repair stage.
8. Counterfactual beam-margin adaptation on a sixth disjoint development split
   with speakers 188/011/036. Matched replay and the single frozen candidate use
   the same 62 feedback positions and top-10 decoder evidence; only the candidate
   adds one hard-negative sequence-margin term at feedback updates.

Tracks 2--8 are development-only. The train pool was not used by the documented
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

All three target-dev5 arms completed and passed final integrity acceptance at
681 samples, with 28 history rows, three checkpoints, attempt 1, and zero
errors. Feature-FiLM significantly improves static by 0.3442 CER points, but is
2.8343 points worse than replay and also has significantly worse
static-corrected revisit forgetting. Its 1,536-parameter state is about 49 times
smaller than replay, but the pre-registered accuracy and forgetting conditions
fail. Feature-FiLM is therefore `NO_GO`; holdout2 remains frozen and unread,
with no extra seed or parameter sweep.

The dev6 beam-10 preflight was intentionally stopped at 356/681 samples after
the pre-registered substitution-coverage condition became mathematically
unreachable. The prefix exactly reproduces replay rank-1 predictions, query
decisions, and update states, and contains ten finite-scored hypotheses per
sample with zero structured errors. N-best oracle headroom is 0.03039 CER, but
substitution coverage is only 0.07871. Even perfect coverage on every remaining
substitution could raise final coverage to at most 0.45162, below the 0.55 gate.
This is an `EARLY_NO_GO` for direct repair from the existing beam candidates,
not a completed 681-sample run and not a test of training-time counterfactual
visual learning. No language model is downloaded or trained, and holdout2
remains frozen and unread.

Both target-dev7 arms completed 625 samples and passed final strict acceptance
at attempt 1 with 25 history rows, 62 fixed queries, three checkpoints, matching
UID/order/hashes, and zero structured errors. Replay CER is 0.7750089 and the
counterfactual candidate reaches 0.7731310. Candidate minus replay is -0.0018779
CER, paired 95% CI [-0.0046971, +0.0010184], so it misses the -0.003 materiality
threshold and the interval crosses zero. Its A2-minus-A1 forgetting difference
is -0.0069556, CI [-0.0179301, +0.0040156], whose upper bound exceeds the
allowed +0.002. All 62 feedback points have valid negatives and positive
pre-update violations, but the aggregate violation reduction is only 1.929%
(95% CI [1.712%, 2.154%]), below the required 20%. Dev7 is therefore `NO_GO`:
no additional seed, parameter sweep, or holdout2 read is performed.

## Layout

- `provenance/study_manifest.json`: initial next-round inputs and run states
  through target-dev2, plus the frozen target-dev3/holdout inputs. Later tracks
  use their dedicated audit and artifact-hash files below.
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
- `analysis/dev5_feature_film_analysis.json`: strict three-arm target-dev5
  analysis with 10,000 paired bootstrap replicates.
- `analysis/dev5_decision_resources.json`: dev5 integrity, resource, and
  promotion decision summary.
- `analysis/dev7_counterfactual_margin_analysis.json`: strict two-arm dev7
  analysis with 10,000 paired bootstrap replicates.
- `analysis/dev7_decision_resources.json`: dev7 integrity, mechanism, resource,
  and component-gate decision summary.
- `provenance/dev3_audit.json`: hash, disjointness, video-existence, and
  deterministic-regeneration audit for target-dev3.
- `provenance/dev4_audit.json`: source, manifest, stream-order, vocabulary, and
  fixed-commit audit for target-dev4.
- `provenance/audit_dev4_runs.py`: prefix/final run validator for all five
  dev4 arms.
- `provenance/audit_dev5_runs.py`: prefix/final run validator for all three
  dev5 arms.
- `provenance/dev5_final_audit.json`: accepted final target-dev5 audit output.
- `provenance/dev5_artifacts.sha256`: immutable target-dev5 analysis,
  provenance, and accepted-run hashes.
- `analysis/dev6_nbest_phase0a_early_stop.json`: machine-readable prefix
  evidence and the final-coverage reachability bound.
- `provenance/dev6_phase0a_early_stop.json`: intentional-stop state, prefix
  integrity, resource checks, input hashes, and archived-artifact hashes.
- `provenance/dev6_artifacts.sha256`: immutable hashes for the dev6 preflight,
  analysis tools, research note, launcher, and archived partial run.
- `research/dev6_llm_visual_correction_landscape.md`: primary-source review of
  LLM/VSR correction neighbors, novelty boundaries, and the staged EviCo-VSR
  falsification plan. SHA-256:
  `7211fa06e64a01c4ad5974e68da226984b48fc383b9ed5053f16325d3e6e1ff3`.
- `provenance/dev7_source.csv`, `dev7_manifest.jsonl`, and
  `dev7_manifest.meta.json`: frozen dev7 source rows and A-B-C-A stream.
- `provenance/dev7_manifest_audit.json`: deterministic speaker-selection,
  ordering, UID, feedback, and input-hash audit for dev7.
- `provenance/audit_dev7_runs.py`: prefix/final validator for both dev7 arms,
  including N-best, counterfactual arithmetic, rollback, checkpoint, and early
  viability checks.
- `provenance/dev7_final_audit.json`: accepted final target-dev7 audit output.
- `provenance/dev7_artifacts.sha256`: immutable hashes for dev7 analyses,
  provenance inputs/tools, and accepted run artifacts. Manifest SHA-256:
  `67fc4056466f1a1f72e2b79527748b219aec76ebd45c151ff936d5df64e35f1c`.
- `provenance/launch_dev7_wave.sh`: hash-locked two-GPU launcher for the sole
  matched replay and counterfactual-margin comparison.
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
