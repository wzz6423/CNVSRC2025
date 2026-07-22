# SFC-VSR paper workspace

This directory is a clean paper rewrite. It reuses only the IEEE class and bibliography style from the archived manuscript; the old PhonoMotion method text and figures are not part of this draft.

## Evidence status

- Legacy exploratory matrix analysis: `data/final_analysis_ci.json`
- Legacy exploratory matrix analysis SHA-256: `ff4c3ac51e6fbd9be27f4b433ad05463d429990be6fa8bafac753ca26943f8b3`
- Validation calibration: `data/validation_route_calibration.json` (`e7c0f40afde37b2271c9c622beb97fdd6a8aadfcf4b384ffe44deba43fba4fc4`)
- Validation revisit analysis: `data/validation_revisit_analysis.json` (`973a074a67ade0a21a40699932fc58b5a905fcb30f057acb4b9592bf0bcc2362`)
- Validation revisit routing: `data/validation_expert_revisit_route_diagnostics.json` (`4e5dc625edc13b7e85c77e23b1ac1735d308ed193418f98c1093835213ab9b29`)
- Validation correct-span analysis: `data/validation_correct_span_analysis.json` (`69278ce8693b2f807017d14ef52e842a1d2f11079409fad799151fae1e21e7ff`)
- Validation adapter-only TENT analysis: `data/validation_tent_adapter_analysis.json` (`76c02987f9e74ef827376368a0f22154b49009c236af95ad508f8cead39874ca`)
- Validation error-localized analysis: `data/validation_ctc_error_local_analysis.json` (`e01f484b17852959900d5526ea592b27e2a362e86a99a2f4daf4b2b30da5ddd9`)
- Analysis code commit: `14bd2fc251f6c5c4158c670cfe3035d946dc4778`
- Error-localized adaptation code commit: `1d99eb8bb161368329c0c2f59e1f89615d1dea68`
- Next-round local evidence archive: `../experiment_archives/rsp_vsr/next_round_20260722/`
- Next-round strict no-feedback replication: seeds 7 and 123 running from commit `1d99eb8`; seed 42 queued for the next safe GPU slot.
- Next-round method protocol: feedback-only update-source decomposition plus a full-replay CTC-error hybrid on new hash-locked development/holdout streams. No candidate number is a paper result before the holdout gate passes.
- Target-dev2: previously unused Chinese-LiPS train-pool speakers `015/098/133`, 804 utterances, manifest SHA-256 `7defae9074e78d3893edc41bde1654b709cc59efaddb183531fecce20778e590`.
- Target-holdout2: disjoint train-pool speakers `046/001/093`, 778 utterances, manifest SHA-256 `29b4005205e18c8a2b1fa643d0fc9fb35bbf6266692e12a4c73ee0ad42a07651`; frozen until dev2 selects a candidate.
- Feedback-only/hybrid implementation commit: `353c47dc37351cad410139cbacbd69b5c0e0b14e`.
- Validated claim: sparse feedback with one adapter improves the frozen Chinese-LiPS stream.
- Rejected claim: the current pre-fix dynamic expert router provides material specialization or accuracy gains.
- Resolved validation gate: the calibrated router fragments, fails returning-A reuse, and is significantly worse than one adapter; no repaired-router test run is allowed.
- Resolved localization gate: target-conditioned localization beats randomized support but is significantly worse than full-sequence replay; no localized test run or tuning is allowed.
- Pending before submission: stronger matched TTA/personalization baselines beyond adapter-only TENT, a second real shift type, resource accounting, and final figures.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft also builds with `tectonic main.tex --outdir <temporary-directory>`.

Build products are not committed and must be removed after verification.
