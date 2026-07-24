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
- Next-round strict no-feedback replication: seeds 7, 42, and 123 strictly accepted from commit `1d99eb8` with CER 67.9651%, 67.5844%, and 67.5897%; mean 67.7131%, population standard deviation 0.1782 points.
- Strict no-feedback three-seed analysis: `data/strict_nofeedback_3seed_analysis.json`, SHA-256 `068d6e6a0a8ac826dec64abe9dedba5446222b33c9912b154f0f648a011615d7`.
- Next-round method protocol: partial update-source comparison (feedback-only vs combined replay) plus full-replay CTC-error hybrid on hash-locked development/holdout streams; same-stream pseudo-only still required for a complete causal split. No candidate number is a confirmatory result before the holdout gate passes.
- Target-dev2: previously unused Chinese-LiPS train-pool speakers `015/098/133`, 804 utterances, A--B--C--A 154/253/243/154, manifest SHA-256 `7defae9074e78d3893edc41bde1654b709cc59efaddb183531fecce20778e590` (development-only).
- Target-holdout2: disjoint train-pool speakers `046/001/093`, 778 utterances, manifest SHA-256 `29b4005205e18c8a2b1fa643d0fc9fb35bbf6266692e12a4c73ee0ad42a07651`; confirmatory holdout frozen/unread until a single candidate is frozen.
- Target-dev3: disjoint train-pool speakers `120/176/183`, 758 utterances, A--B--C--A 135/247/242/134, manifest SHA-256 `e3ecd63cd9b07f1df82bc198f6944a52fb92a660f1b3445308c14257dfc651de` (development-only active-query gate).
- Active-query implementation and target-dev3 protocol commit: `f27a819a02346c91a0b697830ea4b8348b6e1d1f`. All six target-dev3 arms passed strict integrity checks at 758 samples, attempt 1, with 31 history rows, three checkpoints, final checkpoint count 758, correct UID/order/hashes, and zero errors.
- Target-dev3 analysis: `data/dev3_active_feedback_analysis.json`, SHA-256 `e3da525d17d01fa38e2920a42a2a33df54e98d136a1c0fb090b65876094927cd` (development-only; not test/holdout). CER is 72.7151% static, 69.7410% pseudo-only, 69.5489% feedback-only, 69.2272% combined-periodic, 68.9348% combined-random, and 68.9557% combined-uncertainty. Pseudo-only, feedback-only, and combined-periodic each significantly improve over static, while combined-periodic does not significantly improve over either single-source arm.
- Target-dev3 active-query gate: uncertainty - periodic = -0.2715 CER points, paired 95% CI [-0.6846, +0.1474]; uncertainty - random = +0.0209 points, CI [-0.4158, +0.4543]. Queried true-error-rate difference uncertainty - random = -1.333 points, CI [-4.000, 0.000]. Static-corrected forgetting difference uncertainty - periodic = -0.1139 points, CI [-1.0394, +0.7927]. Each policy spends exactly 75 queries, but every effectiveness condition fails. Gate=`NO-GO`: holdout2 remains frozen/unread, with no extra seed or sweep.
- Target-dev3 figure: `figures/active_feedback_dev3.pdf`, SHA-256 `47dfff9e736547c22a80744120778b1ecb098d0cc0da88cdbccf6836f86b82e1`; editable/source exports are retained beside it, with source-data SHA-256 `17744828f55a231ab4c655d961e575761e2ecccba685521ecbcdf70b4e24a9e9`.
- Target-dev4: disjoint train-pool speakers `128/047/202`, 700 utterances,
  A--B--C--A 120/231/230/119, manifest SHA-256
  `a3b1a5842d05ec2caf09e63eb087a6d0623770ce4edcaeca9e9d9b14eb1bf132`
  (development-only strong-baseline audit; one pre-specified seed).
- Target-dev4 implementation and protocol commit:
  `4191437d734f34cb524b049ba42415ec224a7ecb`. Static, BN-TENT-VSR, ETA-VSR,
  combined replay F10, and online LoRA-F10 each passed final integrity checks
  at 700 samples, attempt 1, with 28 history rows, three retained checkpoints,
  matching UID/order/hashes, and zero errors.
- Target-dev4 analysis: `data/dev4_strong_baselines_analysis.json`, SHA-256
  `3fbd637e537c66590ae93a566fe6b2bfc60cc1008dccdd08ab9a6316c94742bd`.
  CER is 66.8778% static, 103.6330% BN-TENT-VSR, 105.7537% ETA-VSR,
  64.8404% combined replay, and 66.0695% online LoRA. LoRA minus replay is
  +1.2291 points, paired 95% CI [+0.7371, +1.7195]; the static-corrected
  forgetting difference is +2.2901 points, CI [+0.5995, +3.9699].
- Target-dev4 decision/resources: `data/dev4_decision_resources.json`, SHA-256
  `a02e43ed2765386c304b782e3bf920666499941a6d4443bcd98308bf9950b865`.
  BN-TENT-VSR, ETA-VSR, and online LoRA-F10 are development `NO-GO`; replay is
  retained as the development incumbent and holdout2 remains frozen/unread.
- Target-dev4 figure: `figures/dev4_strong_baselines.pdf`, SHA-256
  `c2a8fb39339c0fd092e63e5eaf8eb8961ebf5e02c05a5322f94c05f5cc4ba486`;
  editable SVG and 600-dpi PNG/TIFF exports are retained beside it. Source-data
  SHA-256: `f75e895c08c73458ca1cc26410abb2b9238f8c80046cff96362e17a909a89866`.
- Target-dev5: disjoint train-pool speakers `071/126/045`, 681 utterances,
  A--B--C--A 115/229/223/114, manifest SHA-256
  `8c8e967e7076562da70d47f45883ead84c5c2dd2ef47f69412467db2e89ebf56`
  (development-only Feature-FiLM gate; one pre-specified seed).
- Target-dev5 analysis: `data/dev5_feature_film_analysis.json`, SHA-256
  `f7faa3f0da3e214cd15bceba92ff0a93d73ebb69ba5e05e57e6b285e58c8afc2`.
  Static, replay, and Feature-FiLM CER are 60.3918%, 57.2133%, and 60.0476%.
  Feature-FiLM minus replay is +2.8343 points, paired 95% CI
  [+2.2991, +3.3802]; the static-corrected forgetting difference is +3.1203
  points, CI [+1.3035, +4.9885].
- Target-dev5 decision/resources: `data/dev5_decision_resources.json`, SHA-256
  `2e8fce06f0f38fab98e79b37f497ccaf59005624a807446111e7484a197fa07c`.
  Feature-FiLM is a development `NO-GO` despite its 1,536-parameter state;
  replay remains the incumbent and holdout2 remains frozen/unread.
- Target-dev6 N-best preflight: `data/dev6_nbest_phase0a_early_stop.json`,
  SHA-256
  `6d14ed43cf0e23e497c8692bab3579fa8a838dc90b2d5cb5faa6e36367daf52c`.
  The audited 356-sample prefix has 3.0391 CER points of oracle headroom, but
  the full-stream reachability bound caps substitution coverage at 45.162%,
  below the fixed 55% gate. Direct beam-only text repair is `EARLY_NO_GO`.
- Target-dev7: disjoint train-pool speakers `188/011/036`, 625 utterances,
  A--B--C--A 107/206/206/106, manifest SHA-256
  `22e94cffece7f496219225058c4547ec038f4046eb2f794a2cf6187d299467b8`
  (development-only counterfactual beam-margin gate; one pre-specified seed).
- Target-dev7 analysis: `data/dev7_counterfactual_margin_analysis.json`, SHA-256
  `1a26a147bb1ea83d192c141f7fdb2e4d58c35176f1b4a30a1cc49ddbaf5bba1b`.
  Candidate minus replay is -0.1878 CER points, paired 95% CI
  [-0.4697, +0.1018]. The forgetting-difference interval crosses the allowed
  bound, and the local violation falls only 1.929% despite a directional paired
  interval.
- Target-dev7 decision/resources: `data/dev7_decision_resources.json`, SHA-256
  `11b30c143774757bc9edf03ad2e4cd4eec7d7a7e690a34ca5b11c2d898d79bf8`.
  Counterfactual beam margin is development `NO_GO`; no extra seed, sweep, or
  holdout2 read is performed.
- Feedback-only/hybrid implementation commit: `353c47dc37351cad410139cbacbd69b5c0e0b14e`.
- Target-dev2 machine-readable analysis: `data/dev2_update_source_analysis.json`, SHA-256 `4c4d2dbcd2de25a0f85da13afd8449c617fd9962d866388002bea821814ccb60`.
- Target-dev2 hybrid analysis: `data/dev2_hybrid_analysis.json`, SHA-256 `e81c5bbb6d812dff92987053551fcc281cb3bdb897b4f9d3a30a8277b5d6a0db`.
- Target-dev2 exploratory evidence (not test/holdout): feedback10 combined CER 60.1061% (13373/22249; 107/2/695 accepted/rolled_back/skipped); feedback-only CER 60.3218% (13421/22249; 78/2/724; 724 non-feedback samples update-forbidden). Difference feedback-only − combined = +0.2157 CER points, paired bootstrap 95% CI [-0.0644, +0.4927] points (includes 0). A2: feedback-only 0.5209448819, combined 0.5083464567. Revisit DoD difference +1.0013 points, 95% CI [-0.3415, +2.2740] (includes 0). Strict reading: partial update-source comparison only; no significant pseudo-update gain claim; no full pseudo/feedback/interaction split without same-stream pseudo-only.
- Target-dev2 CTC-error hybrid (strictly accepted, development-only; not test/holdout): full-replay baseline CER 60.1061% (13373/22249); CTC-error hybrid CER 60.4027% (13439/22249; accepted/skipped 109/695); random-support hybrid CER 64.4433% (14338/22249; accepted/rolled_back/skipped 49/53/702). hybrid − full = +0.2966 CER points, paired 95% CI [-0.1150, +0.6478] (includes 0; hybrid worse on point estimate). hybrid − random = −4.0406 points, 95% CI [-4.8926, −3.2626] (wholly below 0). A2: full 50.8346%, hybrid 50.7402%, random 56.8819%. A2−A1 forgetting difference hybrid−full = −0.7086 points, 95% CI [-3.3541, +1.0373] (includes 0); hybrid−random = −4.5257 points, 95% CI [-8.3818, −1.5038] (wholly below 0). Protocol checks: 804 UID/order, 80 feedback positions, localization strategy/auxiliary objectives, 33 history, final checkpoint=804, manifest/sidecar/base hash, .pt=3, attempt1, errors=0. Pre-registered gate required hybrid ≥0.3 points better than full with CI wholly below 0, and significantly better than random; only the random arm passed. Gate=NO-GO: no holdout2, no extra seed/sweep. Claimable: target-conditioned location signal vs matched random is real and significant. Not claimable: using it as a full-replay auxiliary improves CER or forgetting. High random rollback is an observation only (no mechanism attribution).
- Target-dev2 static + static-corrected forgetting (strictly accepted, development-only; not test/holdout): `data/dev2_static_corrected_analysis.json`, SHA-256 `0eccc9cb6913fa68170369bfb36f7bb7de43d43fc182dbc43d54b6cca54d88a3`. Static CER 61.2747% (13633/22249; 804/804 updates skipped; feedback_used all false; attempt1; 33 history; final checkpoint processed_samples=804; UID/order/hash OK; errors=0; .pt=3). Full CER 60.1061% (13373/22249). full−static = −1.1686 CER points, paired 95% CI [−1.6053, −0.7473] (wholly below 0). Segments A1/A2: static 52.6180%/54.0157%; full 51.3575%/50.8346%. Static-corrected A2−A1 forgetting (lower better): full −1.9206 points, 95% CI [−3.6183, −0.2875] (wholly below 0); feedback-only −0.9193, CI [−2.2787, +0.4746] (includes 0); hybrid −2.6292, CI [−4.8232, −0.8518] (wholly below 0); random-support +1.8965, CI [−1.0576, +5.6161] (includes 0). Hybrid−full revisit DoD remains −0.7086, CI [−3.3541, +1.0373] (includes 0); hybrid still +0.2966 worse than full on overall CER. Claimable: full replay vs static improves overall CER and static-corrected forgetting on target-dev2. Not claimable: hybrid better than full; feedback-only revisit improvement significant; random-rollback mechanism; any holdout/test conclusion. Completing static does not reverse hybrid NO-GO or reopen holdout2.
- Hybrid status: target-dev2 hybrid route stopped at NO-GO after strict acceptance; target-holdout2 remains frozen/unread.
- Exploratory / development claim (legacy streams + validated gates): sparse feedback with one adapter can improve a frozen Chinese-LiPS stream under the reported protocols; not a confirmatory holdout result.
- Rejected claim: the current pre-fix dynamic expert router provides material specialization or accuracy gains.
- Resolved validation gate: the calibrated router fragments, fails returning-A reuse, and is significantly worse than one adapter; no repaired-router test run is allowed.
- Resolved localization gate: target-conditioned localization beats randomized support but is significantly worse than full-sequence replay; no localized test run or tuning is allowed.
- Remaining evidence boundary: a future pre-registered candidate must pass
  development before any confirmatory holdout2 run, and external confirmation
  still requires a provenance-resolved second real shift. The immediate matched
  baseline audit is complete, but full EATA/CoTTA-style VSR adaptations are not
  claimed. Hybrid, active-query, BN-TENT-VSR, ETA-VSR, online LoRA-F10, and
  Feature-FiLM and counterfactual beam margin are development `NO-GO`; holdout2
  remains unread.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft also builds with `tectonic main.tex --outdir <temporary-directory>`.

Build products are not committed and must be removed after verification.
