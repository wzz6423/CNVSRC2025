# SFC-VSR paper workspace

This directory is a clean paper rewrite. It reuses only the IEEE class and bibliography style from the archived manuscript; the old PhonoMotion method text and figures are not part of this draft.

## Evidence status

- Bound analysis: `data/final_analysis_ci.json`
- Bound analysis SHA-256: `ff4c3ac51e6fbd9be27f4b433ad05463d429990be6fa8bafac753ca26943f8b3`
- Validation calibration: `data/validation_route_calibration.json` (`e7c0f40afde37b2271c9c622beb97fdd6a8aadfcf4b384ffe44deba43fba4fc4`)
- Validation revisit analysis: `data/validation_revisit_analysis.json` (`973a074a67ade0a21a40699932fc58b5a905fcb30f057acb4b9592bf0bcc2362`)
- Validation revisit routing: `data/validation_expert_revisit_route_diagnostics.json` (`4e5dc625edc13b7e85c77e23b1ac1735d308ed193418f98c1093835213ab9b29`)
- Analysis code commit: `14bd2fc251f6c5c4158c670cfe3035d946dc4778`
- Validated claim: sparse feedback with one adapter improves the frozen Chinese-LiPS stream.
- Rejected claim: the current dynamic expert router provides material specialization or accuracy gains.
- Resolved validation gate: the calibrated router fragments, fails returning-A reuse, and is significantly worse than one adapter; no repaired-router test run is allowed.
- Pending before submission: matched TTA/personalization baselines, a second real shift type, resource accounting, and final figures.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft also builds with `tectonic main.tex --outdir <temporary-directory>`.

Build products are not committed and must be removed after verification.
