# P1 complete-result update (2026-10-19)

P1 generation and original-algorithm scoring are complete: 24 verified videos, 24 scored rows, 2.912882 allocated GPUh within the 12 GPUh cap. Independent re-decode, source/output/metric hashes and formula/aggregate checks passed. See docs/34_Teacher_CFG开发消融结论.md and eval/diag/teacher_p1_results_20261019.json. Lower CFG is rejected as an A07 repair: imaging, last-frame/motion guards and manipulation fail. Base passes the preregistered machine screen but has imaging/semantic tradeoffs; no promotion or Teacher approval. Teacher automatic FAIL and user review pending remain unchanged.

The dated plan/code-only section below describes that earlier publication stage, not the present completion status. Private cases, media, weights, original CSV and credentials remain excluded.

# Verified Teacher experiment results

All3033 original outputs and full frame decoding, per-model WM/public VBench8 dimensions, GTref108, Physics198/66, WMB350 and fixed120 validation losses passed strict coverage verification. See [final report](docs/32_插值Teacher完整评测结论.md) for measured scores, candidate trade-offs, automatic gate failures and human review pending. This appendix does not change the parent IPVG competition hypotheses/rules/results. No claim of SOTA or accepted streaming Teacher is made. The source snapshot excludes private media, weights, credentials, identifying logs and original source Git history.

## 2026-10-19: bounded P1 plan and code increment

The six-source, two-model, CFG5/CFG3 internal diagnostic is running. This increment publishes the frozen protocol, bounded generation/scoring implementations and regression tests only; complete generation, scoring, semantic-action review and final P1 conclusions remain pending. Original/source-corrected captions are distinct. Development sources are training-visible; public source mapping remains unknown. Runtime DiffSynth package and entrypoints are copied locally and hashed. Generation is capped at 4 allocated GPUh, scoring at 8 and their total at 12. Separate temporary idle resource reservation is not experiment throughput or evidence of quality.

The prior complete interpolation evaluation remains valid: Teacher automatic conditions FAIL and user review pending. No Student, distillation, AR or new full-scale training is authorized by this increment. Private media, case manifests, original CSV, source identifiers, weights, operational state and credentials are excluded.
