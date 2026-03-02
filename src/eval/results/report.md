# Evaluation Summary

Generated at (UTC): `2026-03-02T17:57:11.087856+00:00`

## Table 1. Local Optimality Gap Results vs Baseline or Reference

| Task | Dataset | Optimality gap | Gap reference | Local result | Baseline / Reference | Baseline gap | Reference gap | Absolute delta | Relative improvement |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| TSP | test100+test200 | optimality gap (%) | optimal tour | 2.767855 | paper best single program | - | 2.446000 | 0.321855 | - |
| BP | OR3_val | optimality gap (%) | L1 lower bound | 1.984426 | best-fit | 5.048983 | - | 3.064557 | 60.70% |
| FP | test_flatpack_dynamic_0_seed | optimality gap (fraction) | full occupancy | 0.092607 | all_equal | 0.190074 | - | 0.097467 | 51.28% |
| FP | test_flatpack_dynamic_0_seed | optimality gap (fraction) | full occupancy | 0.092607 | heuristic_flatpack | 0.140680 | - | 0.048073 | 34.17% |

## Table 2. Comparison with Paper-Reported Optimality Gaps

| Task | Paper optimality gap | Gap reference | Paper human-designed | Paper EvoTune | Paper FunSearch | Local reproduction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TSP | optimality gap (%) | optimal tour | - | 2.446000 (best single program) | - | 2.767855 |
| BP | optimality gap (%) | L1 lower bound | 5.37 | 2.06 | 2.96 | 1.984426 |
| FP | optimality gap (fraction) | full occupancy | 0.1092 | 0.0829 | 0.0898 | 0.092607 |

## Notes

- All reported results are expressed as optimality gap. Lower is better.
- TSP optimality gap is measured against the optimal tour.
- BP optimality gap is measured against the L1 lower bound used by the original repository and paper.
- FP optimality gap is defined as `1 - occupied_proportion`, i.e. gap to full occupancy.
- BP local result compares the extracted `best_priority` against the `best-fit` baseline.
- FP local result compares the extracted `best_priority` against both `all_equal` and `heuristic_flatpack` baselines.
