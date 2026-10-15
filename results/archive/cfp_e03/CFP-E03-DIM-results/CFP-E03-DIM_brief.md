# CFP-E03-DIM results

128-bit control: all 70,000 E02 pair scores reproduced exactly.

| Bits | Bytes | TAR (%) | Observed FAR (%) | EER (%) | TAR change vs 128 (pp) | Worst cross-key AUC |
|---|---|---|---|---|---|---|
| 64 | 8 | 14.563 | 0.077 | 18.863 | -21.366 | 0.5341 |
| 128 | 16 | 35.929 | 0.103 | 13.028 | +0.000 | 0.5367 |
| 256 | 32 | 61.494 | 0.129 | 8.997 | +25.566 | 0.5342 |
| 512 | 64 | 80.503 | 0.106 | 6.731 | +44.574 | 0.5454 |

Raw E01-R4 TAR: 87.714%; observed FAR: 0.143%.

TAR is measured using thresholds learned on nine folds; actual test FAR can exceed the 0.1% training target.
All dimensions are reported. No final deployment dimension is selected by this notebook.
Intervals in dimension_summary.csv describe key variability, not uncertainty over new identities or datasets.
Cross-key scores compare identical encoded images against different encoded images. Different images are not verified different people.
Length-specific projections are not nested. At 512 bits sign binarization still discards information.
This is an ablation supporting method design, not a standalone novelty or security claim.