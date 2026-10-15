# LFW-E06-PROJ — One-Page Experiment Brief

**Experiment:** LFW-E06-PROJ-projection-comparison  
**Researcher:** Abhishek A K  
**Completed (UTC):** 2026-08-26T13:42:47.653218+00:00

## Research question

Under identical 128-bit templates, keys, identity-disjoint folds, and multi-exposure
attacker conditions, does SRHT or PCA-guided RO offer a better
recognition–privacy–efficiency balance than RO?

## Controlled design

- Frozen MobileFaceNet cache: 7,682 valid of 7,701 images.
- PCA calibration: 1,024 identities and 1,024 images.
- Calibration/evaluation identity overlap: **0**.
- Evaluation: 6,000 balanced pairs in 10 identity-disjoint folds.
- Methods: RO, SRHT/Hadamard, PCA(512→256)+keyed RO(256→128).
- Twenty keys, two image directions, nested K=1,2,5,10 exposures.

## Main results

| Method | TAR@FAR 0.1% | EER | Attacker AUC K=1 | Attacker AUC K=10 | D_sys K=10 | Bit entropy | CPU p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| RO | 0.8197 | 0.0335 | 0.5555 | 0.5594 | 0.0724 | 0.9969 | 0.0281 |
| SRHT/Hadamard | 0.8240 | 0.0336 | 0.5609 | 0.5577 | 0.0713 | 0.9969 | 0.1125 |
| PCA-guided RO | 0.8603 | 0.0279 | 0.5690 | 0.5687 | 0.0782 | 0.9992 | 0.0472 |

## Controlled interpretation

- Highest recognition TAR: **PCA-guided RO**.
- Lowest K=10 attacker AUC: **SRHT/Hadamard**.
- Fastest CPU template generation: **RO**.
- Non-RO methods passing the predeclared screen: **SRHT/Hadamard, PCA-guided RO**.

The screen requires TAR within 0.5 percentage points of RO and K=10 attacker AUC no
more than 0.01 above RO. A projection should be called better only when paired
differences, key stability, privacy, and efficiency jointly support it.

## Scope

This result covers LFW, 20 deterministic keys, 128-bit templates, and a
protected-template-only identity-disjoint attacker. It does not cover compromised keys,
reconstruction, impersonation, cross-dataset validation, or physical-device energy.
