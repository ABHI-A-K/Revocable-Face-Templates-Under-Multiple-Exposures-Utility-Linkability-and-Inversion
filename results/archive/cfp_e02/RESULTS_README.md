# CFP-FP-E02 — One-Page Experiment Brief

**Experiment:** CFP-FP-E02-seeded-RO-128bit  
**Completed (UTC):** 2026-09-03T09:33:47.217362+00:00

## Research question

Can frozen 512-D MobileFaceNet embeddings be converted into compact, key-controlled 128-bit templates while preserving frontal-to-profile verification utility and reducing cross-key linkability?

## Controlled design

- Reused the verified E01-R4 aligned cache: 14,000 pair sides, 7,000 pairs, 5,890 unique encoded images.
- Ten official contiguous folds; each fold has 350 genuine and 350 impostor pairs.
- Ten independent seeded Random Orthogonal projections, 512-D to 128 values, followed by sign binarization.
- Same-key Hamming similarity for verification.
- Every threshold learned from nine folds and evaluated on the held-out fold.
- No SCRFD, image decoding, re-alignment, embedding re-extraction, or model retraining.

## Main results

| Metric | E01-R4 raw | E02 protected mean across keys | 95% CI across keys |
|---|---:|---:|---:|
| Mean fold accuracy | 95.771% | 87.306% | 87.094%–87.517% |
| Pooled AUC | 97.379% | 93.615% | 93.450%–93.780% |
| Pooled EER | 5.514% | 13.028% | 12.748%–13.308% |
| TAR at training FAR 0.1% | 87.714% | 35.929% | 32.536%–39.322% |

## Cross-key unlinkability

- Key pairs evaluated: 45
- Mean same-image cross-key Hamming similarity: 0.5001
- Mean different-image cross-key Hamming similarity: 0.5000
- Pooled attacker-oriented linkability AUC: 0.5010
- Worst key-pair attacker-oriented AUC: 0.5367

Values near 0.5 for cross-key Hamming similarity and linkability AUC indicate that changing the key makes same-image templates difficult to distinguish from unrelated-image templates.

## Compactness

- Raw float32 embedding: 2,048 bytes
- Packed protected template: 16 bytes
- Storage reduction: 128×

## Interpretation boundary

This experiment establishes protected-template **utility and same-image cross-key unlinkability** on the aligned CFP-FP protocol. It does not establish cryptographic irreversibility, resistance to key exposure, reconstruction resistance, or identity-disjoint attack performance.


## Run notes

Fresh CFP-FP-E02 run using the verified CFP-FP-E01-R4 cache.
