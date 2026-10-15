# CFP-FP-E01-R4 — One-Page Experiment Brief

**Experiment:** CFP-FP-E01-R4-aligned-baseline  
**Completed (UTC):** 2026-09-03T06:29:33.761031+00:00

## Research question

How well does the frozen 512-D MobileFaceNet verify frontal-to-profile pairs when supplied with the standard aligned 112×112 CFP-FP crops?

## Why R4 was necessary

- Raw R1 coverage: 5.730%
- R2 recovery coverage: 47.660%
- R3 multi-scale/tiled coverage: 49.700%
- R3 remained below the frozen 90% validity gate, so no R3 recognition metrics were allowed.

## Controlled design

- Standard aligned package checksum: `76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370`
- 7,000 pairs, 10 contiguous folds, 350 genuine and 350 impostor pairs per fold.
- Frozen `w600k_mbf` MobileFaceNet, 512-D L2-normalized embedding, cosine similarity.
- No flip augmentation.
- Every threshold learned from nine folds and tested on the held-out fold.
- Complete 14,000/14,000 pair-side embedding coverage required.

## Main results

| Metric | Aligned benchmark |
|---|---:|
| Evaluated pairs | 7,000 |
| Mean fold accuracy | 95.771% |
| Pooled ROC AUC | 97.379% |
| Pooled EER | 5.514% |
| TAR at training FAR ≤ 0.1% | 87.714% |
| Observed pooled FAR | 0.143% |
| FRR | 12.286% |

## Interpretation boundary

R4 measures **recognition under pose variation after standard alignment**. R3 separately measures **raw preprocessing transfer**. These numbers must not be merged or labelled as the same pipeline.

The cache can support pair-based protected-template utility and unlinkability experiments. Before E05-ID or identity-grouped E07, a separately verified identity-to-aligned-image mapping is required; the notebook does not infer identities from pair labels.
