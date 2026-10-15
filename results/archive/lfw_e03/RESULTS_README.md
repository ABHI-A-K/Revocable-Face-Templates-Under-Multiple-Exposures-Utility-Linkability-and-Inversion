# LFW-E03-cross-key-unlinkability results

# LFW-E03 — One-Page Experiment Brief

**Experiment:** LFW-E03-cross-key-unlinkability  
**Researcher:** Abhishek A K  
**Completed (UTC):** 2026-08-13T06:52:30.963614+00:00

## Research question

Can an attacker decide whether two 128-bit protected face templates generated under different seeded RO keys belong to the same identity?

## Controlled design

- Frozen LFW deep-funnelled protocol: 6,000 pairs and 10 folds.
- Frozen 512-D MobileFaceNet embeddings: 7682 valid of 7701 images.
- Five fixed RO keys produce 10 unordered key pairs.
- Both key/image orientations are scored: 119,300 matcher-only comparisons.
- Established global metric is D_sys (omega=1, 100 bins); ideal is 0.
- Complementary attacker metric is direction-aware AUC: `max(AUC, 1-AUC)`; ideal is 0.50.
- A 10-fold attacker learns direction and threshold from nine folds and is tested on the held-out fold.

## Main results

| Measure | Result | Ideal |
|---|---:|---:|
| Pooled standard D_sys | 0.006490 | 0.000000 |
| Pooled attacker AUC | 0.502024 | 0.500000 |
| Pooled AUC linkability index | 0.004048 | 0.000000 |
| Worst key-pair attacker AUC | 0.525608 | 0.500000 |
| Worst key-pair D_sys | 0.026861 | 0.000000 |
| Worst absolute mated/non-mated mean gap | 0.003889 | 0.000000 |
| Out-of-fold balanced linkage accuracy | 0.501590 | 0.500000 |
| Worst same-image attacker AUC | 0.533152 | 0.500000 |

Cross-key mated similarity: 0.500011 ± 0.044330.  
Cross-key non-mated similarity: 0.499678 ± 0.044139.

## E03 interpretation

**SUPPORTED_WITHIN_E03_GUARDRAILS**

This label applies only to the pre-registered E03 guardrails. It is not a proof of universal unlinkability. Security interpretation must include the worst key pair and same-image diagnostic, not only pooled AUC.

## Next experiment boundary

LFW-E04 should test revocation operationally: invalidate an old key version, issue a new version, confirm legitimate verification under the new version, and confirm that the revoked version is rejected. Repeated compromise remains reserved for LFW-E05.


## Interpretation rules

- A raw AUC below 0.5 is reversible; always use direction-aware attacker AUC.
- Detection failures are omitted from unlinkability scoring because no protected template exists.
- Per-key-pair results are primary for weakness discovery; pooled values are supporting summaries.
- Threshold attack results are leakage-free across the official folds.
- Do not use E03 to claim revocation, repeated-compromise, inversion, or impersonation resistance.

## Editable run notes

- Frozen input: LFW-E01 MobileFaceNet embeddings and LFW official pair protocol.
- Frozen protection: five seeded RO transforms, 128 bits, threshold at zero.
- Controlled E03 change: compare protected templates produced under different keys.
- All 10 unordered key pairs and both image/key orientations are evaluated.
- Detection failures are excluded because no protected template exists to link.
- Same-image cross-key comparison is reported separately as a stronger diagnostic.
- E03 does not claim revocation, repeated-exposure, inversion, or impersonation security.
