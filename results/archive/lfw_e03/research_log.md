# LFW-E03-cross-key-unlinkability research log

**Researcher:** Abhishek A K  
**Institution:** IIIT Dharwad / MIT Manipal  
**Run started (UTC):** 2026-08-13T06:48:11.397479+00:00

## Purpose

Measure whether same-identity templates protected under different seeded RO keys can be distinguished from different-identity cross-key templates.

## Hypothesis

Cross-key mated and non-mated Hamming similarities will overlap near 0.50, giving a direction-aware attacker AUC and linkage accuracy near chance.

## Pre-registered exploratory guardrails

- Worst key-pair attacker AUC <= 0.55
- Worst key-pair standard D_sys <= 0.10
- Leakage-free out-of-fold balanced linkage accuracy <= 0.55
- Worst same-image attacker AUC <= 0.55

## Notes

- Frozen input: LFW-E01 MobileFaceNet embeddings and LFW official pair protocol.
- Frozen protection: five seeded RO transforms, 128 bits, threshold at zero.
- Controlled E03 change: compare protected templates produced under different keys.
- All 10 unordered key pairs and both image/key orientations are evaluated.
- Detection failures are excluded because no protected template exists to link.
- Same-image cross-key comparison is reported separately as a stronger diagnostic.
- E03 does not claim revocation, repeated-exposure, inversion, or impersonation security.

## Completion

Run completed (UTC): 2026-08-13T06:52:30.963614+00:00.  
Conclusion: **SUPPORTED_WITHIN_E03_GUARDRAILS**.
