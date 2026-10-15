# LFW-E05-multi-template-exposure research log

**Researcher:** Abhishek A K  
**Institution:** IIIT Dharwad / MIT Manipal  
**Run started (UTC):** 2026-08-21T09:12:55.372611+00:00

## Purpose

Measure whether collecting 1, 2, 5, or 10 differently keyed protected templates improves cross-key identity linkage against a fresh-key query.

## Hypothesis

Increasing the number of exposed independently keyed 128-bit templates will not materially increase a leakage-free attacker's AUC, D_sys, or balanced accuracy.

## Threat model

For each official LFW pair, an attacker receives K historical 128-bit templates generated
from one image under K distinct keys. A second image is protected under a fresh key. The
attacker uses only per-bit agreement and cross-key Hamming-score evidence to decide same
versus different identity. Secret projection matrices, raw embeddings, and face images
are excluded from the attack features.

## Pre-registered exploratory guardrails

- Worst same-key AUC across the 20-key validity check >= 0.98
- Mean learned-attacker AUC at every K <= 0.55
- Worst learned-attacker AUC over any trial/direction/K <= 0.60
- Mean learned-attacker D_sys at K=10 <= 0.10
- Mean learned-attacker AUC growth from K=1 to K=10 <= 0.02
- Mean learned-attacker balanced accuracy at K=10 <= 0.55

## Notes

- Frozen input: official LFW pairs and E01 MobileFaceNet embedding cache.
- Frozen protection: seeded Gaussian RO, reduced QR, 128 bits, threshold at zero.
- The original five E02–E04 seeds are preserved; fifteen keys are added before scoring.
- Each of the 20 keys serves once as a fresh query key; exposed keys never include it.
- Exposure sets are nested at K=1,2,5,10 for paired growth analysis.
- Both A→B and B→A image directions are evaluated.
- A leakage-free logistic attacker is trained on nine folds and tested on the tenth.
- Attacker inputs are 128 per-bit agreement rates, sorted cross-key scores, and
  order-invariant summary statistics.
- The attack does not use raw embeddings, face images, RO matrices, or secret seeds.
- E05 measures operational cumulative linkability, not information-theoretic maximal leakage.
- Key-assisted reconstruction and impersonation remain reserved for later experiments.

## Completion

Run completed (UTC): 2026-08-21T09:15:44.137688+00:00.  
Conclusion: **CUMULATIVE_LINKABILITY_LEAKAGE_OR_VALIDITY_WEAKNESS_OBSERVED**.
