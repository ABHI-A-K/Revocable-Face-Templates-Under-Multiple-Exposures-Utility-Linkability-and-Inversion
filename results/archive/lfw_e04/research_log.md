# LFW-E04-revocation research log

**Researcher:** Abhishek A K  
**Institution:** IIIT Dharwad / MIT Manipal  
**Run started (UTC):** 2026-08-18T09:02:12.874305+00:00

## Purpose

Test whether an old protected-template version can be revoked, replaced by a new keyed version without major recognition loss, and rejected if replayed.

## Hypothesis

A current-version template will retain strong same-key recognition, while a revoked old-key template will be rejected by the version gate and will rarely pass the current version's matcher threshold if metadata is forged or removed.

## Threat model

An attacker has a revoked 128-bit protected template. The attacker may replay it with
its original version metadata or may remove/forge that metadata so the current matcher
receives the old bit string.

## Pre-registered exploratory guardrails

- Worst current-version matcher-only TAR >= 0.95
- Worst old-template acceptance rate <= 0.01
- Worst absolute TAR change between old and renewed key versions <= 0.02
- Version-gate rejection rate = 1.00

## Notes

- Frozen input: official LFW protocol and E01 MobileFaceNet embeddings.
- Frozen protection: five seeded RO transforms, 128 bits, zero binarisation.
- All 20 ordered old-key -> new-key transitions are tested.
- Current-version thresholds are learned only from nine training folds.
- Revoked-template scores never influence threshold selection.
- The version gate and the matcher fallback are reported separately.
- Seeds are reproducible experimental stand-ins for keys, not production secrets.
- E04 tests one revocation event; repeated compromise remains LFW-E05.

## Completion

Run completed (UTC): 2026-08-18T09:07:07.638435+00:00.  
Conclusion: **REVOCATION_WEAKNESS_OBSERVED**.
