# LFW-E04 — One-Page Experiment Brief

**Experiment:** LFW-E04-revocation  
**Researcher:** Abhishek A K  
**Completed (UTC):** 2026-08-18T09:07:07.638435+00:00

## Research question

After a protected template is compromised and revoked, does a newly keyed 128-bit
template preserve legitimate recognition while the old template is rejected?

## Controlled design

- Frozen LFW protocol: 6,000 official pairs and 10 folds.
- Frozen 512-D MobileFaceNet embeddings: 7682 valid of 7701 images.
- Five fixed seeded RO keys; all 20 ordered old->new transitions.
- Current thresholds learned from nine folds at target training FAR 1e-03.
- Revoked scores never used for threshold calibration.
- Two layers: explicit version gate and matcher fallback after metadata bypass.

## Main results

| Measure | Result | E04 guardrail |
|---|---:|---:|
| Mean current-version matcher-only TAR | 82.992% | Descriptive |
| Worst current-version matcher-only TAR | 79.504% | >= 95.000% |
| Pooled revoked-template acceptance rate | 0.047% | Descriptive |
| Worst transition revoked-template acceptance rate | 0.101% | <= 1.000% |
| Pooled explicit version-gate rejection rate | 100.000% | = 100.000% |
| Worst absolute TAR change after renewal | 5.636% | <= 2.000% |

## E04 interpretation

**REVOCATION_WEAKNESS_OBSERVED**

This label applies only to the pre-registered E04 guardrails. The explicit version gate
proves the software invalidation rule worked; matcher-fallback RTAR measures old-bit-string
reuse after metadata bypass. Neither result proves unlimited revocability or resistance to
multiple-template compromise.

## Next experiment boundary

LFW-E05 should expose 1, 2, 5, and 10 protected templates for the same identity and test
whether cumulative exposure improves linkage, inference, or impersonation attacks.
