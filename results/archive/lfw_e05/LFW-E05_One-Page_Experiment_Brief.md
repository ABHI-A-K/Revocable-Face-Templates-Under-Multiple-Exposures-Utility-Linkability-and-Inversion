# LFW-E05 — One-Page Experiment Brief

**Experiment:** LFW-E05-multi-template-exposure  
**Researcher:** Abhishek A K  
**Completed (UTC):** 2026-08-21T09:15:44.137688+00:00

## Research question

Does collecting 1, 2, 5, or 10 historical 128-bit protected templates help an attacker
link a fresh-key query to the same identity?

## Controlled design

- Frozen LFW protocol: 6,000 official pairs and 10 folds.
- Frozen MobileFaceNet cache: 7682 valid embeddings of 7701 images.
- Twenty deterministic RO keys: the original five plus fifteen pre-registered extensions.
- Twenty key trials; every key serves exactly once as the unseen query key.
- Nested K=1,2,5,10 exposures and both A→B/B→A directions.
- Primary learned attacker trained on nine folds and tested on the untouched fold.
- E03-compatible direction-aware AUC and standard D_sys (omega=1, 100 bins).

## Main learned-attacker results

| Exposed templates | Mean attacker AUC [95% CI] | Worst AUC | Mean D_sys | Mean balanced accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.555823 [0.552641, 0.559134] | 0.583577 | 0.070484 | 0.538619 |
| 2 | 0.558687 [0.554715, 0.562539] | 0.586891 | 0.072386 | 0.541982 |
| 5 | 0.558138 [0.553983, 0.561802] | 0.574746 | 0.071773 | 0.541774 |
| 10 | 0.558314 [0.554261, 0.562083] | 0.584346 | 0.070509 | 0.540934 |

Mean learned-attacker AUC growth from K=1 to K=10: **0.002492**.  
Worst same-key AUC across the expanded key bank: **0.990897**.

## E05 interpretation

**CUMULATIVE_LINKABILITY_LEAKAGE_OR_VALIDITY_WEAKNESS_OBSERVED**

This label applies only to the pre-registered E05 guardrails. A pass means that this
protected-template-only, per-bit-agreement and score-combination attacker did not obtain material cumulative
linkability under the tested LFW protocol. It does not establish resistance to exposed
projection keys, reconstruction, inversion, impersonation, unlimited exposures, or
other datasets.

## Next experiment boundary

A later experiment should assume that the attacker also knows the compromised projection
matrices/keys and should directly measure embedding reconstruction and impersonation
success as exposure increases.
