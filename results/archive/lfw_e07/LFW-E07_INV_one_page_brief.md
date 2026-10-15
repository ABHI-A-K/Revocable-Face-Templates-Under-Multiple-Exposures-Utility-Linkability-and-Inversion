# LFW-E07_INV — One-Page Experiment Brief

**Experiment:** LFW-E07_INV-embedding-inversion
**Researcher:** Abhishek A K
**Completed (UTC):** 2026-08-27T06:24:07.357213+00:00

## Research question

Can 1/2/5/10 protected templates and their known projection operators be inverted into an embedding that impersonates the target under a fresh key?

## Controlled design

- Frozen LFW cache: 7682 valid 512-D MobileFaceNet embeddings.
- Methods: RO, SRHT, PCA_GUIDED_RO; 128-bit templates; 20 reproducible keys.
- Targets: 1200 identities, two different images per identity, balanced across ten identity-disjoint folds.
- Key trials: 10; exposures K = [1, 2, 5, 10].
- Primary endpoint: cross-image fresh-key embedding-level ASR at a threshold trained for FAR = 0.1%.
- Attacks: analytical signed backprojection and identity-disjoint diagonal calibration.

## K=10 primary results

| Method | Exposed-image raw cosine | Cross-image fresh-key ASR | Baseline ASR | Cross-image raw ASR | Same-image fresh-key ASR |
|---|---:|---:|---:|---:|---:|
| RO | 0.8103 | 66.63% | 0.53% | 97.22% | 99.99% |
| SRHT | 0.8096 | 64.53% | 0.40% | 97.30% | 100.00% |
| PCA_GUIDED_RO | 0.8635 | 80.96% | 0.13% | 97.53% | 100.00% |

## Exposure growth (K=10 minus K=1)

| Method | Raw-cosine change | Fresh-key ASR change | 95% CI for ASR change |
|---|---:|---:|---:|
| RO | +0.4055 | +57.50 pp | [+55.62, +59.25] pp |
| SRHT | +0.4055 | +56.13 pp | [+54.39, +57.86] pp |
| PCA_GUIDED_RO | +0.3076 | +46.46 pp | [+44.76, +47.99] pp |

## Conservative interpretation

- Lowest mean K=10 cross-image fresh-key ASR: **SRHT**.
- Highest mean K=10 cross-image fresh-key ASR: **PCA_GUIDED_RO**.
- Compare attack ASR with the uninformed baseline and the 0.1% FAR target before calling the leakage practically important.
- A positive paired K=10−K=1 confidence interval supports cumulative multi-template exposure risk.
- This is embedding inversion, not pixel-face reconstruction or proof of PAD bypass.

## Scope

One encoder, one dataset, ten key trials, and a key-compromise threat model. Replication on another face dataset and encoder is required before a broad claim.