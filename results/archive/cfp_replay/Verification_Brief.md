# CFP-E03 → CFP-E04: reproduction completed

Private research update — 17 September 2026

**Outcome: PASS.** The newly supplied E03 package contains the protected-template cache required to reproduce E04. Its nested results and cache archive hashes exactly match the inputs recorded in E04.

## What was done

An independently written local CPU program calculated Hamming similarity directly from packed template bytes. Hamming similarity means the proportion of matching bits. This program used XOR and bit counting, rather than copying E04's unpacked-bit comparison implementation. It also recalculated thresholds using nine training folds and evaluated the remaining fold. All archive manifest entries passed size and checksum checks.

| Verification | Exact matches |
|---|---:|
| E03 same-key comparison scores | 140,000 |
| Retrained thresholds | 200 |
| E04 fold control records, including genuine/impostor counts | 200 |
| E04 renewal comparison scores | 2,520,000 |
| E04 renewal acceptance decisions | 2,520,000 |

The totals cover 512-bit and 128-bit configurations. Renewal comparisons cover all four scenarios and 90 directed old-to-new key transitions per configuration.

## Results remain unchanged

| Primary result | 512 bits | 128 bits |
|---|---:|---:|
| Old-template acceptance | 10 / 315,000 (0.00317%) | 106 / 315,000 (0.03365%) |
| Genuine acceptance after renewal | 80.50% | 35.93% |
| Observed impostor acceptance | 0.1057% | 0.1029% |
| Packed template size | 64 bytes | 16 bytes |

The 512-bit configuration remains the primary setting for continued investigation. Its genuine rejection rate of approximately 19.5% remains a substantial usability limitation. Low old-template acceptance does not establish zero replay risk.

## Updated manuscript paragraph

The renewal experiment was independently reproduced from the archived CFP-E03 protected templates using a packed-byte XOR and population-count implementation of Hamming similarity. The E03 source archive hashes matched those recorded in CFP-E04. Across the 512-bit and 128-bit configurations, all 140,000 same-key scores, 200 recalibrated thresholds and associated fold control counts, and 2,520,000 renewal scores and decisions matched the archived results exactly. Primary unchanged old-template acceptance remained 10/315,000 at 512 bits and 106/315,000 at 128 bits. This verification supports computational reproducibility from the protected-template cache; it does not constitute a new dataset evaluation or an independent statistical replication.

## What this changes in the previous addendum

The earlier statement that E03 templates were unavailable is superseded by this verification. We can now substantiate score regeneration and threshold recalibration from cached protected templates. The original case-study manuscript and previous addendum have not been silently edited; use this paragraph when integrating the evidence, and update any abstract or methods statement that still says no experiments are available.

This is not a full image-to-result reproduction: original images, face alignment, embedding extraction, and projection from raw embeddings were not rerun. Repeated people, images, keys and pairs remain dependent; the pair folds are not identity-disjoint. Projection configurations are independently shaped, not nested truncations. No inference about inversion resistance, learned attacks, or deployed cloud security follows from this audit.

## Next action

For the cloud case study, use the verified biometric results alongside the already completed CLOUD-CS01 recovery-control exercise, explicitly separating measured biometric comparisons from deterministic software checks. The next new cloud experiment should authenticate request metadata and test two verifier instances, including a stale verifier, so that changing an old template's version label is rejected before comparison. This requires an explicit request-binding design; a version label alone is insufficient.

For the main biometric research, continue the existing CFP-E05 identity-disjoint attacker work once its image-to-identity mapping checks pass. Identity-disjoint means the attacker learns from different people than those used for testing. Do not bypass the earlier mapping failure or count this verification as a new security experiment.

Your immediate task is simply to read this brief. A rerun is optional: the verification has already completed locally. The supplied notebook accepts the same two private outer ZIP packages and runs on CPU without internet.
