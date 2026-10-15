# LFW-E02-seeded-RO results

**Researcher:** Abhishek A K  
**Completed (UTC):** 2026-08-11T06:33:07.036448+00:00

## Controlled comparison

E02 reuses the frozen LFW-E01 detector, recognizer, embeddings, image order, official folds, and failure handling. Only the stored representation and matcher change: 512-D float cosine matching becomes 128-bit seeded-RO BioHash Hamming matching.

## Main low-FAR result

| Evaluation | E01 raw TAR | E02 TAR mean ± SD | Change (percentage points) | E02 observed FAR mean ± SD |
|---|---:|---:|---:|---:|
| end_to_end_operational | 98.0333% | 82.4667% ± 2.2619% | -15.5666 | 0.0933% ± 0.0279% |
| matcher_only | 98.6582% | 82.9923% ± 2.2763% | -15.6659 | 0.0938% ± 0.0280% |

## Storage

- Raw template: 2048 bytes.
- Packed protected template: 16 bytes.
- Reduction: 128.0×.

## Interpretation rules

- E02 values are mean and sample standard deviation across five fixed seeds.
- Every held-out threshold is learned from the other nine folds only.
- Detection failures are rejected in the operational view and omitted only in matcher-only analysis.
- EER and pooled ROC AUC are descriptive; they are not deployment thresholds.
- Cross-key unlinkability, revocation, and inversion resistance require separate experiments.

## Editable run notes

- Controlled change: raw 512-D cosine matching -> 128-bit Hamming matching.
- Five fixed experimental seeds are evaluated independently.
- One common transform is used for all samples within a seed trial.
- Cross-key unlinkability and inversion resistance are outside E02 recognition scope.
