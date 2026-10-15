# Final validation status — LFW and CFP-FP

**Status: analysis completed on the existing screened data; independent CFP identity provenance remains open.**

## Completed work
- Checked all 40 CFP E07 output hashes and recomputed its 432,000 acceptance decisions. Verified saved target identities, partition membership and fresh-key exclusions.
- Reanalysed E07 uncertainty for LFW (1,200 distinct target identities) and CFP (195 distinct test identities across overlapping splits). Used 2,000 bootstrap draws of identities, sharing each identity weight across all its repeats. Split/fold estimates retain equal weighting.
- Computed paired K10−K1 and method-minus-RO intervals using the same resampling weights.
- Recomputed CFP matcher scores using the saved E06 binary templates and hash-bound raw embedding cache. Confirmed the subset matches E07 and reproduced all 363 original matcher operating points.
- Selected thresholds ONLY from validation impostors at FAR targets 0.01%, 0.05%, 0.1%, 0.5% and 1%. Re-evaluated legitimate TAR, observed FAR and raw/protected attack success. No threshold was selected from test outcomes.
- Added conditional CFP matcher-rate intervals: an impostor pair is weighted by both identities so shared subjects are not treated as independent pairs.

## Interpretation and limits of the intervals
These are conditional target-identity bootstrap intervals: fitted attackers, PCA models, keys, partitions, selected images and validation thresholds remain fixed. They do not estimate variability from retraining, new key schedules, new images, calibration-sample selection or uncertain identity mapping. They are exploratory pointwise intervals, not simultaneous multiple-comparison guarantees. Overlapping training sets remain a limitation. Do not describe them as full-pipeline population confidence intervals.

## K10 calibrated fresh-key attack success

| Dataset | Bits | Method | ASR | Conditional 95% interval |
|---|---:|---|---:|---:|
| CFP_FP_SCREENED | 128 | PCA_GUIDED_RO | 32.94% | 29.35–36.96% |
| CFP_FP_SCREENED | 128 | RO | 19.66% | 17.23–22.38% |
| CFP_FP_SCREENED | 128 | SRHT | 18.84% | 16.42–21.41% |
| CFP_FP_SCREENED | 512 | PCA_GUIDED_RO | 78.54% | 73.89–83.03% |
| CFP_FP_SCREENED | 512 | RO | 78.02% | 73.38–82.52% |
| CFP_FP_SCREENED | 512 | SRHT | 77.52% | 72.95–82.10% |
| LFW | 128 | PCA_GUIDED_RO | 80.96% | 79.47–82.44% |
| LFW | 128 | RO | 66.63% | 64.96–68.31% |
| LFW | 128 | SRHT | 64.53% | 62.86–66.22% |

All calibrated K10−K1 attack-success intervals remain above zero in both datasets. This supports cumulative exposure risk within these fixed experimental configurations. It does not establish a universally monotonic attack for every individual.

## CFP 512-bit threshold sensitivity

| Validation FAR target | Method | Test TAR | Observed test FAR | K10 calibrated ASR |
|---:|---|---:|---:|---:|
| 0.01% | PCA_GUIDED_RO | 74.19% | 0.0736% | 69.81% |
| 0.01% | RO | 76.04% | 0.0687% | 70.56% |
| 0.01% | SRHT | 75.20% | 0.0656% | 69.59% |
| 0.05% | PCA_GUIDED_RO | 80.28% | 0.1418% | 76.39% |
| 0.05% | RO | 80.18% | 0.1149% | 75.33% |
| 0.05% | SRHT | 80.29% | 0.1216% | 75.49% |
| 0.1% | PCA_GUIDED_RO | 82.38% | 0.2002% | 78.54% |
| 0.1% | RO | 82.52% | 0.1715% | 78.02% |
| 0.1% | SRHT | 81.95% | 0.1670% | 77.52% |
| 0.5% | PCA_GUIDED_RO | 88.13% | 0.6291% | 84.82% |
| 0.5% | RO | 88.08% | 0.5905% | 84.47% |
| 0.5% | SRHT | 87.88% | 0.5745% | 84.19% |
| 1% | PCA_GUIDED_RO | 90.40% | 1.1683% | 87.38% |
| 1% | RO | 90.38% | 1.0556% | 87.17% |
| 1% | SRHT | 90.37% | 1.0446% | 87.28% |

At the strictest target, calibrated ASR remains 69.59–70.56%. Tightening thresholds reduces attack success but does not remove the demonstrated compromise risk.
**Low-FAR resolution:** validation has only 2,105–2,173 impostor pairs per split. A target of 0.01% permits zero validation false accepts and selects a threshold just above the maximum validation impostor score. This cannot demonstrate that the population FAR is 0.01%; observed test FAR is 0.0656–0.0736%. Zero observed events do not establish zero risk. Conditional percentile intervals may also be unreliable in sparse-event tails.

## Identity provenance — NOT CLOSED
The 5,704 retained crops are SCREEN_SUPPORTED by the earlier geometric procedure. The 186 unresolved crops remain excluded. Source pixel hashes show internally consistent image annotations, and partitions are disjoint under these annotations. Those checks do not independently establish that each bin crop corresponds to its proposed original source image.
QA01 explicitly describes its review as AI-assisted, not human review or definitive identity verification. Its 55 exported flagged crop images are not the full original source-image corpus. No labels were promoted and no independent-verification flag was changed.
The missing inputs are the original CFP source image directory (the chinafax/cfpw-dataset input used earlier) and the exact frozen cfp_fp.bin (SHA-256 76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370), or an authoritative source-to-crop generation manifest. The accompanying offline collector gathers those inputs without downloading or uploading anything. Collection alone does not verify identity provenance.
Once supplied, independently establish photographic correspondence to dataset-annotated source images without using the evaluated MobileFaceNet embeddings as label evidence. If any included correspondence fails, freeze a corrected manifest and rerun the affected PCA/training/evaluation splits; simply filtering test outcomes after seeing them is insufficient.

## Scope still open
- No LFW threshold sweep was run: its result archive has attack scores and selected matcher thresholds, but not the matcher calibration scores or raw embedding cache needed to select new thresholds correctly.
- Uncertainty work here covers E07 and CFP matcher utility; it is not a new statistical audit of every E01–E06 endpoint.
- No manuscript file was edited. This package supplies analysis tables and qualified interpretation for the next manuscript revision.

## Reproduce
Run the two Python commands below with the original input ZIPs. No internet is needed. Dependencies: Python, numpy, pandas, scipy, threadpoolctl. Plotting/report generation additionally used matplotlib.
```bash
python run_validation.py --e06 CFP-E06.zip --cfp CFP-E07.zip --lfw LFW-E07.zip --cache CFP-cache.zip --out results
python matcher_uncertainty.py --cfp CFP-E07.zip --results results
```
Fractions are stored on the 0–1 scale in CSVs; multiply by 100 for percentages. Read score/threshold CSVs with pandas float_precision="round_trip". Retain all source reports and input hashes. This is post-hoc analysis; target grid and bootstrap specifications were chosen after the original experiments, not preregistered.

## Files
- results/paper_attack_intervals.csv and paper_threshold_table.csv: compact paper tables.
- results/identity_bootstrap_intervals.csv, paired_identity_contrasts.csv, matcher_identity_intervals.csv: full conditional intervals.
- results/threshold_utility_detail.csv and threshold_attack_detail.csv: split/key operating points.
- results/cfp_matcher_scores.npz: regenerated validation/test score vectors; pair ordering follows the original E07 pair manifests.
- results/independent_source_review_required.csv: pending evidence register, not a demand to manually label 5,704 faces.
- CFP-FP-source-evidence-collector.ipynb: offline collection step for missing inputs.
- tracker_rows.tsv: header-free tracker updates.
