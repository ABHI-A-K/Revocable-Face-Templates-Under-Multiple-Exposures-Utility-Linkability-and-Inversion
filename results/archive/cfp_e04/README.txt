CFP-E04-REV — Revoked-template acceptance

START HERE
1. Import CFP-E04-REV.ipynb into Kaggle.
2. Attach the previous CFP-E03-DIM-package.zip as a dataset, keeping the ZIP intact.
   Alternative: attach its CFP-E03-DIM-results.zip and CFP-E03-DIM-protected-template-cache.zip.
3. CPU is sufficient. Run All. No internet, model or raw image files are needed.
4. Download CFP-E04-REV-results.zip from /kaggle/working.

CONTENTS
CFP-E04-REV.ipynb: clean, ready-to-run notebook with explanations.
CFP-E04-REV-executed.ipynb: locally executed copy with captured text outputs.
CFP-E04-REV-results.zip: verified real-data results, per-trial decisions, fold tables, plot and provenance.
CFP-E04-REV_brief.md: short results explanation.
tracker_rows.csv: two rows using your existing five tracker columns.

VALIDATION
All code cells executed against the exact saved E03 templates.
140,000 same-key scores and 200 calibration thresholds/count sets reproduced.
Saved replay decisions and fold aggregation checked for all 2,520,000 scenario trials.
These repeated trials are dependent and must not be described as independent samples.

SETTING AND SCOPE
512 bits primary; 128 bits comparison. Ten fixed E03 keys, 90 ordered renewals.
Thresholds learned only from 9 folds of new-key matching; remaining fold held out.
Pair folds are not identity-disjoint. No threshold tuning on revoked scores.
Primary test: old-key side b against new-key side a, same-person CFP pairs.
Supplementary: reverse capture direction and exact-image renewal on both pair sides.
This evaluates unchanged old-template replay at the matcher, not learned key translation,
known-operator inversion, or a service's obsolete-key version rejection policy.
Public experiment seeds are for reproducibility, not production secrets.

PRIMARY RESULTS
512 bits: 10/315000 replay acceptances (0.0031746%); new-key TAR 80.502857%; observed FAR 0.105714%.
128 bits: 106/315000 replay acceptances (0.0336508%); new-key TAR 35.928571%; observed FAR 0.102857%.
Rates in result CSVs are fractions. Training FAR target is 0.1%, not a guarantee for test FAR.
