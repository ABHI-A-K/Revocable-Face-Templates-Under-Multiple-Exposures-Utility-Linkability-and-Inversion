CFP-E03-DIM experiment package

Open CFP-E03-DIM.ipynb for a clean, commented Kaggle run.
Open CFP-E03-DIM-executed.ipynb to inspect the completed local CPU run.
The two ZIPs contain complete results and reusable protected templates/projections.
The brief and tracker CSV are also included at the top level for convenience.

Kaggle inputs (reuse your existing files; raw embeddings are not duplicated here):
- CFP-FP-E01-R4-aligned-embedding-cache.zip
- CFP-FP-E01-R4-aligned-baseline-results.zip
- CFP-FP-E02-seeded-RO-128bit-results.zip

Attach one copy of each ZIP, upload the clean notebook, select CPU, Run All.
Ten seeds, four lengths (64/128/256/512), fixed E01-R4 pair protocol.
The 128-bit control reproduced all 70,000 E02 pair scores exactly.

This package contains an actual local execution. Kaggle rerun is optional for
independent reproduction. Compare metrics and input hashes; latency and matrix
byte hashes can depend on the runtime and numerical libraries.

Intervals describe random-key variability on fixed pairs. Thresholds are trained
at FAR <= 0.1%; report the actual held-out FAR alongside TAR. No final dimension
is selected. Cross-key linkage is same-image only; security attacks remain future work.
