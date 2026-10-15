# CFP photographic correspondence audit

Status: **audit complete; 26 correspondences remain unresolved**. No more file uploads are needed for this review.

Both bin uploads have the expected SHA-256 and are identical. All 14,000 encoded crop hashes match the earlier cache metadata. The previous source check matched all 5,901 source references by decoded pixel hash. Together these checks bind the audit to the actual experiment inputs.

## Results
- Strict first-pass criterion: 4295 / 5,704.
- Follow-up overlap-only support: 1380. These satisfy the feature and correlation criteria but fail the initial 60% whole-crop overlap requirement. The change in interpretation is explicitly post hoc; it is not a silently relaxed original pass gate.
- Follow-up ECC registration support: 3 / 29 inspected exceptions.
- Still unresolved: 26. A failed registration criterion is not proof of a wrong identity.
- Negative controls: 0 / 500 sampled different-identity photographs and 0 / 500 sampled same-identity different photographs passed the strict check. None passed the follow-up feature/correlation rule either. ECC was applied to proposed positives only, so it has no separately measured control error rate.

## Method
SIFT local image features (not face-recognition embeddings), ratio matching at 0.7, deduplicated coordinate pairs, affine RANSAC at 2 pixels with 3,000 iterations. Strict support requires ≥8 inliers, convex-hull coverage ≥10% of the crop, overlapping-pixel correlation ≥0.95, overlap ≥60%, and overlapping target standard deviation ≥10. Crop images are resized to 224×224; sources capped at 640 pixels. Follow-up overlap support omits the 60% criterion and is separately labeled. Three ECC-supported cases require correlation ≥0.98, overlap ≥40% and target standard deviation ≥10. These are analyst-selected, post-hoc diagnostics, not externally validated identity-certification thresholds.

## What remains
Open `review_26_photographic_pairs.html` locally and review photographic correspondence for the 26 cases. Fill `review_decisions.csv` with SAME_SOURCE, DIFFERENT_SOURCE or UNCERTAIN, reviewer and evidence. Do not infer real-world identity from the face. Original dataset annotation is the identity source once photographic correspondence is established.

The unresolved images touch 5,760 saved E07 attack rows involving 3 distinct test identities. `unresolved_partition_impact.csv` also records their occurrence in calibration, training and validation partitions. If any source mapping or inclusion changes, rerun affected fits and evaluation; merely deleting affected test scores does not validate the original pipeline.

The 5,678 supported correspondences are technical corroboration, not a blanket statement that every identity has been independently certified. No identity labels, templates, scores or manuscript conclusions were changed. The earlier statistical and threshold analyses remain conditional on the existing screened mapping.

## Reproducibility
The scripts use relative paths matching the existing workspace: `upload/cfp_fp.bin`, `upload/CFP-FP-source-evidence.zip`, and `final_audit/input/` manifests. Included input manifests reproduce those references. Use Python with numpy, pandas, Pillow and OpenCV (run used opencv-python-headless 5.0.0.93). Run `check.py`, then `controls.py`, then `refine.py`. The first 100 rows were used to exercise the implementation before full execution. All outputs and analyst-selected follow-up rules are retained.
