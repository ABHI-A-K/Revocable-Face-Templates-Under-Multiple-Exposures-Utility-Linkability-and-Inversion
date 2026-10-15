# LFW-ENC-E01: frozen encoder comparison

Completed 9 October 2026. This recognition extension compares the frozen MobileFaceNet checkpoint with InsightFace w600k_r50 (ArcFace) on the original 6,000 LFW pairs and ten official folds. It does not rerun LFW renewal, linkage, or inversion. CFP renewal/linkage/inversion and identity-bootstrap comparisons use the separate CFP-ENC-E01A–E04 archives.

## Inputs and alignment

All 7,701 original image records were recovered from the public LFW archive. SCRFD det_500m uses threshold 0.5, input size 640x640, all detections, and largest bounding-box area when several faces are found. InsightFace norm_crop produces 112x112 BGR crops. The same crop goes to both encoders. Each output is L2-normalized float32 with 512 coordinates. The original detection presence and face count must match for every image. Every regenerated MobileFaceNet coordinate must equal the frozen embedding exactly. All 7,682 valid images passed; 19 detection failures retain NaN embeddings. The complete checkpoint retains crops, both embeddings, landmarks, bounding boxes, source JPEG hashes, crop hashes, and per-image comparison outcomes.

## Recognition protocol

Raw similarity is float32 dot product. RO uses the original five stored 512x128 orthonormal projection matrices and seeds 2026081101–2026081105. All original packed RO templates are reproduced exactly. SRHT uses a normalized 512-dimensional Sylvester Hadamard matrix, random input signs, and 128 sampled columns. Comparison seeds are derived from the first eight SHA-256 bytes, interpreted little-endian, of `LFW-ENC-E01|{method}|{seed}`. Projection signs are encoded with >=0. Protected similarity is the fraction of equal bits.

PCA-guided RO is fitted separately for each official held-out fold. Fitting uses unique valid images from the other nine folds, excluding all identities appearing in the held-out fold. The float64 centered covariance eigendecomposition retains 256 components; each component sign makes its maximum-absolute coordinate positive. Projected vectors are normalized, without whitening. A 256x128 Gaussian QR orthogonal matrix uses the PCA_GUIDED_RO namespace and positive-diagonal QR sign convention. Fit image rows, means, components, and zero-overlap audits are saved. This fold-specific recognition protocol is distinct from the original identity-disjoint LFW attack protocol. Official folds can share identities during threshold estimation; no identity-disjoint evaluation claim is made.

For each method/key/fold, the threshold is learned solely from impostors in the other nine folds. The acceptance budget is floor(0.001*N); nextafter above the relevant boundary ensures ties cannot exceed that budget with >= acceptance. The held-out fold is never used to choose its threshold. All 640 method/key/fold/view rows meet the training-FAR constraint. Original MobileFaceNet raw and RO fold TP, FP, and pair denominators are reproduced exactly.

Two views are reported: end_to_end_operational includes all 6,000 pairs and automatically rejects detection failures; matcher_only includes 5,965 pairs with two valid faces. Per-key counts are pooled across folds, then key rates are averaged equally. Raw has one unkeyed estimate. `tar_seed_sd` measures key variation, not an identity-bootstrap confidence interval. No new LFW bootstrap interval or paired significance claim is made. When reading scores and thresholds from CSV to recompute decisions, use pandas `float_precision="round_trip"` to preserve nextafter boundaries.

## Reproduction and files

Run `python lfw_encoder/evaluate.py` from the extracted results package to replay recognition from frozen caches. Dependencies: numpy, pandas, scipy. The complete work checkpoint additionally supports `python lfw_encoder/assemble.py` from its 80 completed chunks. Full extraction requires the original dataset ZIP at datasets/lfw-dataset.zip and three ONNX files under models/; invoke extract_all.py with shard arguments 0, 1, 2, 3. Chunk writes are atomic and completed chunks resume automatically. The extraction code asserts model hashes and runtime versions; runtime_versions.json and input_hashes.json record the exact recovered environment and inputs. The full checkpoint includes the resume scripts under recovery/; their Library upload helper is environment-owned and is not required to replay scientific calculations.

`results/run_status.json` is the final validation gate. `alignment_gate.json` and alignment_audit.csv establish frozen-record correspondence. `results/out_of_fold_decisions.csv` provides pair-level decisions. `results/fold_results.csv`, seed_summary.csv, and summary.csv provide aggregates; pca_fit_audit.csv records the identity exclusions. PCA model NPZs and original RO matrices/templates are included. README and manifests distinguish measured results from deterministic validation.

## Reset recovery

A session reset removed an earlier completed run before persistence. The run was repeated from the frozen inputs; the current evidence is complete and revalidates the logged results. No missing observation was filled with a logged number. The expected rounded TAR assertion is an additional regression gate, not a substitute for evaluation or baseline checks. Durable checkpoints were saved during recovery and after completion.
