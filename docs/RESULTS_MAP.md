# Results-to-experiment mapping

| Paper evidence | Location |
|---|---|
| Official LFW recognition | LFW E01/E02 archived fold summaries and pair decisions |
| Official CFP recognition and length | CFP E01 R4/E02/E03-DIM |
| Revoked replay | LFW E04 and CFP E04-REV |
| Learned linkage and projection comparison | LFW E05-ID/E06 and screened CFP E05/E06 |
| Original inversion, intervals and sensitivity | LFW E07, CFP E07, final-validation evidence; reproduction/data |
| Matched CFP encoder study | CFP-ENC E01A/E01B/E02/E03/E04; reproduction/data/encoder_extension |
| LFW encoder recognition | experiments/lfw_encoder/results |
| Payload scaling | reproduction/build_storage.py; deterministic arithmetic only |

## Figure and numerical generators

- `reproduction/build_compact_comparisons.py`
- `reproduction/build_encoder_figures.py`
- `reproduction/build_extended_figures.py`
- `reproduction/build_figures.py`
- `reproduction/build_flowchart.py`
- `reproduction/build_methodology_bw.py`
- `reproduction/build_storage.py`
- `reproduction/editorial_metrics.py`
- `reproduction/validate_encoder_evidence.py`

## Curated evidence used by the paper

- `reproduction/data/attack_rows.tex`
- `reproduction/data/cfp_correspondence_audit_summary.json`
- `reproduction/data/encoder_extension/CFP_encoder_comparison.csv`
- `reproduction/data/encoder_extension/E01A_recognition_summary.csv`
- `reproduction/data/encoder_extension/E01A_run_status.json`
- `reproduction/data/encoder_extension/E01B_protocol.json`
- `reproduction/data/encoder_extension/E01B_raw_subset_summary.csv`
- `reproduction/data/encoder_extension/E01B_run_status.json`
- `reproduction/data/encoder_extension/E01B_utility_summary.csv`
- `reproduction/data/encoder_extension/E02_linkage_summary.csv`
- `reproduction/data/encoder_extension/E02_replay_summary.csv`
- `reproduction/data/encoder_extension/E02_run_status.json`
- `reproduction/data/encoder_extension/E03_attack_summary.csv`
- `reproduction/data/encoder_extension/E03_run_status.json`
- `reproduction/data/encoder_extension/E04_identity_bootstrap_intervals.csv`
- `reproduction/data/encoder_extension/E04_paired_K10_minus_K1_intervals.csv`
- `reproduction/data/encoder_extension/E04_paired_encoder_intervals.csv`
- `reproduction/data/encoder_extension/E04_protocol.json`
- `reproduction/data/encoder_extension/E04_run_status.json`
- `reproduction/data/encoder_extension/E04_threshold_sweep.csv`
- `reproduction/data/encoder_extension/cross_archive_validation.json`
- `reproduction/data/encoder_extension/editorial_validation.json`
- `reproduction/data/encoder_extension/experiment_tracker.tsv`
- `reproduction/data/encoder_extension/lfw/PROTOCOL.md`
- `reproduction/data/encoder_extension/lfw/alignment_gate.json`
- `reproduction/data/encoder_extension/lfw/fold_results.csv`
- `reproduction/data/encoder_extension/lfw/input_hashes.json`
- `reproduction/data/encoder_extension/lfw/pca_fit_audit.csv`
- `reproduction/data/encoder_extension/lfw/run_status.json`
- `reproduction/data/encoder_extension/lfw/runtime_versions.json`
- `reproduction/data/encoder_extension/lfw/seed_summary.csv`
- `reproduction/data/encoder_extension/lfw/summary.csv`
- `reproduction/data/encoder_extension/manuscript_validation.json`
- `reproduction/data/identity_bootstrap_intervals.csv`
- `reproduction/data/legacy_results.csv`
- `reproduction/data/lfw_experiment_config.json`
- `reproduction/data/lfw_matcher_thresholds.csv`
- `reproduction/data/lfw_pca_audit.json`
- `reproduction/data/matcher_identity_intervals.csv`
- `reproduction/data/paired_identity_contrasts.csv`
- `reproduction/data/paper_attack_intervals.csv`
- `reproduction/data/paper_threshold_table.csv`
- `reproduction/data/template_storage_accounting.csv`
- `reproduction/data/threshold_attack_detail.csv`
- `reproduction/data/threshold_attack_summary.csv`
- `reproduction/data/threshold_utility_detail.csv`
- `reproduction/data/threshold_utility_summary.csv`
