# Evidence and reporting map

| Manuscript item (LaTeX label) | Evidence | Scope |
|---|---|---|
| Review tables (`tab:review1`–`tab:review3`) | Cited primary papers, reports and surveys | 30 reviewed studies; not external reruns |
| Threat / protocol tables (`tab:threat`, `tab:protocol`) | Executed LFW E07 and CFP E06/E07 configuration | Different attacker access and dataset partitions |
| Official utility / replay (`tab:official`, `tab:replay`) | `data/legacy_results.csv` | Official pairs; replay trials are dependent |
| Learned linkage (`tab:linkage`) | LFW legacy E06 and executed CFP E06-PROJ outputs | K10 AUC on custom identity-disjoint protocols |
| Custom utility (`tab:attackutility`) | `data/lfw_matcher_thresholds.csv`; `data/threshold_utility_summary.csv` at target 0.001 | Equal fold/key or split/key means |
| Inversion / exposure (`tab:inversion`, `fig:exposure`) | `data/identity_bootstrap_intervals.csv` | K=1/2/5/10; calibrated fresh ASR, cosine, raw ASR |
| Paired contrasts | `data/paired_identity_contrasts.csv` | Shared global identity bootstrap weights |
| Threshold plot (`fig:thresholds`) | `data/threshold_utility_summary.csv`, `data/threshold_attack_summary.csv` | CFP 512-bit, K10 calibrated, five validation targets |
| Methodology (`fig:methodology`) | Methods and archived experiment configuration | Diagram of existing protocol; no additional experiment |
| Prior-work comparison (`tab:priorcomparison`) | Cited primary studies | Qualitative scope and threat comparison; no numerical ranking |
| Threshold data | `data/paper_threshold_table.csv` | TAR, observed FAR and ASR by operating point |
| CFP audit | `data/cfp_correspondence_audit_summary.json` | All 26 researcher confirmations; no mapping/subset changes |

All rate fields in source CSVs are fractions unless a column explicitly states otherwise. Manuscript tables convert acceptance rates to percentages. AUC and cosine remain fractions. Values are rounded only for display. Figure error bars are pointwise conditional 95% intervals, with 2000 global identity bootstrap replicates, not independent key-trial intervals.

The audit summary preserves its original first-pass `REVIEW_REQUIRED: 26` count. Its final fields `human_review_same_source: 26`, `unresolved_flagged_cases: 0`, and status `FLAGGED_CASE_REVIEW_CLOSED_BY_RESEARCHER_CONFIRMATION` give the disposition. The initial category count must not be mistaken for 26 still-unresolved cases.

Exact frozen QA01 bin SHA-256:
`76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370`

Earlier key-variation intervals have not been relabeled as identity-population intervals. Official CFP pairs and screened CFP attack pairs are not pooled. The statistical/threshold work reuses completed experiment evidence; it is not a fresh encoder-training experiment.

## Publication-reference verification

The current bibliography cites 13 journal articles and 16 conference/workshop papers, plus two ISO/IEC standards and the original LFW dataset report. Conference workshops are identified as workshops, rather than described as main-conference papers. The report and standards are retained for dataset provenance and evaluation terminology, not counted as peer-reviewed research articles.

Verified publication substitutions:

| Citation | Published source used |
|---|---|
| Neural face-template survey | IEEE TIFS 18, 639–666 (2023), DOI 10.1109/TIFS.2022.3228494 |
| MLP-Hash | EUSIPCO 2023, 605–609, DOI 10.23919/EUSIPCO58844.2023.10289780 |
| Face adapter | CVPR Workshops 2025, 5623–5632, verified against the CVF proceedings record |
| FaceCloak | IEEE FG 2025, verified against the authors' publication list and official author repository |
| Replacement survey | Rathgeb and Uhl, EURASIP Journal on Information Security 2011, article 3, DOI 10.1186/1687-417X-2011-3 |

Primary verification records:
- https://www.idiap.ch/~vkrivokuca/
- https://publications.idiap.ch/publications/show/5066
- https://openaccess.thecvf.com/content/CVPR2025W/ABAW/html/Shahreza_Face_Reconstruction_from_Face_Embeddings_using_Adapter_to_a_Face_CVPRW_2025_paper.html
- https://github.com/sudban3089/FaceCloak
- https://chinmayhegde.github.io/pubs/
- https://doi.org/10.1186/1687-417X-2011-3

The remaining references retain their journal or conference publication identities. The benchmark DOI 10.1186/s13640-025-00679-y is dated 2 April 2026 by the publisher, despite 2025 appearing in its DOI. Public searches used literature titles only, without private experiment data. No experiments were rerun for this editorial revision.

## October 2026 editorial update

The bibliography replaces Dong et al. (2024) with two published sources whose claims are kept distinct:
- Wang et al., IEEE/CVF WACV Workshops 2021, 70–77, DOI 10.1109/WACVW52041.2021.00012: constrained pre-image attacks on BioHashing and Index-of-Max hashing.
- Li and Hu, Concurrency and Computation: Practice and Experience 26(8), 1593–1605 (2014), DOI 10.1002/cpe.3042: record-multiplicity attacks with transformation parameters, evaluated on fingerprint constructions.

Primary records: https://openaccess.thecvf.com/content/WACV2021W/XAI4B/html/Wang_Interpretable_Security_Analysis_of_Cancellable_Biometrics_Using_Constrained-Optimized_Similarity-Based_Attack_WACVW_2021_paper.html and https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.3042 . The publisher assigns Li and Hu to its 2014 issue, although the online-first date is 2013. The original reference's removal is an author-requested citation change, not an assessment that its journal is illegitimate.

## Edge-design revision, 6 October 2026

`tab:storage` is derived from the representation definitions: 512 float32 coordinates require 2048 bytes; 128 and 512 packed bits require 16 and 64 bytes. `build_storage.py` reproduces `data/template_storage_accounting.csv`. Reduction factors are 128 and 32; exact percentage savings are 99.21875% and 96.875%, rounded in the table to 99.22% and 96.88%. These are payload calculations, not hardware measurements.

The two stated contributions combine repeated-exposure analysis with compact-template design for intended edge use. No first-in-literature claim is made for either repeated compromise or compact binary representations. The abstract, numbered contributions, results, and conclusion use the same ASR ranges and official CFP verification values. All 15 pre-existing evidence files remain byte-identical to the 5 October source package. No new attack run, recognition trial, or hardware measurement was performed for this revision.

The payload table excludes model weights, intermediate memory, operators, PCA parameters, keys, and metadata. Hardware latency, peak memory, and energy remain future work. The diagram and captions use this same boundary.

## Encoder-comparison revision, 9 October 2026

The preceding editorial notes are historical and describe their stated revision dates. The 9 October revision adds a genuinely executed LFW recognition extension and integrates supplied CFP replication evidence.

| Manuscript item | Evidence | Scope |
|---|---|---|
| Second-encoder CFP table (`tab:encoder-cfp`) | `data/encoder_extension/CFP_encoder_comparison.csv`, E01B utility, E03 attacks, E04 paired intervals | Screened custom protocol; encoder-specific fits and thresholds |
| LFW encoder table (`tab:encoder-lfw`) | `data/encoder_extension/lfw/summary.csv`, fold/seed summaries, run status | Official ten-fold recognition, 128 bits; five keys; no new LFW attack claim |
| Encoder exposure figure (`fig:encoder-exposure`) | E04 identity-bootstrap intervals, checked against E03 | Explicitly filtered by encoder; pointwise conditional intervals |
| Replicated renewal/linkage | E02 replay/linkage summaries | Descriptive tested protocols; no new significance claim |
| Replicated threshold sensitivity | E04 identity-bootstrap intervals and threshold sweep | Three nominal validation targets plus descriptive discrete sweep |
| Cross-archive validation | `cross_archive_validation.json` | All 672 primary E04 estimates match E03 to 1e-12 |

The original E04 plotting cell omitted the encoder filter in exposure and threshold loops. `build_encoder_figures.py` fixes that presentation error using unchanged numerical tables, explicitly filters encoder, and asserts unique condition rows. Original E04 plots are not used. Supplementary corrected threshold figures are included in `figures/`.

LFW extraction reproduces all 7,682 valid MobileFaceNet vectors exactly, retains 19 detection failures among 7,701 records, and feeds the same crops to ResNet50. It reproduces original raw/RO fold counts and every valid packed RO template. Matcher-only evaluation has 5,965 pairs; operational evaluation includes all 6,000. PCA fitting excludes all identities in each held-out official fold, while official-fold threshold estimation can share identities. Full details and input hashes are in the LFW protocol and replay archive.

The supplied CFP E01A–E04 archives provide the second-encoder attack evidence; these attacks were not rerun in this recovery session. The baseline and encoder replication retain their own reported values and threshold grids. No participant population, hardware experiment, camera attack, or LFW second-encoder inversion result has been added by implication. No submission has been made.

## Editorial structure revision, 10 October 2026

The manuscript now separates methodology from experimental setup and integrates encoder comparisons within Results. The section roadmap, table captions, and explanatory notes have been completed. The prior-work comparison table is located with Related Work. Headline ranges are generated from the existing evidence by `editorial_metrics.py` and reused across the abstract, introduction, results, discussion, and conclusion. `editorial_validation.json` checks the shared ranges, both encoder tables, and all table/figure notes. Original and replicated CFP estimates remain distinctly labelled; their small differences are not errors to be overwritten. No new experiment or bibliography verification was performed in this editorial revision.


## 12 October 2026 extension
New paired plots and recognition/cosine figures read existing encoder_extension CSVs; no synthetic experimental observations were added. Storage curves are deterministic byte accounting. Recent references are logged in RECENT_REFERENCE_VERIFICATION.md. Statistical definitions are in sections/statistical_formulation.tex. Current methodology is generated by build_extended_figures.py.
