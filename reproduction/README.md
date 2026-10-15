# Face-template research manuscript: encoder comparison

**Revised:** 14 October 2026  
**Title:** Revocable Face Templates Under Multiple Exposures: Utility, Linkability, and Inversion  
**Author:** Abhishek A. K.  
**Format:** IEEEtran journal, 10 pt, two columns, US Letter.

## Compile or open in Overleaf

Upload this ZIP as a new Overleaf project, select `main.tex` and pdfLaTeX, and recompile. All figures are included as vector PDFs; Python is not needed to compile. Local command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

The manuscript is provided for author review. No submission or Overleaf upload has been made. Target-venue page limits, author declarations, and submission metadata still need to be checked for the selected journal.

## Current revision

Revised academic prose throughout the main narrative, with technical claims and original equations retained. Added a nonfloating notation table before the first equation and a black-and-white single-column methodology flow. The PDF remains 17 pages. No detector score is asserted.

## Editorial structure update

The paper retains seven numbered sections and the introduction roadmap. This revision expands the mathematical definitions, reconstruction interpretation, operating-point rules, and paired bootstrap formulation. It includes 16 tables, 13 figures, 38 references, and 17 compiled pages. Six published 2024-2025 studies were added; the existing 2026 benchmark was retained. Experimental data were not changed or rerun.

Run `python editorial_metrics.py` to validate the shared metrics and comparison tables. Run `python build_encoder_figures.py` and then `python build_extended_figures.py` to regenerate the encoder figures and current methodology diagram. The older `build_flowchart.py` is retained for provenance but generates the superseded diagram; use the extended script last. Pre-rendered vector PDFs are included, so Python is unnecessary for Overleaf compilation.

`RECENT_REFERENCE_VERIFICATION.md` records verification of the six new citations. `REVISION_VALIDATION.json` records data preservation and editorial checks. The literature discussion uses independent phrasing with citations; no plagiarism or AI-detector score is asserted.

## Experimental changes retained from 9 October

- Added a clearly separated second-encoder section comparing frozen MobileFaceNet and ResNet50 ArcFace checkpoints.
- Completed LFW official-pair recognition for raw embeddings and RO, SRHT, and PCA-guided RO at 128 bits. All 7,701 original image records were checked, with exact reproduction of every valid frozen MobileFaceNet vector and original raw/RO fold counts.
- Integrated the supplied CFP-ENC-E01A–E04 recognition, renewal, template-only linkage, inversion, identity-bootstrap, paired comparison, and threshold evidence. Independently checked all 672 E04 primary estimates against E03 aggregates.
- Added recognition and attack-comparison tables and encoder-specific exposure curves. The figure script explicitly filters encoder and asserts unique condition rows. This corrects a plotting-only issue in the supplied E04 notebook; numerical evidence is retained unchanged.
- Preserved the original MobileFaceNet baseline tables as a separate study. Small numerical differences between original and replicated CFP runs are not silently substituted or pooled.
- Updated the abstract, introduction, methods, discussion, and conclusion. Higher recognition with the second checkpoint did not remove the demonstrated known-operator reconstruction risk.

## Scientific boundaries

The edge contribution connects packed-template storage, recognition, and security. A 512-coordinate float32 embedding is 2048 bytes; packed 128/512-bit templates are 16/64 bytes. These payload calculations exclude encoder weights, activations, operator/PCA parameters, keys, and metadata. No device latency, memory, power, or energy was measured. Kiosks and smart gadgets are intended applications, not tested hardware.

The two pretrained checkpoints differ in more than architecture. Their comparison does not isolate training data or loss and does not establish a general security ranking of encoders. The demonstrated attack injects digital embeddings; it does not generate face photographs or demonstrate camera-level impersonation.

Official benchmark recognition and screened, identity-disjoint attack protocols are separate. The new LFW work covers recognition only; no ResNet50 LFW attack result or new LFW bootstrap interval is claimed. Its PCA is fitted per official fold after excluding held-out identities. The original LFW attack PCA protocol is unchanged.

CFP identity-bootstrap intervals condition on fitted models and thresholds and retain global identity dependencies across overlapping splits and keys. The encoder replication reports three validation-FAR targets (0.0001, 0.001, 0.01) and a descriptive discrete-threshold sweep; the original baseline reports five targets. Renewal/linkage results remain descriptive. Neither low replay rates nor near-chance linkage establishes universal unlinkability.

## Contents and reproduction

- `main.tex`, `sections/`, `references.bib`: editable manuscript and 38-reference bibliography.
- `figures/`: vector figures and supplementary encoder threshold plots.
- `build_encoder_figures.py`: rebuilds corrected encoder-specific figures from supplied CSV evidence.
- `build_figures.py`, `build_storage.py`: original result plots and deterministic payload accounting. `build_extended_figures.py`: current methodology and additional evidence plots.
- `data/`: original evidence plus separately identified `encoder_extension/` data. `encoder_extension/lfw/PROTOCOL.md` documents the new recognition run.
- `SOURCE_MAP.md`: evidence provenance and historical editorial notes.
- `MANIFEST_SHA256.json`: source package hashes.
- `IEEEtran.cls`, `IEEEtran.bst`, `IEEEtran_README.txt`: retained third-party template files.

The separate LFW results-and-replay ZIP contains frozen embeddings, matrices, templates, fitted PCA models, per-pair decisions, scripts, and validation gates. The complete work checkpoint additionally preserves aligned image chunks and recovery scripts. Figure generation does not rerun CFP experiments. Supplied CFP results were checked and integrated, not claimed as newly executed attacks in this session.

No authorship, funding, ethics, or conflict declaration has been invented. No plagiarism or AI-detector score is claimed. Bibliography verification notes below describe earlier work; this revision did not independently reverify every reference.


## Final structural revision, 12 October 2026
All subsection and sub-subsection headings have been removed; seven main section headings remain. Contributions are integrated into a continuous introduction spanning more than one page. The methodology is a monochrome vector flowchart with standard input/output, process, decision and terminator shapes. Regenerate it with `python build_methodology_bw.py` (also called by build_extended_figures.py). All experimental metrics, 38 references, 16 tables and 13 figures are retained. Final length: 17 pages.


## Compact layout revision, 13 October 2026
The final manuscript is 17 pages. The methodology is a compact overview rather than a detailed execution flow. Comparison plots use single-column layouts where appropriate; remaining panel heights and float gaps are reduced while preserving readable legends and intervals. All numerical evidence and 13 figures/16 tables are retained. Run build_encoder_figures.py followed by build_extended_figures.py to recreate the current figures; the latter invokes build_compact_comparisons.py and build_methodology_bw.py.
