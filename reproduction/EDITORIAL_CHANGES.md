# 14 October 2026 revision

- Reworked abstract and narrative across all seven sections for clear academic phrasing.
- Preserved mathematical equations, empirical data, citations and scientific scope.
- Added notation before the first equation.
- Replaced methodology illustration with a single-column black-and-white vector flow.
- Verified 17 pages, 16 tables, 13 figures, no unresolved references or overfull boxes.
- No AI-detection or similarity score was measured or promised.

# Revision completed 12 October 2026

1. Added formal normalization, RO QR construction, SRHT operator, PCA fitting, payload accounting, asymptotic operation counts, and explicit attacker information sets.
2. Added a signed-measurement feasible-set interpretation and Gaussian expectation derivation, explicitly distinguished from a theorem for the tested structured operators.
3. Formalized coordinate calibration, PCA lifting, TAR/FAR, old-record replay, linkage AUC, identity bootstrap, and paired differences.
4. Strengthened abstract and scientific framing while preserving empirical contribution boundaries.
5. Replaced methodology diagram with a vector workflow showing fitting controls, attacker inputs, and the fresh-key verifier boundary.
6. Added recognition plots, paired encoder/exposure forest plots, cosine curves, linkage comparison, deterministic storage scaling, and both encoder threshold-target plots. Error bars distinguish conditional intervals from key standard deviations.
7. Added six published references and a scope-comparison table. Retained existing bibliography, including the 2026 benchmark.
8. Preserved all original experimental evidence, headline metrics, official/custom protocol separation, and unmeasured hardware limitations.

Final format: IEEEtran journal, 17 pages, 15 tables, 13 figures, 38 references. No manuscript was published or submitted. The source package is the revision checkpoint.


## Final structural revision, 12 October 2026
All subsection and sub-subsection headings have been removed; seven main section headings remain. Contributions are integrated into a continuous introduction spanning more than one page. The methodology is a monochrome vector flowchart with standard input/output, process, decision and terminator shapes. Regenerate it with `python build_methodology_bw.py` (also called by build_extended_figures.py). All experimental metrics, 38 references, 15 tables and 13 figures are retained. Final length: 17 pages.


## Compact layout revision, 13 October 2026
The final manuscript is 17 pages. The methodology is a compact overview rather than a detailed execution flow. Comparison plots use single-column layouts where appropriate; remaining panel heights and float gaps are reduced while preserving readable legends and intervals. All numerical evidence and 13 figures/15 tables are retained. Run build_encoder_figures.py followed by build_extended_figures.py to recreate the current figures; the latter invokes build_compact_comparisons.py and build_methodology_bw.py.
