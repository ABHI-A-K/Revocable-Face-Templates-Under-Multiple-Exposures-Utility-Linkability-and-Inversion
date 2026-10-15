# Experiment order

1. **LFW E01:** retained configuration, metadata, pair scores and fold decisions are in the LFW-E01 archive directory. The original standalone extraction notebook was not found. The labelled E01 evidence wrapper checks saved decisions; the recovered LFW encoder extraction script checks the original pipeline against frozen embeddings.
2. **LFW E02 -> E03 -> E04:** RO protection, cross-key diagnostics, renewal. Supply E01 embeddings and official pair lists.
3. **LFW E05-ID -> E06 -> E07:** identity-disjoint linkage, projection comparison, inversion. Retain each notebook's own partitions and key schedule. Early non-identity-disjoint E05 results are historical and must not replace E05-ID evidence.
4. **CFP E01 R4 -> E02 -> E03-DIM -> E04-REV:** aligned-bin baseline, RO protection, length study, renewal. The separate E03/E04 reproduction notebook offers a saved-cache route. Failed earlier raw-image detection attempts are not accepted baselines.
5. **CFP PREP-R3 -> PREP-R4 -> QA01 -> E05-ID-SUBSET:** geometric mapping, human review, original-bin audit and screened attack preparation. Human decisions remain part of the evidence; software does not independently certify identity labels.
6. **CFP E06-PROJ -> E07-INV:** preserve the screened identity pool, splits, selected images and numerical conventions.
7. **CFP-ENC E01A -> E01B -> E02 -> E03 -> corrected E04:** matched extraction, protected recognition, renewal/linkage, inversion, paired identity bootstrap and threshold sensitivity. E01A caches, E01B fitted artifacts and E03 per-attack evidence are required. The E03 results ZIP is not the complete E03-evidence ZIP; regenerate missing intermediate artifacts from predecessor notebooks.
8. **LFW-ENC:** restore optional numerical assets and run `python experiments/lfw_encoder/evaluate.py`. For full extraction, place the LFW ZIP in `experiments/datasets/lfw-dataset.zip` and ONNX files in `experiments/models/`; run `extract_all.py` with arguments 0,1,2,3, then `assemble.py`, then `evaluate.py`.
9. Run root verification and figure-generation scripts.

Exact filenames appear in `notebook_inventory.json`. Notebook opening instructions and configuration cells specify predecessor archive names. Originals retain `/kaggle/input` and `/kaggle/working` paths. Attach inputs there in Kaggle or edit documented path settings. Historical PASS records are not evidence that this packaging session reran those experiments.
