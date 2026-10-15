**Windows:** extract this ZIP to a short folder such as `C:\FR`. See `WINDOWS.txt`. Folder/notebook names are shortened; `docs/PATH_MAP.csv` records the original names.

# Revocable face templates: reproduction materials

**Author: Abhishek A. K.** Companion materials for *Revocable Face Templates Under Multiple Exposures: Utility, Linkability, and Inversion* (14 October 2026 manuscript).

This research compares RO, SRHT and PCA-guided RO templates using MobileFaceNet and ResNet50 ArcFace. Recognition, revoked replay, template-only linkage, known-operator inversion, and packed-template storage are distinct outcomes.

**Status:** preparation for author review, not a published release. Saved-results verification and figure regeneration are supported. Full image-to-attack experiments were not rerun during packaging. See `docs/REPRODUCIBILITY.md` and the validation report before making a broader claim.

## Quick start

Use Python 3.12 in a fresh environment. From the extracted repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_repository.py
python scripts/reproduce_paper.py
```

The first script checks file integrity, clean notebooks, available identity assignments and historical/fresh-key exclusions, and manuscript numerical consistency. The second rebuilds figures and shared metric macros in `reproduction/`; it does not rerun attacks. To compile the paper, run pdfLaTeX, BibTeX and pdfLaTeX twice on `reproduction/main.tex`.

## Repository structure

| Location | Contents |
|---|---|
| `notebooks/` | Cleaned original experiment notebooks and two labelled replay wrappers |
| `scripts/` | Verification, figure generation, notebook execution, hashes and Python exports |
| `experiments/lfw_encoder/` | Original extraction, assembly and recognition evaluation scripts |
| `configs/` | Recorded encoder configurations |
| `protocols/index.json` | Exact paths to partition, mapping, key-schedule and audit evidence |
| `results/archive/` | Preserved machine-readable historical results, thresholds and available trial records |
| `reproduction/` | Latest manuscript source, curated data, plotting scripts and numerical validators |
| `environment/` | Recorded experiment environments, separated from analysis pins |
| `models/` and `datasets/` | Input instructions and exact recorded hashes; no photographs or ONNX weights |
| `docs/` | Execution order, results map, limitations, release checklist and notebook inventory |

## Optional numerical replay assets

Extract `Face_Template_Replay_Assets.zip` **inside this repository**, preserving its `experiments/` and `results/` paths. It supplies available frozen embeddings, templates, operators, PCA parameters and score arrays. These are sensitive derived biometric data; review them separately before choosing to publish. The default `.gitignore` excludes binary caches. No face photos, aligned crop chunks or ONNX weights are distributed.

With the optional assets restored:

```bash
python experiments/lfw_encoder/evaluate.py
```

This replays LFW encoder recognition and overwrites its results; use a working copy. It refits PCA, so an excluded damaged PCA cache is regenerated. Full extraction additionally requires the recorded external dataset, three model files, and strict runtime matching.

## Full experiment execution

Follow `docs/EXECUTION_ORDER.md`. Original notebooks retain Kaggle paths and predecessor-artifact dependencies. Configure those inputs before Run All. They are archived scientific code, not a newly certified portable GPU pipeline. `scripts/run_notebook.py` executes a configured notebook and saves a new executed copy. Python exports cover the same stages without claiming a new implementation.

## Expected outputs and boundaries

Figures, tables and metric checks reproduce the retained evidence. Original MobileFaceNet runs and matched CFP encoder replication remain separate. The LFW encoder extension is recognition only. Packed references require 16/64 bytes versus 2048 bytes for raw features; this is not an on-device latency or energy result. The fresh operator is hidden during reconstruction, and attacks submit digital embeddings rather than camera images.

## Before uploading

Read `docs/RELEASE_CHECKLIST.md`. The LICENSE currently retains rights pending your license choice. CITATION.cff contains no invented URL or DOI. Nothing has been uploaded to GitHub. Add your real repository URL and versioned release before citing it in the paper.

## Validation completed

The packaging report records passing manuscript consistency checks, figure regeneration, and LFW encoder recognition replay from frozen embeddings. See `verification/lfw_replay_comparison.json` and `docs/COVERAGE.md` for exact scope. Full attacks and image extraction were not rerun.

## Compact download

This GitHub ZIP excludes the large trial-level files listed in `docs/DETAILED_EVIDENCE_NOT_INCLUDED.json`. Curated data supporting all manuscript tables and figures are included. The separate 395 MB replay-assets bundle could not be saved in this session and is not part of this download; the recognition-replay instructions require those caches or their regeneration. The recorded replay check was completed before packaging.
