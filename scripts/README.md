# Execution tools

`verify_repository.py` checks retained evidence and numerical consistency; `reproduce_paper.py` regenerates figures and shared metric macros. `run_notebook.py` runs a configured notebook and saves its executed copy. `hash_file.py` checks input bytes.

`notebook_exports/` contains Python exports covering preprocessing, feature extraction, protection, recognition, renewal, linkage, reconstruction and bootstrap. These preserve notebook code and may use IPython calls; use `ipython PATH` after preparing inputs. They are not a newly refactored backend. Original reusable LFW scripts are in `experiments/lfw_encoder/`; audit scripts remain beside their archive evidence.
