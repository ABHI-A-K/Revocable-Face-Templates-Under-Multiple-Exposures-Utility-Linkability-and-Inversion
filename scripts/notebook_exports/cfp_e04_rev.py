# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 1
import json, hashlib, zipfile, shutil, platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display, FileLink
except ImportError:
    def display(value):
        print(value)

    def FileLink(value):
        return value


# ============================================================
# CHANGE ONLY THIS PATH: paste the uploaded dataset folder path.
# The path below is an example; use Kaggle's actual copied path.
# ============================================================
INPUT_ROOT = Path("/kaggle/input/datasets/abhiak0709/cfp-e03-best-template-dimension")

EXPERIMENT = "CFP-E04-REV"
WORK_ROOT = (
    Path("/kaggle/working")
    if Path("/kaggle/working").exists()
    else Path.cwd()
)

DIMENSIONS = [512, 128]
SEEDS = list(range(24001, 24011))
TARGET_FAR = 0.001


def sha(path):
    """Calculate a file fingerprint to verify input consistency."""
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def find_extracted_folder(folder_name, required_files):
    """
    Locate one extracted folder inside the selected dataset.
    Also supports selecting the required folder itself.
    """
    candidates = []

    if INPUT_ROOT.name == folder_name:
        candidates.append(INPUT_ROOT)

    candidates.extend(
        path
        for path in INPUT_ROOT.rglob(folder_name)
        if path.is_dir()
    )

    candidates = sorted(set(path.resolve() for path in candidates))

    # A matching name alone is insufficient: check its contents.
    valid = [
        folder
        for folder in candidates
        if all((folder / name).is_file() for name in required_files)
    ]

    if len(valid) != 1:
        found = "\n".join(str(path) for path in candidates) or "None"
        raise RuntimeError(
            f"Expected exactly one complete '{folder_name}' folder, "
            f"found {len(valid)}.\n"
            f"Folders with that name:\n{found}\n\n"
            "Set INPUT_ROOT to the dataset folder containing both "
            "extracted E03 folders. Remove duplicate copies if present."
        )

    return valid[0]


# ------------------------------------------------------------
# 1. Check the dataset path.
# ------------------------------------------------------------
if not INPUT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Dataset folder does not exist:\n{INPUT_ROOT}\n\n"
        "In Kaggle's Input panel, copy the dataset folder path "
        "and paste it into INPUT_ROOT."
    )


# ------------------------------------------------------------
# 2. Locate the already-extracted results and template cache.
#    No ZIP upload or extraction is required.
# ------------------------------------------------------------
E03 = find_extracted_folder(
    "CFP-E03-DIM-results",
    required_files=[
        "manifest.json",
        "experiment_record.json",
        *[
            f"bits_{bits}/same_key_pair_scores_all_keys.csv"
            for bits in DIMENSIONS
        ],
        *[
            f"bits_{bits}/tar_at_far_1e-3_fold_results_all_keys.csv"
            for bits in DIMENSIONS
        ],
    ],
)

CACHE = find_extracted_folder(
    "CFP-E03-DIM-protected-template-cache",
    required_files=[
        "manifest.json",
        "aligned_pair_side_metadata.csv",
        "unique_aligned_metadata.csv",
        *[
            f"bits_{bits}/templates_packed.npy"
            for bits in DIMENSIONS
        ],
        *[
            f"bits_{bits}/config.json"
            for bits in DIMENSIONS
        ],
    ],
)


# ------------------------------------------------------------
# 3. Create a fresh output folder.
#    Inputs remain in Kaggle's read-only dataset directory.
# ------------------------------------------------------------
RUN_DIR = WORK_ROOT / (
    EXPERIMENT + "-"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
)

STAGE = RUN_DIR / "inputs"
OUT = RUN_DIR / "results"

STAGE.mkdir(parents=True, exist_ok=False)
OUT.mkdir()


# ------------------------------------------------------------
# 4. Record provenance for extracted inputs.
#    The next cell verifies individual files against manifests.
#    These are manifest hashes, not original ZIP hashes.
# ------------------------------------------------------------
inputs = {
    "input_format": "extracted_directories",
    "dataset_root": str(INPUT_ROOT),
    "e03_results_directory": str(E03),
    "e03_cache_directory": str(CACHE),
    "e03_results_manifest_sha256": sha(E03 / "manifest.json"),
    "e03_cache_manifest_sha256": sha(CACHE / "manifest.json"),
    "e03_experiment_record_sha256": sha(
        E03 / "experiment_record.json"
    ),
}

print("Extracted inputs found successfully.")
print("E03 results :", E03)
print("E03 cache   :", CACHE)
print("Output      :", OUT)
print("Bit lengths :", DIMENSIONS)


# Cell 3
for root in [E03, CACHE]:
    for entry in json.loads((root / "manifest.json").read_text()):
        path = (root / entry["path"]).resolve()
        assert path.is_relative_to(root.resolve()), "Unsafe manifest path"
        assert sha(path) == entry["sha256"], f"Hash mismatch: {path.name}"
record = json.loads((E03 / "experiment_record.json").read_text())
assert record["key_seeds"] == SEEDS
meta = pd.read_csv(CACHE / "aligned_pair_side_metadata.csv")
unique_meta = pd.read_csv(CACHE / "unique_aligned_metadata.csv")
assert sha(CACHE / "aligned_pair_side_metadata.csv") == record["cache_audit"]["pair_metadata_sha256"]
assert sha(CACHE / "unique_aligned_metadata.csv") == record["cache_audit"]["unique_metadata_sha256"]
assert len(meta) == 14000 and len(unique_meta) == 5890
assert np.array_equal(meta.bin_index, np.arange(14000))
assert np.array_equal(meta.pair_id, np.repeat(np.arange(1, 7001), 2))
assert np.array_equal(meta.side, np.tile(["a", "b"], 7000))
a = meta.iloc[::2].reset_index(drop=True); b = meta.iloc[1::2].reset_index(drop=True)
assert np.array_equal(a.label, b.label) and np.array_equal(a.fold, b.fold)
labels = a.label.to_numpy(); folds = a.fold.to_numpy(); ids = a.pair_id.to_numpy()
ia = a.unique_image_id.to_numpy(); ib = b.unique_image_id.to_numpy()
assert set(labels) == {0, 1} and set(folds) == set(range(1, 11))
assert min(ia.min(), ib.min()) >= 0 and max(ia.max(), ib.max()) < 5890
for f in range(1, 11):
    assert ((folds == f) & (labels == 1)).sum() == 350
    assert ((folds == f) & (labels == 0)).sum() == 350
print("E03 archive hashes, metadata and 10-fold protocol verified.")


# Cell 5
def similarity(x, y):
    return np.mean(x == y, axis=1, dtype=np.float64)

def calibrate(scores, y):
    imp = np.sort(scores[y == 0])[::-1]
    budget = int(np.floor(TARGET_FAR * len(imp)))
    threshold = float(np.nextafter(imp[budget], np.inf))
    accepted = int((imp >= threshold).sum())
    assert accepted / len(imp) <= TARGET_FAR
    return threshold, accepted, len(imp)

# A tie at the budget boundary must be rejected as a group.
t, fp, den = calibrate(np.array([.9, .9, .8]), np.zeros(3, dtype=int))
assert t > .9 and fp == 0


# Cell 7
control_rows, rev_rows, recheck_rows = [], [], []
for bits in DIMENSIONS:
    cfg = json.loads((CACHE / f"bits_{bits}/config.json").read_text())
    assert cfg["bits"] == bits and cfg["seeds"] == SEEDS and cfg["bitorder"] == "big"
    assert cfg["input_cache_audit"] == record["cache_audit"]
    packed = np.load(CACHE / f"bits_{bits}/templates_packed.npy", allow_pickle=False)
    assert packed.dtype == np.uint8 and packed.shape == (10, 5890, bits // 8)
    templates = np.unpackbits(packed, axis=-1, count=bits, bitorder="big").astype(bool)
    ref = pd.read_csv(E03 / f"bits_{bits}/same_key_pair_scores_all_keys.csv")
    ref_cal = pd.read_csv(E03 / f"bits_{bits}/tar_at_far_1e-3_fold_results_all_keys.csv")
    thresholds = np.empty((10, 10), dtype=np.float64)
    for k, seed in enumerate(SEEDS):
        scores = similarity(templates[k, ia], templates[k, ib])
        expected = ref[ref.key_seed == seed].sort_values("pair_id")
        assert np.array_equal(expected.pair_id, ids)
        assert np.array_equal(expected.label, labels) and np.array_equal(expected.fold, folds)
        assert np.allclose(expected.hamming_similarity, scores, atol=1e-12, rtol=0)
        for f in range(1, 11):
            train = folds != f; test = folds == f
            t, train_fp, train_n = calibrate(scores[train], labels[train])
            prior = ref_cal[(ref_cal.key_seed == seed) & (ref_cal.fold == f)].iloc[0]
            assert abs(t - prior.threshold) < 1e-14
            pred = scores[test] >= t; y = labels[test]
            tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
            assert tp == prior.tp and fp == prior.fp
            thresholds[k, f-1] = t
            control_rows.append(dict(bits=bits, key=seed, fold=f, threshold=t,
                training_fp=train_fp, training_impostors=train_n, tp=tp, genuine=int((y==1).sum()),
                fp=fp, impostors=int((y==0).sum())))
        recheck_rows.append(dict(bits=bits, key=seed, scores_reproduced=7000, thresholds_reproduced=10))
    # Only genuine pairs test stolen templates from the enrolled person.
    g = labels == 1; ga = ia[g]; gb = ib[g]; gf = folds[g]; gp = ids[g]
    score_chunks = []
    for new in range(10):
        threshold_per_trial = thresholds[new, gf-1]
        for old in range(10):
            if old == new: continue
            scenarios = {
                "primary_old_b_new_a": (gb, ga),
                "reverse_capture_old_a_new_b": (ga, gb),
                "same_image_a": (ga, ga),
                "same_image_b": (gb, gb),
            }
            for scenario, (old_ids, new_ids) in scenarios.items():
                s = similarity(templates[old, old_ids], templates[new, new_ids])
                accepted = s >= threshold_per_trial
                score_chunks.append(pd.DataFrame(dict(bits=bits, old_key=SEEDS[old], new_key=SEEDS[new],
                    scenario=scenario, pair_id=gp, fold=gf, similarity=s,
                    threshold=threshold_per_trial, accepted=accepted.astype(np.int8))))
                for f in range(1, 11):
                    mask = gf == f
                    rev_rows.append(dict(bits=bits, old_key=SEEDS[old], new_key=SEEDS[new], scenario=scenario,
                        fold=f, threshold=thresholds[new, f-1], accepted=int(accepted[mask].sum()),
                        trials=int(mask.sum()), mean_similarity=float(s[mask].mean())))
    pd.concat(score_chunks, ignore_index=True).to_csv(OUT / f"revoked_scores_{bits}.csv.gz", index=False, compression="gzip")
    del score_chunks, templates
    print(f"{bits} bits: E03 scores/thresholds/counts reproduced; all 90 renewals evaluated.", flush=True)
controls = pd.DataFrame(control_rows); rev = pd.DataFrame(rev_rows)
controls.to_csv(OUT / "legitimate_new_key_fold_controls.csv", index=False)
rev.to_csv(OUT / "revoked_acceptance_by_key_transition_and_fold.csv", index=False)
pd.DataFrame(recheck_rows).to_csv(OUT / "e03_reproduction_audit.csv", index=False)
transitions = rev.groupby(["bits", "scenario", "old_key", "new_key"], as_index=False)[["accepted", "trials"]].sum()
transitions["revoked_acceptance"] = transitions.accepted / transitions.trials
transitions.to_csv(OUT / "revoked_acceptance_by_key_transition.csv", index=False)


# Cell 9
summary_rows = []
for bits in DIMENSIONS:
    c = controls[controls.bits == bits]
    for scenario in transitions.scenario.unique():
        r = transitions[(transitions.bits == bits) & (transitions.scenario == scenario)]
        worst = r.loc[r.revoked_acceptance.idxmax()]
        summary_rows.append(dict(bits=bits, role="primary" if bits == 512 else "comparator", packed_bytes=bits//8,
            scenario=scenario, accepted=int(r.accepted.sum()), trials=int(r.trials.sum()),
            revoked_acceptance=float(r.accepted.sum()/r.trials.sum()),
            key_pair_min=float(r.revoked_acceptance.min()), key_pair_max=float(r.revoked_acceptance.max()),
            worst_old_key=int(worst.old_key), worst_new_key=int(worst.new_key),
            legitimate_tar=float(c.tp.sum()/c.genuine.sum()), observed_far=float(c.fp.sum()/c.impostors.sum())))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "summary.csv", index=False)
display(summary)
primary = summary[summary.scenario == "primary_old_b_new_a"]
tracker = []
brief = ["# CFP-E04-REV: execution results", "", "Unchanged stolen-template replay at the matcher; 9-fold new-key calibration and held-out fold evaluation.", ""]
for r in primary.itertuples():
    result = (f"{r.bits} bits: revoked acceptance {r.revoked_acceptance:.4%} ({r.accepted}/{r.trials}); "
              f"worst transition {r.key_pair_max:.4%}; new-key TAR {r.legitimate_tar:.4%}; observed FAR {r.observed_far:.4%}; {r.packed_bytes} B")
    interpretation = "Old-template replay was measured after key renewal. Judge alongside legitimate TAR; this does not establish resistance to learned mapping or inversion."
    tracker.append({"Experiment": f"CFP-E04-REV ({r.bits}-bit)", "Main question": "Can an unchanged old template authenticate after key renewal?",
                    "Important metric": "Revoked acceptance; new-key TAR and observed FAR", "Actual result": result, "Simple interpretation": interpretation})
    brief += [result, ""]
brief += ["Rates pool dependent repeated comparisons. No independent-trial confidence interval or zero-risk claim is made.",
          "Worst transitions are descriptive, not independently selected attacker configurations.",
          "Exact-image and reverse-capture supplementary results are in summary.csv."]
pd.DataFrame(tracker).to_csv(OUT / "tracker_rows.csv", index=False)
(OUT / "CFP-E04-REV_brief.md").write_text("\n".join(brief))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in [(axes[0], "revoked_acceptance", "Old-template acceptance (%)"), (axes[1], "legitimate_tar", "New-key genuine acceptance (%)")]:
    bars = ax.bar(primary.bits.astype(str), primary[col]*100, color=["#2563eb", "#94a3b8"])
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in primary[col]*100], padding=3)
    ax.set_title(title); ax.set_xlabel("Template bits"); ax.set_ylim(0, max(1, primary[col].max()*125))
fig.tight_layout(); fig.savefig(OUT / "revocation_and_utility.png", dpi=160); plt.show()
print("\n".join(brief))


# Cell 11
import os

provenance = dict(
    experiment=EXPERIMENT,
    completed_utc=datetime.now(timezone.utc).isoformat(),
    inputs=inputs,
    dimensions=DIMENSIONS,
    seeds=SEEDS,
    target_training_far=TARGET_FAR,
    python=platform.python_version(),
    numpy=np.__version__,
    pandas=pd.__version__,
    protocol=(
        "Frozen CFP pair folds; thresholds calibrated on 9 folds, "
        "tested on remaining fold; not identity-disjoint"
    ),
    threat=(
        "Unmodified old binary template reaches matcher; "
        "no operator-aware translation, training, inversion "
        "or version-ID gate"
    ),
    e03_reproduction=(
        "140000 same-key pair scores and "
        "200 thresholds/count sets passed"
    ),
    limitations=[
        "Repeated keys/images/pairs create dependent trials",
        "Descriptive worst transition",
        "No security claim beyond unchanged template replay",
    ],
)

# Confirm that the experiment produced results.
if not OUT.is_dir() or not (OUT / "summary.csv").is_file():
    raise FileNotFoundError(
        f"Experiment results are missing from:\n{OUT}\n"
        "Run the evaluation and summary cells before this export cell."
    )

(OUT / "experiment_record.json").write_text(
    json.dumps(provenance, indent=2),
    encoding="utf-8",
)

# Exclude the manifest itself, including on repeated runs.
manifest = [
    {
        "path": p.name,
        "bytes": p.stat().st_size,
        "sha256": sha(p),
    }
    for p in sorted(OUT.iterdir())
    if p.is_file() and p.name != "manifest.json"
]

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

# Save the ZIP directly under /kaggle/working.
archive = Path(
    shutil.make_archive(
        str(WORK_ROOT / "CFP-E04-REV-results"),
        "zip",
        root_dir=str(OUT),
    )
).resolve()

if not archive.is_file():
    raise FileNotFoundError(f"ZIP creation failed: {archive}")

print("ZIP created successfully.")
print("Location:", archive)
print(f"Size: {archive.stat().st_size / (1024**2):.2f} MB")

# Use a notebook-relative path for the clickable link.
relative_path = Path(os.path.relpath(archive, Path.cwd()))
display(FileLink(relative_path.as_posix()))


# Cell 12

