# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 3
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_ID = "LFW-E06-PROJ-projection-comparison"
EXPERIMENT_TITLE = "RO vs SRHT/Hadamard vs PCA-Guided Keyed Projection"
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
E05_ID_BASE_NOTEBOOK_SHA256 = (
    "2d2e38a8c21850b87d7651d13b1d40711bfe5112b29b21fe3a812df4ca1c00db"
)

OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Leave these as None for automatic discovery. Set a Path only if needed.
MANUAL_LFW_DATASET_ROOT = None
MANUAL_CACHE_DIR = None

EMBEDDING_DIMENSION = 512
PCA_DIMENSION = 256
TEMPLATE_BITS = 128
METHODS = ["RO", "SRHT", "PCA_GUIDED_RO"]

ORIGINAL_RO_SEEDS = [2026081101, 2026081102, 2026081103, 2026081104, 2026081105]
E05_EXTENSION_RO_SEEDS = list(range(2026081106, 2026081121))
BASE_KEY_SEEDS = ORIGINAL_RO_SEEDS + E05_EXTENSION_RO_SEEDS

EXPOSURE_COUNTS = [1, 2, 5, 10]
MAX_EXPOSURES = 10
KEY_TRIAL_RANDOM_SEED = 2026082105
ATTACK_MODEL_RANDOM_SEED = 2026082106
BOOTSTRAP_RANDOM_SEED = 2026082107
BOOTSTRAP_REPETITIONS = 5000

PCA_CALIBRATION_RANDOM_SEED = 2026082601
PCA_CALIBRATION_TARGET_IMAGES = 1024
IDENTITY_SPLIT_RANDOM_SEED = 2026082501
PAIR_SAMPLING_RANDOM_SEED = 2026082502
IDENTITY_FOLDS = 10
PAIRS_PER_CLASS_PER_FOLD = 300
MAX_GENUINE_PAIRS_PER_IDENTITY = 10

TARGET_FAR = 1e-3
UTILITY_NONINFERIORITY_TOLERANCE = 0.005
SECURITY_AUC_TOLERANCE = 0.01
SAVE_FULL_OOF_SCORES = False
CPU_BENCHMARK_SAMPLES = 300

EXPECTED_SOURCE_TOTAL_PAIRS = 6000
EXPECTED_SOURCE_GENUINE_PAIRS = 3000
EXPECTED_SOURCE_IMPOSTOR_PAIRS = 3000
EXPECTED_UNIQUE_IMAGES = 7701
EXPECTED_VALID_EMBEDDINGS = 7682
EXPECTED_EVALUATION_PAIRS = IDENTITY_FOLDS * PAIRS_PER_CLASS_PER_FOLD * 2

if len(BASE_KEY_SEEDS) != 20 or len(set(BASE_KEY_SEEDS)) != 20:
    raise RuntimeError("E06 requires the unchanged 20 unique E05 key seeds.")
if not (TEMPLATE_BITS <= PCA_DIMENSION <= EMBEDDING_DIMENSION):
    raise RuntimeError("Expected 128 <= PCA dimension <= 512.")

RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
package_versions = {}
for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "torch"]:
    try:
        package_versions[name] = version(name)
    except PackageNotFoundError:
        package_versions[name] = "NOT INSTALLED"

runtime_audit = {
    "python": sys.version.split()[0],
    "operating_system": platform.platform(),
    "packages": package_versions,
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "batch_generation_device": COMPUTE_DEVICE,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
experiment_config = {
    "experiment_id": EXPERIMENT_ID,
    "base_e05_id_notebook_sha256": E05_ID_BASE_NOTEBOOK_SHA256,
    "methods": METHODS,
    "dimensions": {"embedding": 512, "pca": 256, "template_bits": 128},
    "base_key_seeds": BASE_KEY_SEEDS,
    "exposure_counts": EXPOSURE_COUNTS,
    "identity_folds": IDENTITY_FOLDS,
    "pairs_per_class_per_fold": PAIRS_PER_CLASS_PER_FOLD,
    "pca_calibration_target_images": PCA_CALIBRATION_TARGET_IMAGES,
    "target_far": TARGET_FAR,
    "save_full_oof_scores": SAVE_FULL_OOF_SCORES,
}
with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as file:
    json.dump(experiment_config, file, indent=2)

print("Experiment:", EXPERIMENT_ID)
print("Output:", OUTPUT_DIR)
print("Batch-generation device:", COMPUTE_DEVICE)
print(json.dumps(runtime_audit, indent=2))


# Cell 5
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_key(person, image_number):
    number = int(float(image_number))
    person = str(person)
    return f"{person}/{person}_{number:04d}.jpg"


def key_from_path(value):
    path = Path(str(value))
    return f"{path.parent.name}/{path.name}"


def find_lfw_root():
    roots = (
        [Path(MANUAL_LFW_DATASET_ROOT)]
        if MANUAL_LFW_DATASET_ROOT is not None
        else sorted({path.parent for path in Path("/kaggle/input").rglob("pairs.csv")})
    )
    valid = []
    for root in roots:
        image_roots = [
            root / "lfw-deepfunneled" / "lfw-deepfunneled",
            root / "lfw-deepfunneled",
        ]
        image_root = next((item for item in image_roots if item.is_dir()), None)
        if (root / "pairs.csv").is_file() and image_root is not None:
            valid.append((root, image_root))
    if not valid:
        raise FileNotFoundError(
            "LFW pairs.csv and deep-funnelled images were not found. "
            "Attach jessicali9530/lfw-dataset or set MANUAL_LFW_DATASET_ROOT."
        )
    if len(valid) > 1:
        print("Multiple LFW inputs found; using:", valid[0][0])
    return valid[0]


def find_cache_dir():
    candidates = (
        [Path(MANUAL_CACHE_DIR)]
        if MANUAL_CACHE_DIR is not None
        else sorted(
            path.parent
            for path in Path("/kaggle/input").rglob("raw_embeddings.npy")
            if path.with_name("embedding_metadata.csv").is_file()
        )
    )
    valid = []
    for candidate in candidates:
        emb_path = candidate / "raw_embeddings.npy"
        meta_path = candidate / "embedding_metadata.csv"
        try:
            shape = np.load(emb_path, mmap_mode="r").shape
            rows = len(pd.read_csv(meta_path, usecols=[0]))
        except Exception as error:
            print("Skipping cache:", candidate, "|", error)
            continue
        if shape == (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION) and rows == EXPECTED_UNIQUE_IMAGES:
            valid.append(candidate)
    if not valid:
        raise FileNotFoundError(
            "No complete 7701 x 512 cache was found. Attach raw_embeddings.npy "
            "with embedding_metadata.csv, or set MANUAL_CACHE_DIR."
        )
    preferred = [item for item in valid if "cache" in str(item).lower()]
    selected = preferred[0] if preferred else valid[0]
    if len(valid) > 1:
        print("Multiple valid caches found; using:", selected)
    return selected


DATASET_ROOT, IMAGE_ROOT = find_lfw_root()
PAIRS_PATH = DATASET_ROOT / "pairs.csv"
CACHE_DIR = find_cache_dir()

source_pairs_df = pd.read_csv(PAIRS_PATH)
source_rows = []
for index, row in source_pairs_df.iterrows():
    genuine = pd.isna(row["Unnamed: 3"])
    if genuine:
        person_a = str(row["name"])
        person_b = person_a
        key_a = image_key(person_a, row["imagenum1"])
        key_b = image_key(person_b, row["imagenum2"])
        label = 1
    else:
        person_a = str(row["name"])
        person_b = str(row["imagenum2"])
        key_a = image_key(person_a, row["imagenum1"])
        key_b = image_key(person_b, row["Unnamed: 3"])
        label = 0
    source_rows.append({
        "source_pair_id": index + 1,
        "source_fold": index // 600 + 1,
        "label": label,
        "person_a": person_a,
        "person_b": person_b,
        "image_a_key": key_a,
        "image_b_key": key_b,
    })

source_pair_manifest_df = pd.DataFrame(source_rows)
unique_image_keys = sorted(
    set(source_pair_manifest_df["image_a_key"])
    | set(source_pair_manifest_df["image_b_key"])
)
missing_images = [key for key in unique_image_keys if not (IMAGE_ROOT / key).is_file()]
source_inventory = {
    "official_pairs_used_for_cache_index_only": len(source_pair_manifest_df),
    "official_genuine_pairs": int((source_pair_manifest_df["label"] == 1).sum()),
    "official_impostor_pairs": int((source_pair_manifest_df["label"] == 0).sum()),
    "unique_cache_images": len(unique_image_keys),
    "missing_images": len(missing_images),
}
expected_inventory = {
    "official_pairs_used_for_cache_index_only": EXPECTED_SOURCE_TOTAL_PAIRS,
    "official_genuine_pairs": EXPECTED_SOURCE_GENUINE_PAIRS,
    "official_impostor_pairs": EXPECTED_SOURCE_IMPOSTOR_PAIRS,
    "unique_cache_images": EXPECTED_UNIQUE_IMAGES,
    "missing_images": 0,
}
if source_inventory != expected_inventory:
    raise RuntimeError(f"LFW inventory failed: {source_inventory} != {expected_inventory}")

embeddings_path = CACHE_DIR / "raw_embeddings.npy"
metadata_path = CACHE_DIR / "embedding_metadata.csv"
embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
metadata_df = pd.read_csv(metadata_path).copy()

if embeddings.shape != (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION):
    raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
if len(metadata_df) != EXPECTED_UNIQUE_IMAGES:
    raise RuntimeError("Metadata row count does not match the embedding matrix.")

if "image_id" not in metadata_df:
    metadata_df["image_id"] = np.arange(len(metadata_df))
metadata_df["image_id"] = metadata_df["image_id"].astype(int)
if metadata_df["image_id"].tolist() != list(range(EXPECTED_UNIQUE_IMAGES)):
    raise RuntimeError("The cache image_id column must retain order 0..7700.")
if "image_path" not in metadata_df:
    raise RuntimeError("embedding_metadata.csv must contain image_path.")

metadata_df["image_key"] = metadata_df["image_path"].map(key_from_path)
if metadata_df["image_key"].tolist() != unique_image_keys:
    raise RuntimeError("Cache order does not match the frozen LFW image order.")
if "identity" not in metadata_df:
    metadata_df["identity"] = metadata_df["image_key"].str.split("/").str[0]
metadata_df["identity"] = metadata_df["identity"].astype(str)

finite_mask = np.isfinite(embeddings).all(axis=1)
valid_embedding_mask = (
    finite_mask & metadata_df["status"].eq("valid").to_numpy()
    if "status" in metadata_df
    else finite_mask
)
valid_count = int(valid_embedding_mask.sum())
if valid_count != EXPECTED_VALID_EMBEDDINGS:
    raise RuntimeError(
        f"Expected {EXPECTED_VALID_EMBEDDINGS} valid embeddings; found {valid_count}."
    )

cache_audit = {
    "dataset_root": str(DATASET_ROOT),
    "cache_dir": str(CACHE_DIR),
    "embedding_shape": list(embeddings.shape),
    "valid_embeddings": valid_count,
    "invalid_or_failed_embeddings": int((~valid_embedding_mask).sum()),
    "raw_embeddings_sha256": sha256_file(embeddings_path),
    "metadata_sha256": sha256_file(metadata_path),
}
source_pair_manifest_df.to_csv(
    OUTPUT_DIR / "official_pairs_cache_index_only.csv", index=False
)
metadata_df[["image_id", "image_key", "identity"]].assign(
    valid_embedding=valid_embedding_mask
).to_csv(OUTPUT_DIR / "embedding_cache_index.csv", index=False)
with open(OUTPUT_DIR / "cache_audit.json", "w", encoding="utf-8") as file:
    json.dump(cache_audit, file, indent=2)

print(json.dumps(source_inventory, indent=2))
print(json.dumps(cache_audit, indent=2))


# Cell 7
def l2_normalize_rows(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


valid_rows_df = metadata_df.loc[
    valid_embedding_mask, ["image_id", "image_key", "identity"]
].copy()
identity_stats_df = (
    valid_rows_df.groupby("identity", sort=True)
    .size().rename("valid_images").reset_index()
)
identity_stats_df["raw_genuine_pair_capacity"] = (
    identity_stats_df["valid_images"] * (identity_stats_df["valid_images"] - 1) // 2
)
identity_stats_df["capped_genuine_pair_capacity"] = np.minimum(
    identity_stats_df["raw_genuine_pair_capacity"], MAX_GENUINE_PAIRS_PER_IDENTITY
)

cal_rng = np.random.default_rng(PCA_CALIBRATION_RANDOM_SEED)
identity_stats_df["tie_breaker"] = cal_rng.random(len(identity_stats_df))
selection_df = identity_stats_df.sort_values(
    ["capped_genuine_pair_capacity", "valid_images", "tie_breaker", "identity"],
    ascending=[True, True, True, True],
)

calibration_identities = []
calibration_image_count = 0
for row in selection_df.itertuples(index=False):
    calibration_identities.append(row.identity)
    calibration_image_count += int(row.valid_images)
    if calibration_image_count >= PCA_CALIBRATION_TARGET_IMAGES:
        break

calibration_identity_set = set(calibration_identities)
evaluation_identity_set = set(identity_stats_df["identity"]) - calibration_identity_set
calibration_rows_df = valid_rows_df[
    valid_rows_df["identity"].isin(calibration_identity_set)
].copy()
evaluation_rows_df = valid_rows_df[
    valid_rows_df["identity"].isin(evaluation_identity_set)
].copy()

if calibration_identity_set & evaluation_identity_set:
    raise RuntimeError("Calibration/evaluation identity leakage detected.")
if len(calibration_rows_df) < PCA_DIMENSION + 1:
    raise RuntimeError("Not enough calibration images for 256-D PCA.")

identity_partition_df = identity_stats_df.drop(columns=["tie_breaker"]).copy()
identity_partition_df["partition"] = np.where(
    identity_partition_df["identity"].isin(calibration_identity_set),
    "pca_calibration_only",
    "evaluation_only",
)
identity_partition_df.to_csv(OUTPUT_DIR / "identity_partition.csv", index=False)

partition_audit = {
    "calibration_identities": len(calibration_identity_set),
    "calibration_valid_images": len(calibration_rows_df),
    "evaluation_identities": len(evaluation_identity_set),
    "evaluation_valid_images": len(evaluation_rows_df),
    "calibration_evaluation_identity_overlap": 0,
    "selection_rule": "lowest capped genuine-pair capacity first; complete identities",
}
with open(OUTPUT_DIR / "identity_partition_audit.json", "w", encoding="utf-8") as file:
    json.dump(partition_audit, file, indent=2)
print(json.dumps(partition_audit, indent=2))


# Cell 9
identity_to_images = {
    identity: sorted(group["image_id"].astype(int).tolist())
    for identity, group in evaluation_rows_df.groupby("identity", sort=True)
}
image_id_to_key = metadata_df.set_index("image_id")["image_key"].to_dict()

split_rng = np.random.default_rng(IDENTITY_SPLIT_RANDOM_SEED)
stats = []
for identity, image_ids in identity_to_images.items():
    count = len(image_ids)
    capacity = count * (count - 1) // 2
    stats.append({
        "identity": identity,
        "valid_images": count,
        "raw_genuine_pair_capacity": capacity,
        "capped_genuine_pair_capacity": min(capacity, MAX_GENUINE_PAIRS_PER_IDENTITY),
        "tie_breaker": float(split_rng.random()),
    })
evaluation_identity_stats_df = pd.DataFrame(stats)

fold_state = {
    fold: {"genuine_capacity": 0, "valid_images": 0, "identity_count": 0}
    for fold in range(1, IDENTITY_FOLDS + 1)
}
identity_to_fold = {}

multi_df = evaluation_identity_stats_df[
    evaluation_identity_stats_df["capped_genuine_pair_capacity"] > 0
].sort_values(
    ["capped_genuine_pair_capacity", "valid_images", "tie_breaker", "identity"],
    ascending=[False, False, True, True],
)
for row in multi_df.itertuples(index=False):
    fold = min(
        fold_state,
        key=lambda item: (
            fold_state[item]["genuine_capacity"],
            fold_state[item]["valid_images"],
            fold_state[item]["identity_count"],
            item,
        ),
    )
    identity_to_fold[row.identity] = fold
    fold_state[fold]["genuine_capacity"] += int(row.capped_genuine_pair_capacity)
    fold_state[fold]["valid_images"] += int(row.valid_images)
    fold_state[fold]["identity_count"] += 1

single_df = evaluation_identity_stats_df[
    evaluation_identity_stats_df["capped_genuine_pair_capacity"] == 0
].sort_values(["tie_breaker", "identity"])
for row in single_df.itertuples(index=False):
    fold = min(
        fold_state,
        key=lambda item: (
            fold_state[item]["identity_count"],
            fold_state[item]["valid_images"],
            item,
        ),
    )
    identity_to_fold[row.identity] = fold
    fold_state[fold]["valid_images"] += int(row.valid_images)
    fold_state[fold]["identity_count"] += 1

identity_fold_assignment_df = evaluation_identity_stats_df.drop(
    columns=["tie_breaker"]
).copy()
identity_fold_assignment_df["fold"] = identity_fold_assignment_df["identity"].map(
    identity_to_fold
)
identity_fold_assignment_df = identity_fold_assignment_df.sort_values(
    ["fold", "identity"]
).reset_index(drop=True)

pair_rng = np.random.default_rng(PAIR_SAMPLING_RANDOM_SEED)
pair_rows = []
fold_rows = []

for fold in range(1, IDENTITY_FOLDS + 1):
    fold_identities = sorted(
        identity_fold_assignment_df.loc[
            identity_fold_assignment_df["fold"] == fold, "identity"
        ].tolist()
    )

    # Round-robin sampling prevents one identity from dominating genuine pairs.
    genuine_queues = {}
    for identity in fold_identities:
        candidates = list(combinations(identity_to_images[identity], 2))
        pair_rng.shuffle(candidates)
        if candidates:
            genuine_queues[identity] = candidates[:MAX_GENUINE_PAIRS_PER_IDENTITY]

    identity_order = sorted(genuine_queues)
    pair_rng.shuffle(identity_order)
    selected_genuine = []
    round_index = 0
    while len(selected_genuine) < PAIRS_PER_CLASS_PER_FOLD:
        added = 0
        for identity in identity_order:
            queue = genuine_queues[identity]
            if round_index < len(queue):
                image_a_id, image_b_id = queue[round_index]
                if bool(pair_rng.integers(0, 2)):
                    image_a_id, image_b_id = image_b_id, image_a_id
                selected_genuine.append((identity, image_a_id, image_b_id))
                added += 1
                if len(selected_genuine) == PAIRS_PER_CLASS_PER_FOLD:
                    break
        if added == 0:
            break
        round_index += 1

    if len(selected_genuine) < PAIRS_PER_CLASS_PER_FOLD:
        raise RuntimeError(
            f"Fold {fold} has only {len(selected_genuine)} genuine pairs. "
            "Reduce PCA_CALIBRATION_TARGET_IMAGES if necessary."
        )

    impostor_candidates = list(combinations(fold_identities, 2))
    pair_rng.shuffle(impostor_candidates)
    if len(impostor_candidates) < PAIRS_PER_CLASS_PER_FOLD:
        raise RuntimeError(f"Fold {fold} lacks enough impostor identity pairs.")

    for identity, image_a_id, image_b_id in selected_genuine:
        pair_rows.append({
            "fold": fold,
            "pair_type": "mated_same_identity",
            "label": 1,
            "person_a": identity,
            "person_b": identity,
            "image_a_id": image_a_id,
            "image_b_id": image_b_id,
            "image_a_key": image_id_to_key[image_a_id],
            "image_b_key": image_id_to_key[image_b_id],
        })

    for identity_a, identity_b in impostor_candidates[:PAIRS_PER_CLASS_PER_FOLD]:
        image_a_id = int(pair_rng.choice(identity_to_images[identity_a]))
        image_b_id = int(pair_rng.choice(identity_to_images[identity_b]))
        if bool(pair_rng.integers(0, 2)):
            identity_a, identity_b = identity_b, identity_a
            image_a_id, image_b_id = image_b_id, image_a_id
        pair_rows.append({
            "fold": fold,
            "pair_type": "non_mated_different_identity",
            "label": 0,
            "person_a": identity_a,
            "person_b": identity_b,
            "image_a_id": image_a_id,
            "image_b_id": image_b_id,
            "image_a_key": image_id_to_key[image_a_id],
            "image_b_key": image_id_to_key[image_b_id],
        })

    fold_rows.append({
        "fold": fold,
        "identities": len(fold_identities),
        "valid_images": sum(len(identity_to_images[item]) for item in fold_identities),
        "genuine_pairs": PAIRS_PER_CLASS_PER_FOLD,
        "impostor_pairs": PAIRS_PER_CLASS_PER_FOLD,
    })

pair_manifest_df = pd.DataFrame(pair_rows)
pair_manifest_df.insert(0, "pair_id", np.arange(1, len(pair_manifest_df) + 1))
fold_protocol_df = pd.DataFrame(fold_rows)

fold_identity_sets = {
    fold: set(identity_fold_assignment_df.loc[
        identity_fold_assignment_df["fold"] == fold, "identity"
    ])
    for fold in range(1, IDENTITY_FOLDS + 1)
}
overlap_rows = []
for test_fold in range(1, IDENTITY_FOLDS + 1):
    test_ids = fold_identity_sets[test_fold]
    train_ids = set().union(*[
        fold_identity_sets[item]
        for item in range(1, IDENTITY_FOLDS + 1)
        if item != test_fold
    ])
    overlap_rows.append({
        "test_fold": test_fold,
        "train_identities": len(train_ids),
        "test_identities": len(test_ids),
        "train_test_identity_overlap": len(train_ids & test_ids),
        "test_overlap_with_pca_calibration": len(test_ids & calibration_identity_set),
    })
identity_overlap_audit_df = pd.DataFrame(overlap_rows)

unordered_pairs = pair_manifest_df.apply(
    lambda row: "|".join(sorted([str(row["image_a_id"]), str(row["image_b_id"])])),
    axis=1,
)
per_fold_counts = pair_manifest_df.groupby(["fold", "label"]).size().unstack()
protocol_audit = {
    "total_pairs": len(pair_manifest_df),
    "genuine_pairs": int((pair_manifest_df["label"] == 1).sum()),
    "impostor_pairs": int((pair_manifest_df["label"] == 0).sum()),
    "folds": int(pair_manifest_df["fold"].nunique()),
    "duplicate_unordered_image_pairs": int(unordered_pairs.duplicated().sum()),
    "maximum_train_test_identity_overlap": int(
        identity_overlap_audit_df["train_test_identity_overlap"].max()
    ),
    "maximum_test_overlap_with_pca_calibration": int(
        identity_overlap_audit_df["test_overlap_with_pca_calibration"].max()
    ),
}
if len(pair_manifest_df) != EXPECTED_EVALUATION_PAIRS:
    raise RuntimeError("Evaluation pair count validation failed.")
if not np.array_equal(
    per_fold_counts[[0, 1]].to_numpy(),
    np.full((IDENTITY_FOLDS, 2), PAIRS_PER_CLASS_PER_FOLD),
):
    raise RuntimeError("Every fold must contain 300 impostor and 300 genuine pairs.")
if protocol_audit["maximum_train_test_identity_overlap"] != 0:
    raise RuntimeError("Attacker train/test identity leakage detected.")
if protocol_audit["maximum_test_overlap_with_pca_calibration"] != 0:
    raise RuntimeError("PCA calibration/evaluation leakage detected.")
if protocol_audit["duplicate_unordered_image_pairs"] != 0:
    raise RuntimeError("Duplicate unordered image pairs were generated.")

identity_fold_assignment_df.to_csv(
    OUTPUT_DIR / "identity_fold_assignment.csv", index=False
)
pair_manifest_df.to_csv(
    OUTPUT_DIR / "identity_disjoint_pair_manifest.csv", index=False
)
fold_protocol_df.to_csv(
    OUTPUT_DIR / "identity_fold_protocol_summary.csv", index=False
)
identity_overlap_audit_df.to_csv(
    OUTPUT_DIR / "identity_overlap_audit.csv", index=False
)
with open(OUTPUT_DIR / "identity_disjoint_protocol_audit.json", "w", encoding="utf-8") as file:
    json.dump(protocol_audit, file, indent=2)

print(json.dumps(protocol_audit, indent=2))
display(fold_protocol_df)
display(identity_overlap_audit_df)


# Cell 11
calibration_ids = calibration_rows_df["image_id"].to_numpy(dtype=int)
valid_image_ids = np.flatnonzero(valid_embedding_mask)
raw_valid_embeddings = l2_normalize_rows(embeddings[valid_embedding_mask])
calibration_embeddings = l2_normalize_rows(embeddings[calibration_ids])

pca_started = time.perf_counter()
pca_model = PCA(
    n_components=PCA_DIMENSION,
    svd_solver="randomized",
    whiten=False,
    random_state=PCA_CALIBRATION_RANDOM_SEED,
)
pca_model.fit(calibration_embeddings)
pca_fit_seconds = time.perf_counter() - pca_started
pca_valid_embeddings = l2_normalize_rows(
    pca_model.transform(raw_valid_embeddings)
)

np.savez_compressed(
    OUTPUT_DIR / "pca_public_model.npz",
    mean=pca_model.mean_.astype(np.float32),
    components=pca_model.components_.astype(np.float32),
    explained_variance=pca_model.explained_variance_.astype(np.float32),
    explained_variance_ratio=pca_model.explained_variance_ratio_.astype(np.float32),
)
pca_audit = {
    "fit_partition": "pca_calibration_only",
    "calibration_identities": len(calibration_identity_set),
    "calibration_images": len(calibration_embeddings),
    "evaluation_identity_overlap": 0,
    "input_dimension": EMBEDDING_DIMENSION,
    "output_dimension": PCA_DIMENSION,
    "whiten": False,
    "explained_variance_ratio_sum": float(
        pca_model.explained_variance_ratio_.sum()
    ),
    "fit_seconds": pca_fit_seconds,
}
with open(OUTPUT_DIR / "pca_audit.json", "w", encoding="utf-8") as file:
    json.dump(pca_audit, file, indent=2)
print(json.dumps(pca_audit, indent=2))


# Cell 13
def derive_method_seed(base_seed, method):
    if method == "RO":
        return int(base_seed)
    digest = hashlib.sha256(
        f"LFW-E06|{method}|{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def create_seeded_ro_matrix(input_dim, output_dim, seed):
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((input_dim, output_dim))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float32)


def create_srht_state(input_dim, output_dim, seed):
    if input_dim <= 0 or (input_dim & (input_dim - 1)) != 0:
        raise ValueError("SRHT input dimension must be a power of two.")
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32), size=input_dim
    )
    indices = rng.choice(
        input_dim, size=output_dim, replace=False
    ).astype(np.int64)
    return {
        "signs": signs,
        "indices": indices,
        "scale": float(np.sqrt(input_dim / output_dim)),
    }


def fwht_torch_batch(values):
    if values.ndim != 2:
        raise ValueError("FWHT expects a two-dimensional tensor.")
    width = values.shape[1]
    if width <= 0 or (width & (width - 1)) != 0:
        raise ValueError("FWHT width must be a power of two.")
    output = values
    block = 1
    while block < width:
        shaped = output.reshape(output.shape[0], -1, block * 2)
        left = shaped[:, :, :block]
        right = shaped[:, :, block:]
        output = torch.cat(
            (left + right, left - right), dim=2
        ).reshape(output.shape[0], width)
        block *= 2
    return output


def fwht_numpy_1d(values):
    output = np.asarray(values, dtype=np.float32).copy()
    width = len(output)
    block = 1
    while block < width:
        shaped = output.reshape(-1, block * 2)
        left = shaped[:, :block].copy()
        right = shaped[:, block:].copy()
        shaped[:, :block] = left + right
        shaped[:, block:] = left - right
        block *= 2
    return output / np.sqrt(width)


def dense_hadamard_matrix(order):
    matrix = np.array([[1.0]], dtype=np.float32)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(order)


# Unit test: fast and dense Hadamard transforms must agree.
test_rng = np.random.default_rng(2026082602)
test_vector = test_rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
dense_hadamard = dense_hadamard_matrix(EMBEDDING_DIMENSION)
fwht_max_error = float(np.max(np.abs(
    fwht_numpy_1d(test_vector) - test_vector @ dense_hadamard
)))
print("FWHT dense-equivalence maximum error:", fwht_max_error)
if fwht_max_error >= 1e-4:
    raise RuntimeError("Fast Hadamard transform failed its unit test.")


# Cell 15
torch.backends.cuda.matmul.allow_tf32 = False

method_inputs = {
    "RO": raw_valid_embeddings,
    "SRHT": raw_valid_embeddings,
    "PCA_GUIDED_RO": pca_valid_embeddings,
}
biohashes_by_method = {method: {} for method in METHODS}
projection_states = {method: {} for method in METHODS}
validation_rows = []

generation_started = time.perf_counter()
for method in METHODS:
    input_values = method_inputs[method]
    input_torch = torch.from_numpy(input_values).to(COMPUTE_DEVICE)

    for base_seed in BASE_KEY_SEEDS:
        method_seed = derive_method_seed(base_seed, method)
        if COMPUTE_DEVICE == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        if method == "RO":
            state = create_seeded_ro_matrix(
                EMBEDDING_DIMENSION, TEMPLATE_BITS, method_seed
            )
            state_torch = torch.from_numpy(state).to(COMPUTE_DEVICE)
            projected = input_torch @ state_torch
            gram_error = float(np.max(np.abs(
                state.T @ state - np.eye(TEMPLATE_BITS, dtype=np.float32)
            )))
            explicit_bytes = int(state.nbytes)
            np.save(OUTPUT_DIR / f"projection_RO_key_{base_seed}.npy", state)
            del state_torch

        elif method == "SRHT":
            state = create_srht_state(
                EMBEDDING_DIMENSION, TEMPLATE_BITS, method_seed
            )
            signs_torch = torch.from_numpy(state["signs"]).to(COMPUTE_DEVICE)
            indices_torch = torch.from_numpy(state["indices"]).to(COMPUTE_DEVICE)
            mixed = fwht_torch_batch(input_torch * signs_torch)
            mixed = mixed / np.sqrt(EMBEDDING_DIMENSION)
            projected = torch.index_select(
                mixed, 1, indices_torch
            ) * state["scale"]

            equivalent = (
                state["signs"][:, None]
                * dense_hadamard[:, state["indices"]]
                * state["scale"]
            )
            target = (state["scale"] ** 2) * np.eye(
                TEMPLATE_BITS, dtype=np.float32
            )
            gram_error = float(np.max(np.abs(
                equivalent.T @ equivalent - target
            )))
            # Compact explicit state: 512 int8 signs + 128 int16 indices.
            explicit_bytes = EMBEDDING_DIMENSION + TEMPLATE_BITS * 2
            np.savez_compressed(
                OUTPUT_DIR / f"projection_SRHT_key_{base_seed}.npz",
                signs=state["signs"].astype(np.int8),
                indices=state["indices"].astype(np.int16),
                scale=np.float32(state["scale"]),
                method_seed=np.int64(method_seed),
            )
            del signs_torch, indices_torch, mixed, equivalent

        else:
            state = create_seeded_ro_matrix(
                PCA_DIMENSION, TEMPLATE_BITS, method_seed
            )
            state_torch = torch.from_numpy(state).to(COMPUTE_DEVICE)
            projected = input_torch @ state_torch
            gram_error = float(np.max(np.abs(
                state.T @ state - np.eye(TEMPLATE_BITS, dtype=np.float32)
            )))
            explicit_bytes = int(state.nbytes)
            np.save(
                OUTPUT_DIR / f"projection_PCA_GUIDED_RO_key_{base_seed}.npy",
                state,
            )
            del state_torch

        if COMPUTE_DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        valid_templates = (projected >= 0.0).cpu().numpy()
        full_templates = np.zeros(
            (len(embeddings), TEMPLATE_BITS), dtype=bool
        )
        full_templates[valid_embedding_mask] = valid_templates
        biohashes_by_method[method][base_seed] = full_templates
        projection_states[method][base_seed] = state

        packed = np.packbits(full_templates, axis=1, bitorder="big")
        template_path = (
            OUTPUT_DIR / f"templates_{method}_key_{base_seed}_packed.npy"
        )
        np.save(template_path, packed)
        validation_rows.append({
            "method": method,
            "base_key_seed": base_seed,
            "method_seed": method_seed,
            "input_dimension": input_values.shape[1],
            "template_bits": TEMPLATE_BITS,
            "valid_templates": len(valid_templates),
            "gram_max_error": gram_error,
            "overall_one_rate": float(valid_templates.mean()),
            "explicit_per_key_operator_bytes": explicit_bytes,
            "batch_generation_ms": elapsed_ms,
            "packed_template_sha256": sha256_file(template_path),
        })
        del projected

    del input_torch
    if COMPUTE_DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Completed:", method)

projection_validation_df = pd.DataFrame(validation_rows)
projection_validation_df.to_csv(
    OUTPUT_DIR / "projection_validation.csv", index=False
)
if len(projection_validation_df) != len(METHODS) * len(BASE_KEY_SEEDS):
    raise RuntimeError("Expected exactly 60 projection/template records.")
if projection_validation_df["gram_max_error"].max() >= 1e-4:
    raise RuntimeError("A projection failed its Gram-matrix validation.")

template_generation_audit = {
    "methods": METHODS,
    "keys_per_method": len(BASE_KEY_SEEDS),
    "total_template_sets": len(projection_validation_df),
    "device": COMPUTE_DEVICE,
    "total_elapsed_seconds": time.perf_counter() - generation_started,
    "raw_float32_bytes_per_template": EMBEDDING_DIMENSION * 4,
    "protected_packed_bytes_per_template": TEMPLATE_BITS // 8,
}
display(projection_validation_df.groupby("method").agg(
    mean_one_rate=("overall_one_rate", "mean"),
    worst_gram_error=("gram_max_error", "max"),
    mean_batch_ms=("batch_generation_ms", "mean"),
    explicit_key_bytes=("explicit_per_key_operator_bytes", "mean"),
))
print(json.dumps(template_generation_audit, indent=2))


# Cell 17
a_ids = pair_manifest_df["image_a_id"].to_numpy(dtype=int)
b_ids = pair_manifest_df["image_b_id"].to_numpy(dtype=int)
pair_labels = pair_manifest_df["label"].to_numpy(dtype=np.int8)
pair_folds = pair_manifest_df["fold"].to_numpy(dtype=np.int8)

if not (valid_embedding_mask[a_ids] & valid_embedding_mask[b_ids]).all():
    raise RuntimeError("An evaluation pair contains an invalid embedding.")

trial_rng = np.random.default_rng(KEY_TRIAL_RANDOM_SEED)
query_order = np.asarray(BASE_KEY_SEEDS, dtype=np.int64).copy()
trial_rng.shuffle(query_order)
key_trial_rows = []
for trial_id, query_seed in enumerate(query_order, start=1):
    remaining = np.asarray(
        [seed for seed in BASE_KEY_SEEDS if seed != int(query_seed)],
        dtype=np.int64,
    )
    trial_rng.shuffle(remaining)
    exposure_order = remaining[:MAX_EXPOSURES].tolist()
    row = {"trial_id": trial_id, "query_seed": int(query_seed)}
    row.update({
        f"exposure_seed_{position}": int(seed)
        for position, seed in enumerate(exposure_order, start=1)
    })
    key_trial_rows.append(row)

key_trials_df = pd.DataFrame(key_trial_rows)
if key_trials_df["query_seed"].nunique() != len(BASE_KEY_SEEDS):
    raise RuntimeError("Each key must serve once as query key.")
for _, row in key_trials_df.iterrows():
    exposed = [
        int(row[f"exposure_seed_{i}"])
        for i in range(1, MAX_EXPOSURES + 1)
    ]
    if len(set(exposed)) != MAX_EXPOSURES:
        raise RuntimeError("Exposure keys must be unique.")
    if int(row["query_seed"]) in exposed:
        raise RuntimeError("Fresh query key cannot be exposed.")

key_trials_df.to_csv(OUTPUT_DIR / "key_trial_schedule.csv", index=False)
display(key_trials_df)


# Cell 19
def hamming_similarity(template_a, template_b):
    return 1.0 - np.mean(np.logical_xor(template_a, template_b), axis=1)


def confusion_summary(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tar = tp / (tp + fn)
    far = fp / (fp + tn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tar + tn / (tn + fp)),
        "tar": tar, "far": far, "frr": 1.0 - tar,
    }


def best_balanced_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    balanced = 0.5 * (tpr + 1.0 - fpr)
    best = float(balanced[finite].max())
    tied = thresholds[
        finite & np.isclose(balanced, best, rtol=0, atol=1e-12)
    ]
    return float(tied.max())


def threshold_at_far(labels, scores, target_far):
    impostor = np.sort(
        np.asarray(scores)[np.asarray(labels) == 0]
    )[::-1]
    allowed = int(np.floor(target_far * len(impostor)))
    if allowed >= len(impostor):
        return float("-inf")
    return float(np.nextafter(impostor[allowed], np.inf))


def approximate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return (
        float(0.5 * (fpr[index] + fnr[index])),
        float(thresholds[index]),
    )


def bootstrap_mean_ci(values, seed, repetitions=BOOTSTRAP_REPETITIONS):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(repetitions, len(values))
    )
    means = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def same_key_metrics(labels, scores, folds, target_far):
    operating_predictions = np.full(len(labels), -1, dtype=np.int8)
    strict_predictions = np.full(len(labels), -1, dtype=np.int8)
    fold_rows = []

    for test_fold in sorted(np.unique(folds)):
        train = folds != test_fold
        test = folds == test_fold
        threshold = best_balanced_threshold(labels[train], scores[train])
        strict_threshold = threshold_at_far(
            labels[train], scores[train], target_far
        )
        operating_predictions[test] = (
            scores[test] >= threshold
        ).astype(np.int8)
        strict_predictions[test] = (
            scores[test] >= strict_threshold
        ).astype(np.int8)

        operating = confusion_summary(
            labels[test], operating_predictions[test]
        )
        strict = confusion_summary(
            labels[test], strict_predictions[test]
        )
        fold_rows.append({
            "test_fold": int(test_fold),
            "operating_threshold": threshold,
            "far_target_threshold": strict_threshold,
            **{f"operating_{key}": value for key, value in operating.items()},
            **{f"far_target_{key}": value for key, value in strict.items()},
        })

    if not np.isin(operating_predictions, [0, 1]).all():
        raise RuntimeError("Missing recognition prediction.")
    if not np.isin(strict_predictions, [0, 1]).all():
        raise RuntimeError("Missing FAR-target prediction.")

    operating = confusion_summary(labels, operating_predictions)
    strict = confusion_summary(labels, strict_predictions)
    eer, eer_threshold = approximate_eer(labels, scores)
    metrics = {
        "auc": float(roc_auc_score(labels, scores)),
        "approximate_eer": eer,
        "eer_threshold_pooled": eer_threshold,
        "cv_accuracy": operating["accuracy"],
        "cv_balanced_accuracy": operating["balanced_accuracy"],
        "cv_tar": operating["tar"],
        "cv_far": operating["far"],
        "cv_frr": operating["frr"],
        "tar_at_far_1e_3": strict["tar"],
        "observed_far_at_target": strict["far"],
        "false_accepts_at_target": strict["fp"],
        "false_rejects_at_target": strict["fn"],
        "mean_operating_threshold": float(np.mean([
            row["operating_threshold"] for row in fold_rows
        ])),
        "mean_far_target_threshold": float(np.mean([
            row["far_target_threshold"] for row in fold_rows
        ])),
        "genuine_mean": float(scores[labels == 1].mean()),
        "impostor_mean": float(scores[labels == 0].mean()),
    }
    metrics["genuine_impostor_mean_margin"] = (
        metrics["genuine_mean"] - metrics["impostor_mean"]
    )
    return metrics, fold_rows


# Cell 21
recognition_rows = []
recognition_fold_rows = []

for method in METHODS:
    for base_seed in BASE_KEY_SEEDS:
        templates = biohashes_by_method[method][base_seed]
        scores = hamming_similarity(
            templates[a_ids], templates[b_ids]
        )
        metrics, fold_rows = same_key_metrics(
            pair_labels, scores, pair_folds, TARGET_FAR
        )
        recognition_rows.append({
            "method": method,
            "base_key_seed": base_seed,
            "evaluated_pairs": len(scores),
            **metrics,
        })
        for row in fold_rows:
            recognition_fold_rows.append({
                "method": method,
                "base_key_seed": base_seed,
                **row,
            })
    print("Recognition complete:", method)

recognition_per_key_df = pd.DataFrame(recognition_rows)
recognition_fold_df = pd.DataFrame(recognition_fold_rows)
recognition_per_key_df.to_csv(
    OUTPUT_DIR / "recognition_per_key.csv", index=False
)
recognition_fold_df.to_csv(
    OUTPUT_DIR / "recognition_fold_results.csv", index=False
)

utility_metrics = [
    "tar_at_far_1e_3",
    "observed_far_at_target",
    "approximate_eer",
    "auc",
    "cv_accuracy",
    "genuine_impostor_mean_margin",
]
summary_rows = []
for method in METHODS:
    subset = recognition_per_key_df[
        recognition_per_key_df["method"] == method
    ]
    row = {"method": method, "keys": len(subset)}
    for metric_index, metric in enumerate(utility_metrics):
        values = subset[metric].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(
            values,
            BOOTSTRAP_RANDOM_SEED
            + 1000
            + 100 * METHODS.index(method)
            + metric_index,
        )
        row[f"mean_{metric}"] = float(values.mean())
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high
        row[f"std_{metric}"] = float(values.std(ddof=1))
        row[f"min_{metric}"] = float(values.min())
        row[f"max_{metric}"] = float(values.max())
    summary_rows.append(row)

recognition_summary_df = pd.DataFrame(summary_rows)
recognition_summary_df.to_csv(
    OUTPUT_DIR / "recognition_summary_across_keys.csv", index=False
)
display(recognition_summary_df[[
    "method",
    "mean_tar_at_far_1e_3",
    "tar_at_far_1e_3_ci95_low",
    "tar_at_far_1e_3_ci95_high",
    "mean_approximate_eer",
    "mean_auc",
    "mean_cv_accuracy",
]])


# Cell 23
evaluation_image_ids = evaluation_rows_df["image_id"].to_numpy(dtype=int)
quality_rows = []

for method in METHODS:
    for base_seed in BASE_KEY_SEEDS:
        templates = biohashes_by_method[method][base_seed]
        bits = templates[evaluation_image_ids].astype(np.float64)
        one_rates = bits.mean(axis=0)
        clipped = np.clip(one_rates, 1e-12, 1.0 - 1e-12)
        entropy = -(
            clipped * np.log2(clipped)
            + (1.0 - clipped) * np.log2(1.0 - clipped)
        )
        correlation = np.nan_to_num(
            np.corrcoef(bits, rowvar=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        off_diagonal = correlation[
            ~np.eye(TEMPLATE_BITS, dtype=bool)
        ]
        genuine = pair_labels == 1
        genuine_scores = hamming_similarity(
            templates[a_ids[genuine]], templates[b_ids[genuine]]
        )
        quality_rows.append({
            "method": method,
            "base_key_seed": base_seed,
            "mean_one_rate": float(one_rates.mean()),
            "worst_absolute_bit_imbalance": float(
                np.max(np.abs(one_rates - 0.5))
            ),
            "mean_bit_entropy": float(entropy.mean()),
            "minimum_bit_entropy": float(entropy.min()),
            "mean_absolute_interbit_correlation": float(
                np.mean(np.abs(off_diagonal))
            ),
            "maximum_absolute_interbit_correlation": float(
                np.max(np.abs(off_diagonal))
            ),
            "genuine_bit_flip_rate": float(
                1.0 - genuine_scores.mean()
            ),
        })

bit_quality_per_key_df = pd.DataFrame(quality_rows)
bit_quality_per_key_df.to_csv(
    OUTPUT_DIR / "bit_quality_per_key.csv", index=False
)
bit_quality_summary_df = bit_quality_per_key_df.groupby(
    "method", as_index=False
).agg(
    mean_one_rate=("mean_one_rate", "mean"),
    mean_bit_entropy=("mean_bit_entropy", "mean"),
    mean_absolute_interbit_correlation=(
        "mean_absolute_interbit_correlation", "mean"
    ),
    worst_absolute_interbit_correlation=(
        "maximum_absolute_interbit_correlation", "max"
    ),
    mean_genuine_bit_flip_rate=("genuine_bit_flip_rate", "mean"),
)
bit_quality_summary_df.to_csv(
    OUTPUT_DIR / "bit_quality_summary.csv", index=False
)

pca_mean32 = pca_model.mean_.astype(np.float32)
pca_components32 = pca_model.components_.astype(np.float32)


def apply_single_template_cpu(method, raw_embedding, state):
    value = np.asarray(raw_embedding, dtype=np.float32)
    value = value / max(float(np.linalg.norm(value)), 1e-12)
    if method == "RO":
        projected = value @ state
    elif method == "SRHT":
        mixed = fwht_numpy_1d(value * state["signs"])
        projected = mixed[state["indices"]] * state["scale"]
    else:
        pca_value = (value - pca_mean32) @ pca_components32.T
        pca_value = pca_value / max(
            float(np.linalg.norm(pca_value)), 1e-12
        )
        projected = pca_value @ state
    return np.packbits(projected >= 0.0, bitorder="big")


benchmark_rng = np.random.default_rng(2026082603)
benchmark_ids = benchmark_rng.choice(
    evaluation_image_ids, size=CPU_BENCHMARK_SAMPLES, replace=True
)
benchmark_seeds = benchmark_rng.choice(
    np.asarray(BASE_KEY_SEEDS),
    size=CPU_BENCHMARK_SAMPLES,
    replace=True,
)

latency_rows = []
for method in METHODS:
    for warm_index in range(10):
        image_id = int(benchmark_ids[warm_index])
        seed = int(benchmark_seeds[warm_index])
        _ = apply_single_template_cpu(
            method,
            embeddings[image_id],
            projection_states[method][seed],
        )

    for image_id, base_seed in zip(
        benchmark_ids, benchmark_seeds
    ):
        started_ns = time.perf_counter_ns()
        packed = apply_single_template_cpu(
            method,
            embeddings[int(image_id)],
            projection_states[method][int(base_seed)],
        )
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
        if packed.nbytes != TEMPLATE_BITS // 8:
            raise RuntimeError("Unexpected packed template size.")
        latency_rows.append({
            "method": method,
            "image_id": int(image_id),
            "base_key_seed": int(base_seed),
            "cpu_latency_ms": elapsed_ms,
        })

latency_measurements_df = pd.DataFrame(latency_rows)
latency_measurements_df.to_csv(
    OUTPUT_DIR / "cpu_latency_measurements.csv", index=False
)

public_pca_bytes = int(
    pca_mean32.nbytes
    + pca_components32.nbytes
    + pca_model.explained_variance_.astype(np.float32).nbytes
    + pca_model.explained_variance_ratio_.astype(np.float32).nbytes
)
efficiency_rows = []
for method in METHODS:
    latency = latency_measurements_df.loc[
        latency_measurements_df["method"] == method,
        "cpu_latency_ms",
    ].to_numpy(dtype=float)
    key_bytes = projection_validation_df.loc[
        projection_validation_df["method"] == method,
        "explicit_per_key_operator_bytes",
    ].to_numpy(dtype=float)
    efficiency_rows.append({
        "method": method,
        "benchmark_samples": len(latency),
        "cpu_latency_p50_ms": float(np.quantile(latency, 0.50)),
        "cpu_latency_p95_ms": float(np.quantile(latency, 0.95)),
        "cpu_latency_mean_ms": float(latency.mean()),
        "explicit_per_key_operator_bytes": int(key_bytes.mean()),
        "seed_only_key_bytes": 8,
        "public_model_bytes": (
            public_pca_bytes if method == "PCA_GUIDED_RO" else 0
        ),
        "protected_template_bytes": TEMPLATE_BITS // 8,
        "raw_embedding_bytes": EMBEDDING_DIMENSION * 4,
    })

efficiency_summary_df = pd.DataFrame(efficiency_rows)
efficiency_summary_df.to_csv(
    OUTPUT_DIR / "efficiency_summary.csv", index=False
)
display(bit_quality_summary_df)
display(efficiency_summary_df)


# Cell 25
def standard_unlinkability_metrics(
    mated_scores, non_mated_scores, omega=1.0, n_bins=100
):
    mated_scores = np.asarray(mated_scores, dtype=np.float64)
    non_mated_scores = np.asarray(non_mated_scores, dtype=np.float64)
    lower = min(float(mated_scores.min()), float(non_mated_scores.min()))
    upper = max(float(mated_scores.max()), float(non_mated_scores.max()))
    if lower == upper:
        upper = np.nextafter(upper, np.inf)

    edges = np.linspace(lower, upper, n_bins + 1)
    centers = (edges[1:] + edges[:-1]) / 2.0
    mated_density = np.histogram(
        mated_scores, bins=edges, density=True
    )[0]
    non_mated_density = np.histogram(
        non_mated_scores, bins=edges, density=True
    )[0]
    ratio = np.divide(
        mated_density,
        non_mated_density,
        out=np.ones_like(mated_density),
        where=non_mated_density != 0,
    )
    local_d = 2.0 * (
        omega * ratio / (1.0 + omega * ratio)
    ) - 1.0
    local_d[omega * ratio <= 1.0] = 0.0
    local_d[non_mated_density == 0] = 1.0
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(np.clip(
        integrate(local_d * mated_density, x=centers), 0.0, 1.0
    ))


def direction_aware_metrics(labels, scores):
    raw_auc = float(roc_auc_score(labels, scores))
    attacker_auc = max(raw_auc, 1.0 - raw_auc)
    mated = scores[labels == 1]
    non_mated = scores[labels == 0]
    return {
        "raw_auc": raw_auc,
        "attacker_auc": attacker_auc,
        "attacker_direction": (
            "high_score_is_mated"
            if raw_auc >= 0.5
            else "low_score_is_mated"
        ),
        "auc_linkability_index": 2.0 * (attacker_auc - 0.5),
        "d_sys": standard_unlinkability_metrics(mated, non_mated),
        "mated_mean": float(mated.mean()),
        "non_mated_mean": float(non_mated.mean()),
        "absolute_mean_gap": float(
            abs(mated.mean() - non_mated.mean())
        ),
    }


def make_order_invariant_features(
    score_matrix, per_bit_agreement_rates
):
    sorted_scores = np.sort(score_matrix, axis=1)
    summaries = np.column_stack([
        score_matrix.mean(axis=1),
        score_matrix.std(axis=1, ddof=0),
        score_matrix.min(axis=1),
        score_matrix.max(axis=1),
        np.median(score_matrix, axis=1),
        np.ptp(score_matrix, axis=1),
    ])
    return np.column_stack([
        per_bit_agreement_rates, sorted_scores, summaries
    ])


def select_linkage_rule(training_labels, training_scores):
    candidates = []
    for sign, direction in [
        (1.0, "high_score_is_mated"),
        (-1.0, "low_score_is_mated"),
    ]:
        transformed = sign * training_scores
        fpr, tpr, thresholds = roc_curve(
            training_labels,
            transformed,
            pos_label=1,
            drop_intermediate=False,
        )
        balanced = 0.5 * (tpr + 1.0 - fpr)
        finite = np.isfinite(thresholds)
        best = float(balanced[finite].max())
        tied = thresholds[
            finite & np.isclose(balanced, best, rtol=0, atol=1e-12)
        ]
        candidates.append({
            "direction_sign": sign,
            "direction": direction,
            "threshold_transformed": float(tied.max()),
            "training_balanced_accuracy": best,
        })
    candidates.sort(
        key=lambda item: (
            item["training_balanced_accuracy"],
            item["threshold_transformed"],
        ),
        reverse=True,
    )
    return candidates[0]


def apply_linkage_rule(labels, scores, rule):
    predictions = (
        rule["direction_sign"] * scores
        >= rule["threshold_transformed"]
    ).astype(np.int8)
    result = confusion_summary(labels, predictions)
    return {
        "samples": len(labels),
        "tp": result["tp"],
        "tn": result["tn"],
        "fp": result["fp"],
        "fn": result["fn"],
        "link_detection_rate": result["tar"],
        "false_link_rate": result["far"],
        "balanced_accuracy": result["balanced_accuracy"],
        "predictions": predictions,
    }


# Cell 27
setting_metric_rows = []
fold_attack_rows = []
oof_score_frames = []
attack_started = time.perf_counter()

for method in METHODS:
    method_started = time.perf_counter()
    templates_by_seed = biohashes_by_method[method]

    for trial_position, trial in key_trials_df.iterrows():
        trial_id = int(trial["trial_id"])
        query_seed = int(trial["query_seed"])
        exposure_order = [
            int(trial[f"exposure_seed_{i}"])
            for i in range(1, MAX_EXPOSURES + 1)
        ]

        for orientation, gallery_ids, query_ids in [
            ("A_exposed_to_B_query", a_ids, b_ids),
            ("B_exposed_to_A_query", b_ids, a_ids),
        ]:
            query_templates = templates_by_seed[
                query_seed
            ][query_ids]
            all_bit_agreements = np.stack([
                np.equal(
                    templates_by_seed[seed][gallery_ids],
                    query_templates,
                )
                for seed in exposure_order
            ], axis=1)
            all_scores = all_bit_agreements.mean(axis=2)

            for exposure_count in EXPOSURE_COUNTS:
                used_seeds = exposure_order[:exposure_count]
                score_matrix = all_scores[:, :exposure_count]
                bit_agreement_rates = all_bit_agreements[
                    :, :exposure_count, :
                ].mean(axis=1)
                mean_scores = score_matrix.mean(axis=1)
                maximum_scores = score_matrix.max(axis=1)
                features = make_order_invariant_features(
                    score_matrix, bit_agreement_rates
                )

                mean_metrics = direction_aware_metrics(
                    pair_labels, mean_scores
                )
                maximum_metrics = direction_aware_metrics(
                    pair_labels, maximum_scores
                )
                learned_probability = np.full(
                    len(pair_labels), np.nan, dtype=np.float64
                )
                learned_prediction = np.full(
                    len(pair_labels), -1, dtype=np.int8
                )

                for test_fold in sorted(np.unique(pair_folds)):
                    train = pair_folds != test_fold
                    test = pair_folds == test_fold
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            solver="liblinear",
                            max_iter=2000,
                            random_state=ATTACK_MODEL_RANDOM_SEED,
                        ),
                    )
                    model.fit(features[train], pair_labels[train])
                    training_probability = model.predict_proba(
                        features[train]
                    )[:, 1]
                    rule = select_linkage_rule(
                        pair_labels[train], training_probability
                    )
                    test_probability = model.predict_proba(
                        features[test]
                    )[:, 1]
                    fold_result = apply_linkage_rule(
                        pair_labels[test], test_probability, rule
                    )
                    predictions = fold_result.pop("predictions")
                    learned_probability[test] = test_probability
                    learned_prediction[test] = predictions
                    fold_attack_rows.append({
                        "method": method,
                        "trial_id": trial_id,
                        "orientation": orientation,
                        "exposure_count": exposure_count,
                        "query_seed": query_seed,
                        "exposure_seeds": "|".join(
                            map(str, used_seeds)
                        ),
                        "test_fold": int(test_fold),
                        **rule,
                        **fold_result,
                    })

                if not np.isfinite(learned_probability).all():
                    raise RuntimeError("Missing attacker probability.")
                if not np.isin(
                    learned_prediction, [0, 1]
                ).all():
                    raise RuntimeError("Missing attacker prediction.")

                learned_metrics = direction_aware_metrics(
                    pair_labels, learned_probability
                )
                learned_binary = confusion_summary(
                    pair_labels, learned_prediction
                )
                setting_metric_rows.append({
                    "method": method,
                    "trial_id": trial_id,
                    "orientation": orientation,
                    "exposure_count": exposure_count,
                    "query_seed": query_seed,
                    "exposure_seeds": "|".join(
                        map(str, used_seeds)
                    ),
                    "feature_count": int(features.shape[1]),
                    **{
                        f"mean_{key}": value
                        for key, value in mean_metrics.items()
                    },
                    **{
                        f"maximum_{key}": value
                        for key, value in maximum_metrics.items()
                    },
                    **{
                        f"learned_{key}": value
                        for key, value in learned_metrics.items()
                    },
                    "learned_balanced_accuracy": learned_binary[
                        "balanced_accuracy"
                    ],
                    "learned_link_detection_rate": learned_binary["tar"],
                    "learned_false_link_rate": learned_binary["far"],
                    "learned_tp": learned_binary["tp"],
                    "learned_tn": learned_binary["tn"],
                    "learned_fp": learned_binary["fp"],
                    "learned_fn": learned_binary["fn"],
                })

                if SAVE_FULL_OOF_SCORES:
                    frame = pair_manifest_df[[
                        "pair_id", "fold", "label"
                    ]].copy()
                    frame.insert(0, "method", method)
                    frame.insert(1, "trial_id", trial_id)
                    frame.insert(2, "orientation", orientation)
                    frame.insert(
                        3, "exposure_count", exposure_count
                    )
                    frame["query_seed"] = query_seed
                    frame["exposure_seeds"] = "|".join(
                        map(str, used_seeds)
                    )
                    frame["mean_score"] = mean_scores
                    frame["maximum_score"] = maximum_scores
                    frame["learned_oof_probability"] = (
                        learned_probability
                    )
                    frame["learned_oof_prediction"] = (
                        learned_prediction
                    )
                    oof_score_frames.append(frame)

        print(
            f"{method}: trial {trial_position + 1}/"
            f"{len(key_trials_df)} complete"
        )

    print(
        f"{method} attack time: "
        f"{(time.perf_counter() - method_started) / 60:.2f} min"
    )

setting_metrics_df = pd.DataFrame(setting_metric_rows)
fold_attack_df = pd.DataFrame(fold_attack_rows)
setting_metrics_df.to_csv(
    OUTPUT_DIR / "multi_exposure_metrics_per_setting.csv",
    index=False,
)
fold_attack_df.to_csv(
    OUTPUT_DIR / "learned_attacker_fold_results.csv",
    index=False,
)
if SAVE_FULL_OOF_SCORES:
    oof_scores_df = pd.concat(oof_score_frames, ignore_index=True)
    oof_scores_df.to_csv(
        OUTPUT_DIR / "multi_exposure_out_of_fold_scores.csv.gz",
        index=False,
        compression="gzip",
    )
else:
    oof_scores_df = None

expected_settings = (
    len(METHODS)
    * len(key_trials_df)
    * 2
    * len(EXPOSURE_COUNTS)
)
if len(setting_metrics_df) != expected_settings:
    raise RuntimeError("Attack setting count validation failed.")
if len(fold_attack_df) != expected_settings * IDENTITY_FOLDS:
    raise RuntimeError("Attacker fold count validation failed.")

attack_audit = {
    "methods": len(METHODS),
    "key_trials_per_method": len(key_trials_df),
    "directions": 2,
    "exposure_counts": EXPOSURE_COUNTS,
    "settings": len(setting_metrics_df),
    "fold_models": len(fold_attack_df),
    "full_oof_saved": SAVE_FULL_OOF_SCORES,
    "elapsed_minutes": (time.perf_counter() - attack_started) / 60,
}
print(json.dumps(attack_audit, indent=2))


# Cell 29
attacker_prefixes = {
    "mean_score": "mean",
    "maximum_score": "maximum",
    "learned": "learned",
}
attack_summary_rows = []
for method in METHODS:
    for exposure_count in EXPOSURE_COUNTS:
        subset = setting_metrics_df[
            (setting_metrics_df["method"] == method)
            & (
                setting_metrics_df["exposure_count"]
                == exposure_count
            )
        ]
        for attacker_name, prefix in attacker_prefixes.items():
            auc_values = subset[
                f"{prefix}_attacker_auc"
            ].to_numpy(dtype=float)
            dsys_values = subset[
                f"{prefix}_d_sys"
            ].to_numpy(dtype=float)
            offset = (
                METHODS.index(method) * 100000
                + exposure_count * 100
                + len(attack_summary_rows)
            )
            auc_low, auc_high = bootstrap_mean_ci(
                auc_values, BOOTSTRAP_RANDOM_SEED + offset
            )
            dsys_low, dsys_high = bootstrap_mean_ci(
                dsys_values, BOOTSTRAP_RANDOM_SEED + offset + 1
            )
            row = {
                "method": method,
                "exposure_count": exposure_count,
                "attacker": attacker_name,
                "settings": len(subset),
                "mean_attacker_auc": float(auc_values.mean()),
                "auc_ci95_low": auc_low,
                "auc_ci95_high": auc_high,
                "std_attacker_auc": float(
                    auc_values.std(ddof=1)
                ),
                "worst_attacker_auc": float(auc_values.max()),
                "mean_d_sys": float(dsys_values.mean()),
                "d_sys_ci95_low": dsys_low,
                "d_sys_ci95_high": dsys_high,
                "worst_d_sys": float(dsys_values.max()),
            }
            if prefix == "learned":
                balanced = subset[
                    "learned_balanced_accuracy"
                ].to_numpy(dtype=float)
                bal_low, bal_high = bootstrap_mean_ci(
                    balanced,
                    BOOTSTRAP_RANDOM_SEED + offset + 2,
                )
                row.update({
                    "mean_balanced_accuracy": float(
                        balanced.mean()
                    ),
                    "balanced_accuracy_ci95_low": bal_low,
                    "balanced_accuracy_ci95_high": bal_high,
                    "worst_balanced_accuracy": float(
                        balanced.max()
                    ),
                })
            attack_summary_rows.append(row)

cumulative_summary_df = pd.DataFrame(attack_summary_rows)
cumulative_summary_df.to_csv(
    OUTPUT_DIR / "multi_exposure_summary.csv", index=False
)

growth_rows = []
for (method, trial_id, orientation), group in (
    setting_metrics_df.groupby(
        ["method", "trial_id", "orientation"], sort=False
    )
):
    indexed = group.set_index("exposure_count")
    baseline = indexed.loc[1]
    for exposure_count in [2, 5, 10]:
        current = indexed.loc[exposure_count]
        growth_rows.append({
            "method": method,
            "trial_id": int(trial_id),
            "orientation": orientation,
            "exposure_count": exposure_count,
            "learned_auc_growth_from_k1": float(
                current["learned_attacker_auc"]
                - baseline["learned_attacker_auc"]
            ),
            "learned_d_sys_growth_from_k1": float(
                current["learned_d_sys"]
                - baseline["learned_d_sys"]
            ),
            "learned_balanced_accuracy_growth_from_k1": float(
                current["learned_balanced_accuracy"]
                - baseline["learned_balanced_accuracy"]
            ),
        })

growth_per_setting_df = pd.DataFrame(growth_rows)
growth_per_setting_df.to_csv(
    OUTPUT_DIR / "exposure_growth_per_setting.csv",
    index=False,
)

growth_summary_rows = []
growth_metrics = [
    "learned_auc_growth_from_k1",
    "learned_d_sys_growth_from_k1",
    "learned_balanced_accuracy_growth_from_k1",
]
for (method, exposure_count), group in (
    growth_per_setting_df.groupby(
        ["method", "exposure_count"], sort=True
    )
):
    row = {
        "method": method,
        "exposure_count": int(exposure_count),
        "paired_settings": len(group),
    }
    for metric_index, metric in enumerate(growth_metrics):
        values = group[metric].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(
            values,
            BOOTSTRAP_RANDOM_SEED
            + 500000
            + METHODS.index(method) * 10000
            + exposure_count * 10
            + metric_index,
        )
        row[f"mean_{metric}"] = float(values.mean())
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high
        row[f"worst_{metric}"] = float(values.max())
    growth_summary_rows.append(row)

growth_summary_df = pd.DataFrame(growth_summary_rows)
growth_summary_df.to_csv(
    OUTPUT_DIR / "exposure_growth_summary.csv", index=False
)

display(cumulative_summary_df[
    cumulative_summary_df["attacker"] == "learned"
])
display(growth_summary_df)


# Cell 31
def paired_difference_ci(method_values, ro_values, seed):
    difference = (
        np.asarray(method_values, dtype=float)
        - np.asarray(ro_values, dtype=float)
    )
    low, high = bootstrap_mean_ci(difference, seed)
    return float(difference.mean()), low, high


paired_rows = []
for method in [item for item in METHODS if item != "RO"]:
    method_rec = recognition_per_key_df[
        recognition_per_key_df["method"] == method
    ].sort_values("base_key_seed").reset_index(drop=True)
    ro_rec = recognition_per_key_df[
        recognition_per_key_df["method"] == "RO"
    ].sort_values("base_key_seed").reset_index(drop=True)
    if method_rec["base_key_seed"].tolist() != ro_rec[
        "base_key_seed"
    ].tolist():
        raise RuntimeError("Recognition key pairing failed.")

    for metric_index, metric in enumerate([
        "tar_at_far_1e_3",
        "approximate_eer",
        "auc",
        "cv_accuracy",
    ]):
        difference, low, high = paired_difference_ci(
            method_rec[metric],
            ro_rec[metric],
            BOOTSTRAP_RANDOM_SEED
            + 700000
            + METHODS.index(method) * 1000
            + metric_index,
        )
        paired_rows.append({
            "comparison": f"{method}_minus_RO",
            "area": "recognition",
            "exposure_count": np.nan,
            "metric": metric,
            "paired_units": len(method_rec),
            "mean_difference": difference,
            "ci95_low": low,
            "ci95_high": high,
        })

    for exposure_count in EXPOSURE_COUNTS:
        setting_keys = ["trial_id", "orientation"]
        method_attack = setting_metrics_df[
            (setting_metrics_df["method"] == method)
            & (
                setting_metrics_df["exposure_count"]
                == exposure_count
            )
        ].sort_values(setting_keys).reset_index(drop=True)
        ro_attack = setting_metrics_df[
            (setting_metrics_df["method"] == "RO")
            & (
                setting_metrics_df["exposure_count"]
                == exposure_count
            )
        ].sort_values(setting_keys).reset_index(drop=True)
        if not method_attack[setting_keys].equals(
            ro_attack[setting_keys]
        ):
            raise RuntimeError("Attacker setting pairing failed.")

        for metric_index, metric in enumerate([
            "learned_attacker_auc",
            "learned_d_sys",
            "learned_balanced_accuracy",
        ]):
            difference, low, high = paired_difference_ci(
                method_attack[metric],
                ro_attack[metric],
                BOOTSTRAP_RANDOM_SEED
                + 800000
                + METHODS.index(method) * 10000
                + exposure_count * 100
                + metric_index,
            )
            paired_rows.append({
                "comparison": f"{method}_minus_RO",
                "area": "privacy",
                "exposure_count": exposure_count,
                "metric": metric,
                "paired_units": len(method_attack),
                "mean_difference": difference,
                "ci95_low": low,
                "ci95_high": high,
            })

paired_comparison_df = pd.DataFrame(paired_rows)
paired_comparison_df.to_csv(
    OUTPUT_DIR / "paired_differences_vs_RO.csv", index=False
)

recognition_indexed = recognition_summary_df.set_index("method")
learned_indexed = cumulative_summary_df[
    cumulative_summary_df["attacker"] == "learned"
].set_index(["method", "exposure_count"])
efficiency_indexed = efficiency_summary_df.set_index("method")

ro_tar = float(
    recognition_indexed.loc["RO", "mean_tar_at_far_1e_3"]
)
ro_k10_auc = float(
    learned_indexed.loc[("RO", 10), "mean_attacker_auc"]
)
ro_p50 = float(
    efficiency_indexed.loc["RO", "cpu_latency_p50_ms"]
)
ro_key_bytes = float(
    efficiency_indexed.loc[
        "RO", "explicit_per_key_operator_bytes"
    ]
)

decision_rows = []
for method in METHODS:
    tar = float(
        recognition_indexed.loc[
            method, "mean_tar_at_far_1e_3"
        ]
    )
    k1_auc = float(
        learned_indexed.loc[
            (method, 1), "mean_attacker_auc"
        ]
    )
    k10_auc = float(
        learned_indexed.loc[
            (method, 10), "mean_attacker_auc"
        ]
    )
    p50 = float(
        efficiency_indexed.loc[
            method, "cpu_latency_p50_ms"
        ]
    )
    key_bytes = float(
        efficiency_indexed.loc[
            method, "explicit_per_key_operator_bytes"
        ]
    )
    utility_ok = (
        tar >= ro_tar - UTILITY_NONINFERIORITY_TOLERANCE
    )
    security_ok = (
        k10_auc <= ro_k10_auc + SECURITY_AUC_TOLERANCE
    )
    if method == "RO":
        label = "REFERENCE_BASELINE"
    elif utility_ok and security_ok:
        label = "VIABLE_PROJECTION_CANDIDATE"
    elif not utility_ok and not security_ok:
        label = "UTILITY_AND_SECURITY_CONCERNS"
    elif not utility_ok:
        label = "UTILITY_CONCERN"
    else:
        label = "SECURITY_CONCERN"

    decision_rows.append({
        "method": method,
        "mean_tar_at_far_1e_3": tar,
        "tar_difference_vs_ro": tar - ro_tar,
        "mean_learned_attacker_auc_k1": k1_auc,
        "mean_learned_attacker_auc_k10": k10_auc,
        "k10_auc_difference_vs_ro": k10_auc - ro_k10_auc,
        "cpu_p50_ms": p50,
        "cpu_speedup_vs_ro": ro_p50 / p50,
        "explicit_key_bytes": key_bytes,
        "explicit_key_storage_reduction_vs_ro": (
            ro_key_bytes / key_bytes
        ),
        "utility_noninferior_to_ro": bool(utility_ok),
        "k10_security_not_worse_than_ro": bool(security_ok),
        "screening_label": label,
    })

decision_summary_df = pd.DataFrame(decision_rows)
decision_summary_df.to_csv(
    OUTPUT_DIR / "projection_decision_summary.csv", index=False
)
display(decision_summary_df)
display(paired_comparison_df)


# Cell 33
method_colors = {
    "RO": "tab:blue",
    "SRHT": "tab:orange",
    "PCA_GUIDED_RO": "tab:green",
}
method_labels = {
    "RO": "RO",
    "SRHT": "SRHT/Hadamard",
    "PCA_GUIDED_RO": "PCA-guided RO",
}

# Recognition utility.
plot_rec = (
    recognition_summary_df.set_index("method")
    .loc[METHODS].reset_index()
)
means = plot_rec["mean_tar_at_far_1e_3"].to_numpy()
errors = np.vstack([
    means - plot_rec["tar_at_far_1e_3_ci95_low"].to_numpy(),
    plot_rec["tar_at_far_1e_3_ci95_high"].to_numpy() - means,
])
plt.figure(figsize=(8.0, 5.0))
plt.bar(
    [method_labels[item] for item in plot_rec["method"]],
    means,
    yerr=errors,
    capsize=5,
    color=[method_colors[item] for item in plot_rec["method"]],
)
plt.ylabel("Mean TAR at FAR 0.1%")
plt.title("Same-key recognition across 20 keys")
plt.ylim(max(0.0, means.min() - 0.05), min(1.0, means.max() + 0.03))
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "recognition_tar_at_far_comparison.png",
    dpi=180,
)
plt.show()

# Multi-exposure privacy.
plt.figure(figsize=(8.8, 5.4))
for method in METHODS:
    plot_df = cumulative_summary_df[
        (cumulative_summary_df["method"] == method)
        & (cumulative_summary_df["attacker"] == "learned")
    ].sort_values("exposure_count")
    lower = (
        plot_df["mean_attacker_auc"]
        - plot_df["auc_ci95_low"]
    )
    upper = (
        plot_df["auc_ci95_high"]
        - plot_df["mean_attacker_auc"]
    )
    plt.errorbar(
        plot_df["exposure_count"],
        plot_df["mean_attacker_auc"],
        yerr=np.vstack([lower, upper]),
        marker="o",
        capsize=4,
        linewidth=2,
        color=method_colors[method],
        label=method_labels[method],
    )
plt.axhline(
    0.5, color="black", linestyle="--", label="Chance = 0.50"
)
plt.xticks(EXPOSURE_COUNTS)
plt.xlabel("Number of exposed templates (K)")
plt.ylabel("Identity-disjoint learned-attacker AUC")
plt.title("Multi-exposure linkability by projection")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "learned_attacker_auc_vs_exposures.png",
    dpi=180,
)
plt.show()

# Bit quality.
quality_plot = (
    bit_quality_summary_df.set_index("method").loc[METHODS]
)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
axes[0].bar(
    [method_labels[item] for item in METHODS],
    quality_plot["mean_bit_entropy"],
    color=[method_colors[item] for item in METHODS],
)
axes[0].set_ylabel("Mean bit entropy (ideal = 1)")
axes[0].set_title("Bit unpredictability")
axes[1].bar(
    [method_labels[item] for item in METHODS],
    quality_plot["mean_absolute_interbit_correlation"],
    color=[method_colors[item] for item in METHODS],
)
axes[1].set_ylabel("Mean absolute correlation (ideal = 0)")
axes[1].set_title("Inter-bit redundancy")
for axis in axes:
    axis.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "bit_quality_comparison.png", dpi=180
)
plt.show()

# Edge-oriented CPU latency.
eff_plot = efficiency_summary_df.set_index("method").loc[METHODS]
plt.figure(figsize=(8.0, 5.0))
plt.bar(
    [method_labels[item] for item in METHODS],
    eff_plot["cpu_latency_p50_ms"],
    color=[method_colors[item] for item in METHODS],
)
plt.ylabel("CPU p50 latency per template (ms)")
plt.title("Single-template generation latency")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "cpu_latency_comparison.png", dpi=180
)
plt.show()

# Privacy–utility plane: left and up are preferable.
plt.figure(figsize=(8.0, 5.5))
for method in METHODS:
    x = float(
        learned_indexed.loc[
            (method, 10), "mean_attacker_auc"
        ]
    )
    y = float(
        recognition_indexed.loc[
            method, "mean_tar_at_far_1e_3"
        ]
    )
    plt.scatter(
        x, y, s=120, color=method_colors[method]
    )
    plt.annotate(
        method_labels[method],
        (x, y),
        xytext=(6, 5),
        textcoords="offset points",
    )
plt.axvline(0.5, color="black", linestyle="--", alpha=0.6)
plt.xlabel("K=10 learned-attacker AUC (lower is better)")
plt.ylabel("TAR at FAR 0.1% (higher is better)")
plt.title("Privacy–utility trade-off")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "privacy_utility_tradeoff.png", dpi=180
)
plt.show()


# Cell 35
RUN_COMPLETED_UTC = datetime.now(timezone.utc).isoformat()

best_utility_method = decision_summary_df.loc[
    decision_summary_df["mean_tar_at_far_1e_3"].idxmax(),
    "method",
]
best_privacy_method = decision_summary_df.loc[
    decision_summary_df[
        "mean_learned_attacker_auc_k10"
    ].idxmin(),
    "method",
]
fastest_method = decision_summary_df.loc[
    decision_summary_df["cpu_p50_ms"].idxmin(),
    "method",
]
viable_candidates = decision_summary_df.loc[
    decision_summary_df["screening_label"]
    == "VIABLE_PROJECTION_CANDIDATE",
    "method",
].tolist()

interpretation = {
    "highest_mean_tar_at_far_1e_3": best_utility_method,
    "lowest_mean_k10_learned_attacker_auc": best_privacy_method,
    "lowest_cpu_p50_latency": fastest_method,
    "non_ro_methods_passing_predeclared_screen": viable_candidates,
    "screen_is_not_a_universal_winner_claim": True,
}

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "title": EXPERIMENT_TITLE,
    "researcher": RESEARCHER_NAME,
    "institution": INSTITUTION,
    "base_e05_id_notebook_sha256": E05_ID_BASE_NOTEBOOK_SHA256,
    "run_started_utc": RUN_STARTED_UTC,
    "run_completed_utc": RUN_COMPLETED_UTC,
    "runtime": runtime_audit,
    "source_inventory": source_inventory,
    "cache_audit": cache_audit,
    "identity_partition": partition_audit,
    "pca": pca_audit,
    "identity_disjoint_protocol": protocol_audit,
    "template_generation": template_generation_audit,
    "attack_audit": attack_audit,
    "recognition_summary": json.loads(
        recognition_summary_df.to_json(orient="records")
    ),
    "bit_quality_summary": json.loads(
        bit_quality_summary_df.to_json(orient="records")
    ),
    "efficiency_summary": json.loads(
        efficiency_summary_df.to_json(orient="records")
    ),
    "multi_exposure_summary": json.loads(
        cumulative_summary_df.to_json(orient="records")
    ),
    "growth_summary": json.loads(
        growth_summary_df.to_json(orient="records")
    ),
    "decision_summary": json.loads(
        decision_summary_df.to_json(orient="records")
    ),
    "interpretation": interpretation,
    "scope_limitations": [
        "Restricted to valid images in the frozen 7,701-image LFW cache.",
        "The deterministic 6,000-pair corpus is identity-disjoint but is not the official LFW verification protocol.",
        "PCA is fitted on reserved LFW identities; cross-dataset PCA transfer is not tested.",
        "Bootstrap intervals quantify key/direction variability, not demographic population uncertainty.",
        "The attacker does not receive projection matrices, raw embeddings, face images, or secret seeds.",
        "Reconstruction, inversion, impersonation, and unlimited exposures are outside E06-PROJ.",
        "CPU timing must later be repeated on the chosen physical edge platform.",
        "The candidate screen is an exploratory research control, not certification.",
    ],
}
with open(
    OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8"
) as file:
    json.dump(experiment_record, file, indent=2)


quality_indexed = bit_quality_summary_df.set_index("method")


def brief_row(method):
    utility = recognition_indexed.loc[method]
    privacy_k1 = learned_indexed.loc[(method, 1)]
    privacy_k10 = learned_indexed.loc[(method, 10)]
    quality = quality_indexed.loc[method]
    efficiency = efficiency_indexed.loc[method]
    return (
        f"| {method_labels[method]} | "
        f"{utility['mean_tar_at_far_1e_3']:.4f} | "
        f"{utility['mean_approximate_eer']:.4f} | "
        f"{privacy_k1['mean_attacker_auc']:.4f} | "
        f"{privacy_k10['mean_attacker_auc']:.4f} | "
        f"{privacy_k10['mean_d_sys']:.4f} | "
        f"{quality['mean_bit_entropy']:.4f} | "
        f"{efficiency['cpu_latency_p50_ms']:.4f} |"
    )


brief_rows = "\n".join(
    brief_row(method) for method in METHODS
)
candidate_text = (
    ", ".join(method_labels[item] for item in viable_candidates)
    if viable_candidates
    else "none"
)
brief = f'''# LFW-E06-PROJ — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Researcher:** {RESEARCHER_NAME}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

Under identical 128-bit templates, keys, identity-disjoint folds, and multi-exposure
attacker conditions, does SRHT or PCA-guided RO offer a better
recognition–privacy–efficiency balance than RO?

## Controlled design

- Frozen MobileFaceNet cache: {valid_count:,} valid of {EXPECTED_UNIQUE_IMAGES:,} images.
- PCA calibration: {len(calibration_identity_set):,} identities and {len(calibration_rows_df):,} images.
- Calibration/evaluation identity overlap: **0**.
- Evaluation: {len(pair_manifest_df):,} balanced pairs in {IDENTITY_FOLDS} identity-disjoint folds.
- Methods: RO, SRHT/Hadamard, PCA(512→256)+keyed RO(256→128).
- Twenty keys, two image directions, nested K=1,2,5,10 exposures.

## Main results

| Method | TAR@FAR 0.1% | EER | Attacker AUC K=1 | Attacker AUC K=10 | D_sys K=10 | Bit entropy | CPU p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
{brief_rows}

## Controlled interpretation

- Highest recognition TAR: **{method_labels[best_utility_method]}**.
- Lowest K=10 attacker AUC: **{method_labels[best_privacy_method]}**.
- Fastest CPU template generation: **{method_labels[fastest_method]}**.
- Non-RO methods passing the predeclared screen: **{candidate_text}**.

The screen requires TAR within 0.5 percentage points of RO and K=10 attacker AUC no
more than 0.01 above RO. A projection should be called better only when paired
differences, key stability, privacy, and efficiency jointly support it.

## Scope

This result covers LFW, 20 deterministic keys, 128-bit templates, and a
protected-template-only identity-disjoint attacker. It does not cover compromised keys,
reconstruction, impersonation, cross-dataset validation, or physical-device energy.
'''

(OUTPUT_DIR / "LFW-E06-PROJ_One-Page_Experiment_Brief.md").write_text(
    brief, encoding="utf-8"
)
(OUTPUT_DIR / "RESULTS_README.md").write_text(
    brief
    + "\n\n## Reading note\n\n"
    + "Negative privacy differences versus RO are favourable.\n",
    encoding="utf-8",
)

manifest_records = []
for result_path in sorted(
    path for path in OUTPUT_DIR.rglob("*") if path.is_file()
):
    manifest_records.append({
        "relative_path": result_path.relative_to(
            OUTPUT_DIR
        ).as_posix(),
        "size_bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
    })
pd.DataFrame(manifest_records).to_csv(
    OUTPUT_DIR / "result_file_manifest.csv", index=False
)

print(brief)
print("Result files before checksum manifest:", len(manifest_records))


# Cell 37
RESULTS_ARCHIVE_BASE = Path(
    "/kaggle/working/LFW-E06-PROJ-projection-comparison-results"
)
results_archive_path = Path(shutil.make_archive(
    str(RESULTS_ARCHIVE_BASE),
    "zip",
    root_dir=OUTPUT_DIR,
))

os.chdir("/kaggle/working")
print("Results archive:", results_archive_path)
print(
    "Archive size (MiB):",
    round(results_archive_path.stat().st_size / 1024**2, 2),
)
print("Click below to download:")
display(FileLink(results_archive_path.name))
print(
    "If the link does not open, use Kaggle's Output/Files panel "
    "and download LFW-E06-PROJ-projection-comparison-results.zip."
)


# Cell 39

