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

EXPERIMENT_ID = "LFW-E07_INV-embedding-inversion"
EXPERIMENT_TITLE = "Identity-Disjoint Multi-Exposure Embedding Inversion"
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
E06_REFERENCE_NOTEBOOK_SHA256 = (
    "d21b829ad6bde5e63a0d4e38f8eac564b1f4032b072b883eb2bd8a4524c13507"
)

OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Leave these as None for automatic Kaggle discovery. Set a Path only if needed.
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
NUMBER_OF_KEY_TRIALS = 10
ATTACK_IDENTITIES_PER_FOLD = 120
SAVE_PER_IMAGE_RESULTS = True

# These values preserve the E06 partition and pair corpus exactly.
PCA_CALIBRATION_RANDOM_SEED = 2026082601
PCA_CALIBRATION_TARGET_IMAGES = 1024
IDENTITY_SPLIT_RANDOM_SEED = 2026082501
PAIR_SAMPLING_RANDOM_SEED = 2026082502
IDENTITY_FOLDS = 10
PAIRS_PER_CLASS_PER_FOLD = 300
MAX_GENUINE_PAIRS_PER_IDENTITY = 10

ATTACK_KEY_TRIAL_RANDOM_SEED = 2026082701
ATTACK_TARGET_RANDOM_SEED = 2026082702
BOOTSTRAP_RANDOM_SEED = 2026082703
BOOTSTRAP_REPETITIONS = 5000
TARGET_FAR = 1e-3

EXPECTED_SOURCE_TOTAL_PAIRS = 6000
EXPECTED_SOURCE_GENUINE_PAIRS = 3000
EXPECTED_SOURCE_IMPOSTOR_PAIRS = 3000
EXPECTED_UNIQUE_IMAGES = 7701
EXPECTED_VALID_EMBEDDINGS = 7682
EXPECTED_EVALUATION_PAIRS = IDENTITY_FOLDS * PAIRS_PER_CLASS_PER_FOLD * 2

if len(BASE_KEY_SEEDS) != 20 or len(set(BASE_KEY_SEEDS)) != 20:
    raise RuntimeError("E07 requires the unchanged 20 unique E05/E06 key seeds.")
if not (TEMPLATE_BITS <= PCA_DIMENSION <= EMBEDDING_DIMENSION):
    raise RuntimeError("Expected 128 <= PCA dimension <= 512.")
if MAX_EXPOSURES >= len(BASE_KEY_SEEDS):
    raise RuntimeError("A fresh query key must remain outside the exposed keys.")

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
    "compute_device": COMPUTE_DEVICE,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
threat_model = {
    "attacker_has_exposed_binary_templates": True,
    "attacker_has_corresponding_projection_operators": True,
    "fresh_query_key_is_not_exposed": True,
    "system_applies_fresh_key_to_reconstructed_embedding": True,
    "attacker_training_and_target_identities_are_disjoint": True,
    "attacker_has_auxiliary_paired_templates_and_embeddings_for_training_identities": True,
    "fresh_enrollment_uses_a_different_image_of_the_same_identity": True,
    "pixel_level_face_generation_evaluated": False,
    "camera_or_pad_bypass_claimed": False,
}
experiment_config = {
    "experiment_id": EXPERIMENT_ID,
    "e06_reference_notebook_sha256": E06_REFERENCE_NOTEBOOK_SHA256,
    "methods": METHODS,
    "dimensions": {"embedding": 512, "pca": 256, "template_bits": 128},
    "base_key_seeds": BASE_KEY_SEEDS,
    "exposure_counts": EXPOSURE_COUNTS,
    "number_of_key_trials": NUMBER_OF_KEY_TRIALS,
    "attack_identities_per_fold_requested": ATTACK_IDENTITIES_PER_FOLD,
    "identity_folds": IDENTITY_FOLDS,
    "pairs_per_class_per_fold": PAIRS_PER_CLASS_PER_FOLD,
    "target_far": TARGET_FAR,
    "save_per_image_results": SAVE_PER_IMAGE_RESULTS,
    "threat_model": threat_model,
}
with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as file:
    json.dump(experiment_config, file, indent=2)

print("Experiment:", EXPERIMENT_ID)
print("Output:", OUTPUT_DIR)
print("Compute device:", COMPUTE_DEVICE)
print(json.dumps(runtime_audit, indent=2))
print(json.dumps(threat_model, indent=2))


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
valid_image_ids = np.flatnonzero(valid_embedding_mask)
raw_valid_embeddings = l2_normalize_rows(embeddings[valid_embedding_mask])
raw_embeddings_full = np.zeros_like(embeddings, dtype=np.float32)
raw_embeddings_full[valid_embedding_mask] = raw_valid_embeddings

calibration_ids = calibration_rows_df["image_id"].to_numpy(dtype=int)
calibration_embeddings = raw_embeddings_full[calibration_ids]

pca_started = time.perf_counter()
pca_model = PCA(
    n_components=PCA_DIMENSION,
    svd_solver="randomized",
    whiten=False,
    random_state=PCA_CALIBRATION_RANDOM_SEED,
)
pca_model.fit(calibration_embeddings)
pca_fit_seconds = time.perf_counter() - pca_started

pca_raw_valid = pca_model.transform(raw_valid_embeddings).astype(np.float32)
pca_valid_embeddings = l2_normalize_rows(pca_raw_valid)
pca_embeddings_full = np.zeros(
    (len(embeddings), PCA_DIMENSION), dtype=np.float32
)
pca_raw_full = np.zeros_like(pca_embeddings_full)
pca_embeddings_full[valid_embedding_mask] = pca_valid_embeddings
pca_raw_full[valid_embedding_mask] = pca_raw_valid

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


test_rng = np.random.default_rng(2026082602)
test_vector = test_rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
dense_hadamard = dense_hadamard_matrix(EMBEDDING_DIMENSION)
fwht_max_error = float(np.max(np.abs(
    fwht_numpy_1d(test_vector) - test_vector @ dense_hadamard
)))
print("FWHT dense-equivalence maximum error:", fwht_max_error)
if fwht_max_error >= 1e-4:
    raise RuntimeError("Fast Hadamard transform failed its unit test.")


# Cell 14
torch.backends.cuda.matmul.allow_tf32 = False

method_inputs = {
    "RO": raw_valid_embeddings,
    "SRHT": raw_valid_embeddings,
    "PCA_GUIDED_RO": pca_valid_embeddings,
}
biohashes_by_method = {method: {} for method in METHODS}
projection_states = {method: {} for method in METHODS}
operator_matrices = {method: {} for method in METHODS}
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
            operator = state
            projected = input_torch @ torch.from_numpy(operator).to(COMPUTE_DEVICE)
            target_gram = np.eye(TEMPLATE_BITS, dtype=np.float32)

        elif method == "SRHT":
            state = create_srht_state(
                EMBEDDING_DIMENSION, TEMPLATE_BITS, method_seed
            )
            operator = (
                state["signs"][:, None]
                * dense_hadamard[:, state["indices"]]
                * state["scale"]
            ).astype(np.float32)
            signs_torch = torch.from_numpy(state["signs"]).to(COMPUTE_DEVICE)
            indices_torch = torch.from_numpy(state["indices"]).to(COMPUTE_DEVICE)
            mixed = fwht_torch_batch(input_torch * signs_torch)
            mixed = mixed / np.sqrt(EMBEDDING_DIMENSION)
            projected = torch.index_select(mixed, 1, indices_torch) * state["scale"]
            target_gram = (state["scale"] ** 2) * np.eye(
                TEMPLATE_BITS, dtype=np.float32
            )

        else:
            state = create_seeded_ro_matrix(
                PCA_DIMENSION, TEMPLATE_BITS, method_seed
            )
            operator = state
            projected = input_torch @ torch.from_numpy(operator).to(COMPUTE_DEVICE)
            target_gram = np.eye(TEMPLATE_BITS, dtype=np.float32)

        if COMPUTE_DEVICE == "cuda":
            torch.cuda.synchronize()
        valid_templates = (projected >= 0.0).cpu().numpy()
        full_templates = np.zeros(
            (len(embeddings), TEMPLATE_BITS), dtype=bool
        )
        full_templates[valid_embedding_mask] = valid_templates
        packed = np.packbits(full_templates, axis=1, bitorder="big")

        biohashes_by_method[method][base_seed] = full_templates
        projection_states[method][base_seed] = state
        operator_matrices[method][base_seed] = operator
        validation_rows.append({
            "method": method,
            "base_key_seed": base_seed,
            "method_seed": method_seed,
            "input_dimension": input_values.shape[1],
            "template_bits": TEMPLATE_BITS,
            "valid_templates": len(valid_templates),
            "gram_max_error": float(np.max(np.abs(
                operator.T @ operator - target_gram
            ))),
            "overall_one_rate": float(valid_templates.mean()),
            "packed_template_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
            "batch_generation_ms": (time.perf_counter() - started) * 1000,
        })
        del projected
        if method == "SRHT":
            del signs_torch, indices_torch, mixed

    del input_torch
    if COMPUTE_DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Completed template generation:", method)

projection_validation_df = pd.DataFrame(validation_rows)
projection_validation_df.to_csv(
    OUTPUT_DIR / "projection_validation.csv", index=False
)
if len(projection_validation_df) != len(METHODS) * len(BASE_KEY_SEEDS):
    raise RuntimeError("Expected exactly 60 projection/template records.")
if projection_validation_df["gram_max_error"].max() >= 1e-4:
    raise RuntimeError("A projection failed Gram-matrix validation.")

template_generation_audit = {
    "methods": METHODS,
    "keys_per_method": len(BASE_KEY_SEEDS),
    "total_template_sets": len(projection_validation_df),
    "device": COMPUTE_DEVICE,
    "elapsed_seconds": time.perf_counter() - generation_started,
}
display(projection_validation_df.groupby("method").agg(
    mean_one_rate=("overall_one_rate", "mean"),
    worst_gram_error=("gram_max_error", "max"),
    mean_batch_ms=("batch_generation_ms", "mean"),
))


# Cell 16
target_rng = np.random.default_rng(ATTACK_TARGET_RANDOM_SEED)
fold_identity_lists = {
    fold: identity_fold_assignment_df.loc[
        identity_fold_assignment_df["fold"] == fold, "identity"
    ].astype(str).loc[
        lambda series: series.map(lambda identity: len(identity_to_images[identity]) >= 2)
    ].tolist()
    for fold in range(1, IDENTITY_FOLDS + 1)
}
actual_identities_per_fold = min(
    ATTACK_IDENTITIES_PER_FOLD,
    min(len(items) for items in fold_identity_lists.values()),
)
if actual_identities_per_fold < ATTACK_IDENTITIES_PER_FOLD:
    print(
        "Requested", ATTACK_IDENTITIES_PER_FOLD,
        "identities/fold; using", actual_identities_per_fold,
        "because of the smallest fold."
    )

attack_target_rows = []
for fold in range(1, IDENTITY_FOLDS + 1):
    identities = np.asarray(fold_identity_lists[fold], dtype=object)
    target_rng.shuffle(identities)
    for identity in identities[:actual_identities_per_fold]:
        exposed_image_id, fresh_image_id = target_rng.choice(
            identity_to_images[str(identity)], size=2, replace=False
        ).astype(int)
        attack_target_rows.append({
            "fold": fold,
            "identity": str(identity),
            "exposed_image_id": exposed_image_id,
            "exposed_image_key": image_id_to_key[exposed_image_id],
            "fresh_image_id": fresh_image_id,
            "fresh_image_key": image_id_to_key[fresh_image_id],
        })

attack_targets_df = pd.DataFrame(attack_target_rows).sort_values(
    ["fold", "identity"]
).reset_index(drop=True)
attack_target_ids = attack_targets_df["exposed_image_id"].to_numpy(dtype=int)
fresh_target_ids = attack_targets_df["fresh_image_id"].to_numpy(dtype=int)
attack_folds = attack_targets_df["fold"].to_numpy(dtype=np.int8)

if attack_targets_df["identity"].duplicated().any():
    raise RuntimeError("Each attack target identity must appear exactly once.")
if set(attack_targets_df["identity"]) & calibration_identity_set:
    raise RuntimeError("Attack target leaked into PCA calibration identities.")
if np.array_equal(attack_target_ids, fresh_target_ids):
    raise RuntimeError("Exposed and fresh image arrays must differ.")
if np.any(attack_target_ids == fresh_target_ids):
    raise RuntimeError("Every identity requires two different images.")
if not (valid_embedding_mask[attack_target_ids] & valid_embedding_mask[fresh_target_ids]).all():
    raise RuntimeError("An attack target has an invalid embedding.")
attack_targets_df.to_csv(OUTPUT_DIR / "attack_target_manifest.csv", index=False)

trial_rng = np.random.default_rng(ATTACK_KEY_TRIAL_RANDOM_SEED)
query_order = np.asarray(BASE_KEY_SEEDS, dtype=np.int64).copy()
trial_rng.shuffle(query_order)
query_order = query_order[:NUMBER_OF_KEY_TRIALS]
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
for row in key_trials_df.itertuples(index=False):
    exposed = [
        int(getattr(row, f"exposure_seed_{i}"))
        for i in range(1, MAX_EXPOSURES + 1)
    ]
    if len(set(exposed)) != MAX_EXPOSURES:
        raise RuntimeError("Exposure keys must be unique within a trial.")
    if int(row.query_seed) in exposed:
        raise RuntimeError("Fresh query key cannot be exposed.")
key_trials_df.to_csv(OUTPUT_DIR / "key_trial_schedule.csv", index=False)

target_audit = {
    "identities_per_fold": actual_identities_per_fold,
    "total_target_identities": len(attack_targets_df),
    "images_per_identity": 2,
    "exposed_fresh_same_image_count": int(
        np.sum(attack_target_ids == fresh_target_ids)
    ),
    "key_trials": len(key_trials_df),
    "fresh_key_exposure_overlap": 0,
}
print(json.dumps(target_audit, indent=2))
display(key_trials_df)


# Cell 18
def hamming_similarity(template_a, template_b):
    return 1.0 - np.mean(np.logical_xor(template_a, template_b), axis=1)


def threshold_at_far(labels, scores, target_far):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    impostor = np.sort(scores[labels == 0])[::-1]
    allowed = int(np.floor(target_far * len(impostor)))
    if allowed >= len(impostor):
        return float("-inf")
    return float(np.nextafter(impostor[allowed], np.inf))


a_ids = pair_manifest_df["image_a_id"].to_numpy(dtype=int)
b_ids = pair_manifest_df["image_b_id"].to_numpy(dtype=int)
pair_labels = pair_manifest_df["label"].to_numpy(dtype=np.int8)
pair_folds = pair_manifest_df["fold"].to_numpy(dtype=np.int8)

if not (valid_embedding_mask[a_ids] & valid_embedding_mask[b_ids]).all():
    raise RuntimeError("An evaluation pair contains an invalid embedding.")

raw_pair_scores = np.sum(
    raw_embeddings_full[a_ids] * raw_embeddings_full[b_ids], axis=1
)
raw_thresholds = {}
protected_thresholds = {}
threshold_rows = []

for test_fold in range(1, IDENTITY_FOLDS + 1):
    train = pair_folds != test_fold
    test = pair_folds == test_fold
    threshold = threshold_at_far(pair_labels[train], raw_pair_scores[train], TARGET_FAR)
    test_predictions = raw_pair_scores[test] >= threshold
    test_labels = pair_labels[test]
    raw_thresholds[test_fold] = threshold
    threshold_rows.append({
        "matcher": "RAW_COSINE",
        "method": "RAW",
        "query_seed": np.nan,
        "test_fold": test_fold,
        "threshold": threshold,
        "observed_test_far": float(test_predictions[test_labels == 0].mean()),
        "observed_test_tar": float(test_predictions[test_labels == 1].mean()),
    })

for method in METHODS:
    for query_seed in key_trials_df["query_seed"].astype(int):
        templates = biohashes_by_method[method][query_seed]
        scores = hamming_similarity(templates[a_ids], templates[b_ids])
        for test_fold in range(1, IDENTITY_FOLDS + 1):
            train = pair_folds != test_fold
            test = pair_folds == test_fold
            threshold = threshold_at_far(pair_labels[train], scores[train], TARGET_FAR)
            test_predictions = scores[test] >= threshold
            test_labels = pair_labels[test]
            protected_thresholds[(method, query_seed, test_fold)] = threshold
            threshold_rows.append({
                "matcher": "PROTECTED_HAMMING",
                "method": method,
                "query_seed": query_seed,
                "test_fold": test_fold,
                "threshold": threshold,
                "observed_test_far": float(
                    test_predictions[test_labels == 0].mean()
                ),
                "observed_test_tar": float(
                    test_predictions[test_labels == 1].mean()
                ),
            })

matcher_thresholds_df = pd.DataFrame(threshold_rows)
matcher_thresholds_df.to_csv(OUTPUT_DIR / "matcher_thresholds.csv", index=False)
display(matcher_thresholds_df.groupby(["matcher", "method"], as_index=False).agg(
    mean_threshold=("threshold", "mean"),
    observed_far=("observed_test_far", "mean"),
    observed_tar=("observed_test_tar", "mean"),
))


# Cell 20
def signed_backprojection(template_rows, operator):
    signed = np.where(template_rows, 1.0, -1.0).astype(np.float32)
    signed_torch = torch.from_numpy(signed).to(COMPUTE_DEVICE)
    operator_torch = torch.from_numpy(operator).to(COMPUTE_DEVICE)
    with torch.no_grad():
        reconstructed = signed_torch @ operator_torch.T
    output = reconstructed.cpu().numpy().astype(np.float16)
    del signed_torch, operator_torch, reconstructed
    return output


backprojection_cache = {method: {} for method in METHODS}
backprojection_started = time.perf_counter()
for method in METHODS:
    for base_seed in BASE_KEY_SEEDS:
        backprojection_cache[method][base_seed] = signed_backprojection(
            biohashes_by_method[method][base_seed][attack_target_ids],
            operator_matrices[method][base_seed],
        )
    if COMPUTE_DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Backprojection cache complete:", method)

backprojection_cache_bytes = sum(
    array.nbytes
    for method_cache in backprojection_cache.values()
    for array in method_cache.values()
)
backprojection_audit = {
    "cache_dtype": "float16",
    "target_images": len(attack_target_ids),
    "operators_per_method": len(BASE_KEY_SEEDS),
    "cache_mib": backprojection_cache_bytes / 1024**2,
    "elapsed_seconds": time.perf_counter() - backprojection_started,
}
print(json.dumps(backprojection_audit, indent=2))


# Cell 22
def normalize_rows(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), eps
    )


def diagonal_fit_predict(train_x, train_y, test_x, eps=1e-8):
    """Independent least-squares scale/offset per coordinate."""
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    mean_x = train_x.mean(axis=0)
    mean_y = train_y.mean(axis=0)
    centered_x = train_x - mean_x
    centered_y = train_y - mean_y
    variance = np.mean(centered_x * centered_x, axis=0)
    covariance = np.mean(centered_x * centered_y, axis=0)
    slope = covariance / np.maximum(variance, eps)
    return (test_x - mean_x) * slope + mean_y


def cross_validated_calibration(backprojected, source_truth, folds):
    predictions = np.zeros_like(source_truth, dtype=np.float32)
    for test_fold in range(1, IDENTITY_FOLDS + 1):
        train = folds != test_fold
        test = folds == test_fold
        predictions[test] = diagonal_fit_predict(
            backprojected[train], source_truth[train], backprojected[test]
        )
    return normalize_rows(predictions)


attack_raw_truth = raw_embeddings_full[attack_target_ids]
fresh_raw_truth = raw_embeddings_full[fresh_target_ids]
attack_pca_raw_truth = pca_raw_full[attack_target_ids]
attack_source_truth = {
    "RO": attack_raw_truth,
    "SRHT": attack_raw_truth,
    "PCA_GUIDED_RO": pca_embeddings_full[attack_target_ids],
}


def source_to_raw(method, source_estimate):
    if method != "PCA_GUIDED_RO":
        return normalize_rows(source_estimate)
    raw_estimate = np.zeros(
        (len(source_estimate), EMBEDDING_DIMENSION), dtype=np.float32
    )
    pca_norms = np.linalg.norm(attack_pca_raw_truth, axis=1)
    for test_fold in range(1, IDENTITY_FOLDS + 1):
        train = attack_folds != test_fold
        test = attack_folds == test_fold
        training_median_norm = float(np.median(pca_norms[train]))
        restored_pca = source_estimate[test] * training_median_norm
        raw_estimate[test] = pca_model.inverse_transform(restored_pca).astype(
            np.float32
        )
    return normalize_rows(raw_estimate)


def raw_to_source(method, raw_estimate):
    raw_estimate = normalize_rows(raw_estimate)
    if method == "PCA_GUIDED_RO":
        return normalize_rows(pca_model.transform(raw_estimate).astype(np.float32))
    return raw_estimate


def templates_from_source(source_estimate, method, seed):
    return source_estimate @ operator_matrices[method][seed] >= 0.0


baseline_raw_by_fold = {}
for test_fold in range(1, IDENTITY_FOLDS + 1):
    train = attack_folds != test_fold
    baseline = normalize_rows(attack_raw_truth[train].mean(axis=0, keepdims=True))[0]
    baseline_raw_by_fold[test_fold] = baseline

baseline_raw = np.stack([
    baseline_raw_by_fold[int(fold)] for fold in attack_folds
]).astype(np.float32)

# Small synthetic unit test for the diagonal calibrator.
unit_rng = np.random.default_rng(2026082704)
unit_x = unit_rng.normal(size=(200, 8)).astype(np.float32)
unit_y = unit_x * np.linspace(0.5, 1.5, 8, dtype=np.float32) + 0.2
unit_prediction = diagonal_fit_predict(unit_x[:150], unit_y[:150], unit_x[150:])
unit_error = float(np.max(np.abs(unit_prediction - unit_y[150:])))
print("Diagonal-calibration unit-test max error:", unit_error)
if unit_error >= 1e-4:
    raise RuntimeError("Diagonal calibration failed its unit test.")


# Cell 24
def rowwise_hamming_similarity(bits_a, bits_b):
    return 1.0 - np.logical_xor(bits_a, bits_b).mean(axis=1)


def fold_threshold_vector(threshold_lookup, method=None, query_seed=None):
    if method is None:
        return np.asarray(
            [threshold_lookup[int(fold)] for fold in attack_folds],
            dtype=np.float32,
        )
    return np.asarray([
        threshold_lookup[(method, int(query_seed), int(fold))]
        for fold in attack_folds
    ], dtype=np.float32)


per_image_frames = []
attack_started = time.perf_counter()

for method in METHODS:
    source_truth = attack_source_truth[method]

    for trial in key_trials_df.itertuples(index=False):
        trial_id = int(trial.trial_id)
        query_seed = int(trial.query_seed)
        exposure_order = [
            int(getattr(trial, f"exposure_seed_{i}"))
            for i in range(1, MAX_EXPOSURES + 1)
        ]
        cumulative = np.zeros_like(source_truth, dtype=np.float32)

        target_same_image_templates = biohashes_by_method[
            method
        ][query_seed][attack_target_ids]
        target_cross_image_templates = biohashes_by_method[
            method
        ][query_seed][fresh_target_ids]
        baseline_source = raw_to_source(method, baseline_raw)
        baseline_fresh_templates = templates_from_source(
            baseline_source, method, query_seed
        )
        baseline_same_image_fresh_scores = rowwise_hamming_similarity(
            baseline_fresh_templates, target_same_image_templates
        )
        baseline_cross_image_fresh_scores = rowwise_hamming_similarity(
            baseline_fresh_templates, target_cross_image_templates
        )
        baseline_same_image_raw_cosine = np.sum(
            baseline_raw * attack_raw_truth, axis=1
        )
        baseline_cross_image_raw_cosine = np.sum(
            baseline_raw * fresh_raw_truth, axis=1
        )
        protected_threshold_vector = fold_threshold_vector(
            protected_thresholds, method, query_seed
        )
        raw_threshold_vector = fold_threshold_vector(raw_thresholds)

        for position, exposure_seed in enumerate(exposure_order, start=1):
            cumulative += backprojection_cache[method][exposure_seed].astype(
                np.float32
            )
            if position not in EXPOSURE_COUNTS:
                continue

            averaged_backprojection = cumulative / float(position)
            analytical_source = normalize_rows(averaged_backprojection)
            calibrated_source = cross_validated_calibration(
                averaged_backprojection, source_truth, attack_folds
            )
            analytical_raw = source_to_raw(method, analytical_source)
            calibrated_raw = source_to_raw(method, calibrated_source)

            analytical_fresh_templates = templates_from_source(
                raw_to_source(method, analytical_raw), method, query_seed
            )
            calibrated_fresh_templates = templates_from_source(
                raw_to_source(method, calibrated_raw), method, query_seed
            )
            analytical_same_image_fresh_scores = rowwise_hamming_similarity(
                analytical_fresh_templates, target_same_image_templates
            )
            calibrated_same_image_fresh_scores = rowwise_hamming_similarity(
                calibrated_fresh_templates, target_same_image_templates
            )
            analytical_cross_image_fresh_scores = rowwise_hamming_similarity(
                analytical_fresh_templates, target_cross_image_templates
            )
            calibrated_cross_image_fresh_scores = rowwise_hamming_similarity(
                calibrated_fresh_templates, target_cross_image_templates
            )

            analytical_exposed_scores = []
            calibrated_exposed_scores = []
            for used_seed in exposure_order[:position]:
                target_exposed = biohashes_by_method[
                    method
                ][used_seed][attack_target_ids]
                analytical_exposed_scores.append(rowwise_hamming_similarity(
                    templates_from_source(analytical_source, method, used_seed),
                    target_exposed,
                ))
                calibrated_exposed_scores.append(rowwise_hamming_similarity(
                    templates_from_source(calibrated_source, method, used_seed),
                    target_exposed,
                ))

            frame = attack_targets_df.copy()
            frame.insert(0, "method", method)
            frame.insert(1, "trial_id", trial_id)
            frame.insert(2, "query_seed", query_seed)
            frame.insert(3, "exposure_count", position)
            frame["analytical_source_cosine"] = np.sum(
                analytical_source * source_truth, axis=1
            )
            frame["calibrated_source_cosine"] = np.sum(
                calibrated_source * source_truth, axis=1
            )
            frame["analytical_exposed_image_raw_cosine"] = np.sum(
                analytical_raw * attack_raw_truth, axis=1
            )
            frame["calibrated_exposed_image_raw_cosine"] = np.sum(
                calibrated_raw * attack_raw_truth, axis=1
            )
            frame["analytical_cross_image_raw_cosine"] = np.sum(
                analytical_raw * fresh_raw_truth, axis=1
            )
            frame["calibrated_cross_image_raw_cosine"] = np.sum(
                calibrated_raw * fresh_raw_truth, axis=1
            )
            frame["analytical_exposed_consistency"] = np.mean(
                analytical_exposed_scores, axis=0
            )
            frame["calibrated_exposed_consistency"] = np.mean(
                calibrated_exposed_scores, axis=0
            )
            frame["analytical_same_image_fresh_score"] = (
                analytical_same_image_fresh_scores
            )
            frame["calibrated_same_image_fresh_score"] = (
                calibrated_same_image_fresh_scores
            )
            frame["analytical_cross_image_fresh_score"] = (
                analytical_cross_image_fresh_scores
            )
            frame["calibrated_cross_image_fresh_score"] = (
                calibrated_cross_image_fresh_scores
            )
            frame["baseline_same_image_fresh_score"] = (
                baseline_same_image_fresh_scores
            )
            frame["baseline_cross_image_fresh_score"] = (
                baseline_cross_image_fresh_scores
            )
            frame["protected_threshold"] = protected_threshold_vector
            frame["analytical_same_image_fresh_accept"] = (
                analytical_same_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["calibrated_same_image_fresh_accept"] = (
                calibrated_same_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["analytical_cross_image_fresh_accept"] = (
                analytical_cross_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["calibrated_cross_image_fresh_accept"] = (
                calibrated_cross_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["baseline_same_image_fresh_accept"] = (
                baseline_same_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["baseline_cross_image_fresh_accept"] = (
                baseline_cross_image_fresh_scores >= protected_threshold_vector
            ).astype(np.int8)
            frame["raw_threshold"] = raw_threshold_vector
            frame["analytical_same_image_raw_accept"] = (
                frame["analytical_exposed_image_raw_cosine"].to_numpy()
                >= raw_threshold_vector
            ).astype(np.int8)
            frame["calibrated_same_image_raw_accept"] = (
                frame["calibrated_exposed_image_raw_cosine"].to_numpy()
                >= raw_threshold_vector
            ).astype(np.int8)
            frame["analytical_cross_image_raw_accept"] = (
                frame["analytical_cross_image_raw_cosine"].to_numpy()
                >= raw_threshold_vector
            ).astype(np.int8)
            frame["calibrated_cross_image_raw_accept"] = (
                frame["calibrated_cross_image_raw_cosine"].to_numpy()
                >= raw_threshold_vector
            ).astype(np.int8)
            frame["baseline_same_image_raw_cosine"] = baseline_same_image_raw_cosine
            frame["baseline_cross_image_raw_cosine"] = baseline_cross_image_raw_cosine
            frame["baseline_same_image_raw_accept"] = (
                baseline_same_image_raw_cosine >= raw_threshold_vector
            ).astype(np.int8)
            frame["baseline_cross_image_raw_accept"] = (
                baseline_cross_image_raw_cosine >= raw_threshold_vector
            ).astype(np.int8)
            per_image_frames.append(frame)

        print(
            f"Attack complete: {method} | trial {trial_id}/{NUMBER_OF_KEY_TRIALS}"
        )

per_image_attack_df = pd.concat(per_image_frames, ignore_index=True)
expected_rows = (
    len(METHODS) * NUMBER_OF_KEY_TRIALS * len(EXPOSURE_COUNTS)
    * len(attack_targets_df)
)
if len(per_image_attack_df) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} per-image attack rows; found "
        f"{len(per_image_attack_df)}."
    )

if SAVE_PER_IMAGE_RESULTS:
    per_image_attack_df.to_csv(
        OUTPUT_DIR / "per_image_inversion_results.csv.gz",
        index=False,
        compression="gzip",
    )

attack_runtime_audit = {
    "per_image_rows": len(per_image_attack_df),
    "elapsed_seconds": time.perf_counter() - attack_started,
    "saved_per_image_results": SAVE_PER_IMAGE_RESULTS,
}
print(json.dumps(attack_runtime_audit, indent=2))


# Cell 26
trial_summary_df = per_image_attack_df.groupby(
    ["method", "trial_id", "query_seed", "exposure_count"], as_index=False
).agg(
    target_identities=("identity", "size"),
    analytical_source_cosine=("analytical_source_cosine", "mean"),
    calibrated_source_cosine=("calibrated_source_cosine", "mean"),
    analytical_exposed_image_raw_cosine=("analytical_exposed_image_raw_cosine", "mean"),
    calibrated_exposed_image_raw_cosine=("calibrated_exposed_image_raw_cosine", "mean"),
    analytical_cross_image_raw_cosine=("analytical_cross_image_raw_cosine", "mean"),
    calibrated_cross_image_raw_cosine=("calibrated_cross_image_raw_cosine", "mean"),
    analytical_exposed_consistency=("analytical_exposed_consistency", "mean"),
    calibrated_exposed_consistency=("calibrated_exposed_consistency", "mean"),
    analytical_same_image_fresh_key_asr=("analytical_same_image_fresh_accept", "mean"),
    calibrated_same_image_fresh_key_asr=("calibrated_same_image_fresh_accept", "mean"),
    baseline_same_image_fresh_key_asr=("baseline_same_image_fresh_accept", "mean"),
    analytical_cross_image_fresh_key_asr=("analytical_cross_image_fresh_accept", "mean"),
    calibrated_cross_image_fresh_key_asr=("calibrated_cross_image_fresh_accept", "mean"),
    baseline_cross_image_fresh_key_asr=("baseline_cross_image_fresh_accept", "mean"),
    analytical_same_image_raw_asr=("analytical_same_image_raw_accept", "mean"),
    calibrated_same_image_raw_asr=("calibrated_same_image_raw_accept", "mean"),
    baseline_same_image_raw_asr=("baseline_same_image_raw_accept", "mean"),
    analytical_cross_image_raw_asr=("analytical_cross_image_raw_accept", "mean"),
    calibrated_cross_image_raw_asr=("calibrated_cross_image_raw_accept", "mean"),
    baseline_cross_image_raw_asr=("baseline_cross_image_raw_accept", "mean"),
)
trial_summary_df.to_csv(OUTPUT_DIR / "inversion_trial_summary.csv", index=False)

fold_summary_df = per_image_attack_df.groupby(
    ["method", "trial_id", "query_seed", "exposure_count", "fold"],
    as_index=False,
).agg(
    target_identities=("identity", "size"),
    calibrated_exposed_image_raw_cosine=("calibrated_exposed_image_raw_cosine", "mean"),
    calibrated_cross_image_raw_cosine=("calibrated_cross_image_raw_cosine", "mean"),
    calibrated_same_image_fresh_key_asr=("calibrated_same_image_fresh_accept", "mean"),
    calibrated_cross_image_fresh_key_asr=("calibrated_cross_image_fresh_accept", "mean"),
    calibrated_same_image_raw_asr=("calibrated_same_image_raw_accept", "mean"),
    calibrated_cross_image_raw_asr=("calibrated_cross_image_raw_accept", "mean"),
)
fold_summary_df.to_csv(OUTPUT_DIR / "inversion_fold_summary.csv", index=False)


def bootstrap_mean_ci(values, seed, repetitions=BOOTSTRAP_REPETITIONS):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


summary_metrics = [
    "analytical_source_cosine",
    "calibrated_source_cosine",
    "analytical_exposed_image_raw_cosine",
    "calibrated_exposed_image_raw_cosine",
    "analytical_cross_image_raw_cosine",
    "calibrated_cross_image_raw_cosine",
    "analytical_exposed_consistency",
    "calibrated_exposed_consistency",
    "analytical_same_image_fresh_key_asr",
    "calibrated_same_image_fresh_key_asr",
    "baseline_same_image_fresh_key_asr",
    "analytical_cross_image_fresh_key_asr",
    "calibrated_cross_image_fresh_key_asr",
    "baseline_cross_image_fresh_key_asr",
    "analytical_same_image_raw_asr",
    "calibrated_same_image_raw_asr",
    "baseline_same_image_raw_asr",
    "analytical_cross_image_raw_asr",
    "calibrated_cross_image_raw_asr",
    "baseline_cross_image_raw_asr",
]
summary_rows = []
for method in METHODS:
    for exposure_count in EXPOSURE_COUNTS:
        subset = trial_summary_df[
            (trial_summary_df["method"] == method)
            & (trial_summary_df["exposure_count"] == exposure_count)
        ]
        row = {
            "method": method,
            "exposure_count": exposure_count,
            "key_trials": len(subset),
            "target_identities_per_trial": int(subset["target_identities"].iloc[0]),
        }
        for metric_index, metric in enumerate(summary_metrics):
            values = subset[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values,
                BOOTSTRAP_RANDOM_SEED
                + METHODS.index(method) * 10000
                + exposure_count * 100
                + metric_index,
            )
            row[f"mean_{metric}"] = float(values.mean())
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"std_{metric}"] = float(values.std(ddof=1))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        summary_rows.append(row)

inversion_summary_df = pd.DataFrame(summary_rows)
inversion_summary_df.to_csv(OUTPUT_DIR / "inversion_summary_across_trials.csv", index=False)

display(inversion_summary_df[[
    "method",
    "exposure_count",
    "mean_calibrated_exposed_image_raw_cosine",
    "mean_calibrated_cross_image_raw_cosine",
    "mean_calibrated_cross_image_fresh_key_asr",
    "calibrated_cross_image_fresh_key_asr_ci95_low",
    "calibrated_cross_image_fresh_key_asr_ci95_high",
    "mean_baseline_cross_image_fresh_key_asr",
    "mean_calibrated_same_image_fresh_key_asr",
    "mean_calibrated_cross_image_raw_asr",
]])


# Cell 28
growth_metrics = [
    "calibrated_exposed_image_raw_cosine",
    "calibrated_cross_image_raw_cosine",
    "calibrated_cross_image_fresh_key_asr",
    "calibrated_cross_image_raw_asr",
    "calibrated_exposed_consistency",
]
growth_rows = []
for method in METHODS:
    method_trials = trial_summary_df[trial_summary_df["method"] == method]
    for metric_index, metric in enumerate(growth_metrics):
        pivot = method_trials.pivot(
            index="trial_id", columns="exposure_count", values=metric
        )
        differences = (pivot[10] - pivot[1]).to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(
            differences,
            BOOTSTRAP_RANDOM_SEED + 50000
            + METHODS.index(method) * 100 + metric_index,
        )
        growth_rows.append({
            "method": method,
            "metric": metric,
            "paired_key_trials": len(differences),
            "mean_k10_minus_k1": float(differences.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "all_trials_positive": bool((differences > 0).all()),
        })

exposure_growth_df = pd.DataFrame(growth_rows)
exposure_growth_df.to_csv(OUTPUT_DIR / "exposure_growth_k10_minus_k1.csv", index=False)

paired_rows = []
for method in [item for item in METHODS if item != "RO"]:
    for exposure_count in EXPOSURE_COUNTS:
        for metric_index, metric in enumerate(growth_metrics):
            ro = trial_summary_df[
                (trial_summary_df["method"] == "RO")
                & (trial_summary_df["exposure_count"] == exposure_count)
            ].sort_values("trial_id")
            candidate = trial_summary_df[
                (trial_summary_df["method"] == method)
                & (trial_summary_df["exposure_count"] == exposure_count)
            ].sort_values("trial_id")
            if ro["trial_id"].tolist() != candidate["trial_id"].tolist():
                raise RuntimeError("Paired trial alignment failed.")
            differences = (
                candidate[metric].to_numpy(dtype=float)
                - ro[metric].to_numpy(dtype=float)
            )
            low, high = bootstrap_mean_ci(
                differences,
                BOOTSTRAP_RANDOM_SEED + 60000
                + METHODS.index(method) * 1000
                + exposure_count * 20 + metric_index,
            )
            paired_rows.append({
                "comparison": f"{method}_minus_RO",
                "exposure_count": exposure_count,
                "metric": metric,
                "paired_key_trials": len(differences),
                "mean_difference": float(differences.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })

paired_method_df = pd.DataFrame(paired_rows)
paired_method_df.to_csv(OUTPUT_DIR / "paired_method_differences_vs_ro.csv", index=False)
display(exposure_growth_df)


# Cell 30
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


def plot_metric(metric, ylabel, title, filename, chance_line=None):
    plt.figure(figsize=(8.8, 5.4))
    for method in METHODS:
        data = inversion_summary_df[
            inversion_summary_df["method"] == method
        ].sort_values("exposure_count")
        means = data[f"mean_{metric}"].to_numpy(dtype=float)
        lower = means - data[f"{metric}_ci95_low"].to_numpy(dtype=float)
        upper = data[f"{metric}_ci95_high"].to_numpy(dtype=float) - means
        plt.errorbar(
            data["exposure_count"], means,
            yerr=np.vstack([lower, upper]),
            marker="o", linewidth=2, capsize=4,
            color=method_colors[method], label=method_labels[method],
        )
    if chance_line is not None:
        plt.axhline(
            chance_line, color="black", linestyle="--",
            label=f"Reference = {chance_line:.3f}",
        )
    plt.xticks(EXPOSURE_COUNTS)
    plt.xlabel("Number of exposed templates (K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.show()


plot_metric(
    "calibrated_exposed_image_raw_cosine",
    "Mean cosine to exposed-image embedding",
    "Embedding reconstruction quality for the exposed image",
    "reconstruction_cosine_vs_exposures.png",
)
plot_metric(
    "calibrated_cross_image_fresh_key_asr",
    "Cross-image fresh-key attack success rate",
    "Impersonation of an independent same-identity enrollment",
    "fresh_key_asr_vs_exposures.png",
    chance_line=TARGET_FAR,
)
plot_metric(
    "calibrated_exposed_consistency",
    "Mean reproduced fraction of exposed bits",
    "Consistency with stolen templates",
    "exposed_bit_consistency_vs_exposures.png",
    chance_line=0.5,
)

# Attack versus uninformed baseline at K=10.
k10 = inversion_summary_df[
    inversion_summary_df["exposure_count"] == 10
].set_index("method").loc[METHODS]
x = np.arange(len(METHODS))
width = 0.36
plt.figure(figsize=(8.4, 5.2))
plt.bar(
    x - width / 2,
    k10["mean_calibrated_cross_image_fresh_key_asr"],
    width,
    label="Calibrated inversion",
    color=[method_colors[item] for item in METHODS],
)
plt.bar(
    x + width / 2,
    k10["mean_baseline_cross_image_fresh_key_asr"],
    width,
    label="Uninformed mean baseline",
    color="lightgray",
    edgecolor="black",
)
plt.axhline(TARGET_FAR, color="black", linestyle="--", label="Target FAR = 0.1%")
plt.xticks(x, [method_labels[item] for item in METHODS])
plt.ylabel("Cross-image fresh-key attack success rate")
plt.title("K=10 cross-image inversion attack versus uninformed baseline")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "k10_attack_vs_baseline.png", dpi=180)
plt.show()


# Cell 32
RUN_COMPLETED_UTC = datetime.now(timezone.utc).isoformat()

k10_summary = inversion_summary_df[
    inversion_summary_df["exposure_count"] == 10
].copy()
most_resistant_method = k10_summary.loc[
    k10_summary["mean_calibrated_cross_image_fresh_key_asr"].idxmin(), "method"
]
highest_risk_method = k10_summary.loc[
    k10_summary["mean_calibrated_cross_image_fresh_key_asr"].idxmax(), "method"
]

growth_lookup = exposure_growth_df[
    exposure_growth_df["metric"] == "calibrated_cross_image_fresh_key_asr"
].set_index("method")
interpretation = {
    "lowest_mean_k10_cross_image_fresh_key_asr": most_resistant_method,
    "highest_mean_k10_cross_image_fresh_key_asr": highest_risk_method,
    "method_exposure_amplification_ci_excludes_zero": {
        method: bool(growth_lookup.loc[method, "ci95_low"] > 0)
        for method in METHODS
    },
    "embedding_level_only": True,
    "pixel_face_reconstruction_claimed": False,
    "pad_bypass_claimed": False,
}

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "title": EXPERIMENT_TITLE,
    "researcher": RESEARCHER_NAME,
    "institution": INSTITUTION,
    "e06_reference_notebook_sha256": E06_REFERENCE_NOTEBOOK_SHA256,
    "run_started_utc": RUN_STARTED_UTC,
    "run_completed_utc": RUN_COMPLETED_UTC,
    "runtime": runtime_audit,
    "threat_model": threat_model,
    "source_inventory": source_inventory,
    "cache_audit": cache_audit,
    "identity_partition": partition_audit,
    "pca": pca_audit,
    "identity_disjoint_protocol": protocol_audit,
    "targets_and_trials": target_audit,
    "template_generation": template_generation_audit,
    "backprojection": backprojection_audit,
    "attack_runtime": attack_runtime_audit,
    "summary_across_key_trials": json.loads(
        inversion_summary_df.to_json(orient="records")
    ),
    "exposure_growth": json.loads(
        exposure_growth_df.to_json(orient="records")
    ),
    "interpretation": interpretation,
    "scope_limitations": [
        "Evaluation uses frozen MobileFaceNet embeddings from valid LFW images only.",
        "The attacker is given exposed projection operators; this is a key-compromise or worst-case disclosure scenario.",
        "The reconstructed object is an embedding, not a pixel-level face image.",
        "Fresh-key ASR is embedding-level impersonation potential; camera, sensor, liveness, and PAD stages are not modeled.",
        "PCA-guided inversion restores one training-only median PCA magnitude because sign templates discard magnitude.",
        "Ten key trials and two LFW images per selected identity do not establish universal behavior across datasets or encoders.",
    ],
}
with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)

brief_lines = [
    "# LFW-E07_INV — One-Page Experiment Brief",
    "",
    f"**Experiment:** {EXPERIMENT_ID}",
    f"**Researcher:** {RESEARCHER_NAME}",
    f"**Completed (UTC):** {RUN_COMPLETED_UTC}",
    "",
    "## Research question",
    "",
    "Can 1/2/5/10 protected templates and their known projection operators be inverted into an embedding that impersonates the target under a fresh key?",
    "",
    "## Controlled design",
    "",
    f"- Frozen LFW cache: {cache_audit['valid_embeddings']} valid 512-D MobileFaceNet embeddings.",
    f"- Methods: {', '.join(METHODS)}; 128-bit templates; 20 reproducible keys.",
    f"- Targets: {target_audit['total_target_identities']} identities, two different images per identity, balanced across ten identity-disjoint folds.",
    f"- Key trials: {NUMBER_OF_KEY_TRIALS}; exposures K = {EXPOSURE_COUNTS}.",
    "- Primary endpoint: cross-image fresh-key embedding-level ASR at a threshold trained for FAR = 0.1%.",
    "- Attacks: analytical signed backprojection and identity-disjoint diagonal calibration.",
    "",
    "## K=10 primary results",
    "",
    "| Method | Exposed-image raw cosine | Cross-image fresh-key ASR | Baseline ASR | Cross-image raw ASR | Same-image fresh-key ASR |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in k10_summary.itertuples(index=False):
    brief_lines.append(
        f"| {row.method} | {row.mean_calibrated_exposed_image_raw_cosine:.4f} | "
        f"{100*row.mean_calibrated_cross_image_fresh_key_asr:.2f}% | "
        f"{100*row.mean_baseline_cross_image_fresh_key_asr:.2f}% | "
        f"{100*row.mean_calibrated_cross_image_raw_asr:.2f}% | "
        f"{100*row.mean_calibrated_same_image_fresh_key_asr:.2f}% |"
    )

brief_lines.extend([
    "",
    "## Exposure growth (K=10 minus K=1)",
    "",
    "| Method | Raw-cosine change | Fresh-key ASR change | 95% CI for ASR change |",
    "|---|---:|---:|---:|",
])
for method in METHODS:
    raw_growth = exposure_growth_df[
        (exposure_growth_df["method"] == method)
        & (exposure_growth_df["metric"] == "calibrated_exposed_image_raw_cosine")
    ].iloc[0]
    asr_growth = exposure_growth_df[
        (exposure_growth_df["method"] == method)
        & (exposure_growth_df["metric"] == "calibrated_cross_image_fresh_key_asr")
    ].iloc[0]
    brief_lines.append(
        f"| {method} | {raw_growth.mean_k10_minus_k1:+.4f} | "
        f"{100*asr_growth.mean_k10_minus_k1:+.2f} pp | "
        f"[{100*asr_growth.ci95_low:+.2f}, {100*asr_growth.ci95_high:+.2f}] pp |"
    )

brief_lines.extend([
    "",
    "## Conservative interpretation",
    "",
    f"- Lowest mean K=10 cross-image fresh-key ASR: **{most_resistant_method}**.",
    f"- Highest mean K=10 cross-image fresh-key ASR: **{highest_risk_method}**.",
    "- Compare attack ASR with the uninformed baseline and the 0.1% FAR target before calling the leakage practically important.",
    "- A positive paired K=10−K=1 confidence interval supports cumulative multi-template exposure risk.",
    "- This is embedding inversion, not pixel-face reconstruction or proof of PAD bypass.",
    "",
    "## Scope",
    "",
    "One encoder, one dataset, ten key trials, and a key-compromise threat model. Replication on another face dataset and encoder is required before a broad claim.",
])

brief_path = OUTPUT_DIR / "LFW-E07_INV_one_page_brief.md"
brief_path.write_text("\n".join(brief_lines), encoding="utf-8")
print("\n".join(brief_lines))


# Cell 34
RESULTS_ARCHIVE_BASE = Path(
    "/kaggle/working/LFW-E07_INV-embedding-inversion-results"
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
    "If the link does not open, use Kaggle's Output/Files panel and download "
    "LFW-E07_INV-embedding-inversion-results.zip."
)
