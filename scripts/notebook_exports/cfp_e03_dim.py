# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 2
import hashlib
import json
import math
import os
import platform
import shutil
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import FileLink, display
from sklearn.metrics import roc_auc_score, roc_curve

EXPERIMENT_ID = "CFP-E03-DIM"
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()

ON_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_INPUT_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
WORKING_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUTPUT_DIR = WORKING_ROOT / "experiments" / EXPERIMENT_ID
CACHE_OUTPUT_DIR = WORKING_ROOT / "experiments" / "CFP-E03-DIM-protected-template-cache"
INPUT_STAGE_DIR = WORKING_ROOT / "experiments" / "CFP-E03-DIM-input-stage"

# A rerun clears only this experiment's own exact output directories.
for directory in [OUTPUT_DIR, CACHE_OUTPUT_DIR, INPUT_STAGE_DIR]:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

EXPECTED_PAIRS = 7000
EXPECTED_PAIR_SIDES = 14000
EXPECTED_UNIQUE_IMAGES = 5890
EXPECTED_FOLDS = 10
EXPECTED_PAIRS_PER_FOLD = 700
EXPECTED_GENUINE_PER_FOLD = 350
EXPECTED_IMPOSTOR_PER_FOLD = 350
EMBEDDING_DIMENSION = 512
TEMPLATE_BITS = 128
TARGET_FAR = 1e-3

# Frozen experiment keys. Changing this list creates a different experiment.
KEY_SEEDS = [24001, 24002, 24003, 24004, 24005,
             24006, 24007, 24008, 24009, 24010]
UNLINKABILITY_SAMPLING_SEED = 90210

print("Experiment:", EXPERIMENT_ID)
print("Input search root:", KAGGLE_INPUT_ROOT)
print("Results directory:", OUTPUT_DIR)
print("Keys:", KEY_SEEDS)

DIMENSIONS = [64, 128, 256, 512]
ROOT_OUTPUT_DIR = OUTPUT_DIR
ROOT_CACHE_DIR = CACHE_OUTPUT_DIR


# Cell 3
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one_zip(required_terms, human_name):
    candidates = []
    for path in KAGGLE_INPUT_ROOT.rglob("*.zip"):
        lowered = path.name.lower()
        if all(term.lower() in lowered for term in required_terms):
            candidates.append(path)
    candidates = sorted(candidates)
    print(f"{human_name} candidates:")
    for candidate in candidates:
        print(" -", candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {human_name} ZIP, found {len(candidates)}. "
            "Attach the requested E01-R4 ZIP once and remove duplicates."
        )
    return candidates[0]


def safe_extract_zip(zip_path, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(destination)


CACHE_ZIP = find_one_zip(["e01", "r4", "embedding", "cache"], "embedding-cache")
BASELINE_ZIP = find_one_zip(["e01", "r4", "baseline", "results"], "baseline-results")

CACHE_INPUT_DIR = INPUT_STAGE_DIR / "cache"
BASELINE_INPUT_DIR = INPUT_STAGE_DIR / "baseline"
CACHE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
safe_extract_zip(CACHE_ZIP, CACHE_INPUT_DIR)
safe_extract_zip(BASELINE_ZIP, BASELINE_INPUT_DIR)

input_archives = {
    "cache_zip": {"path": str(CACHE_ZIP), "sha256": sha256_file(CACHE_ZIP)},
    "baseline_zip": {"path": str(BASELINE_ZIP), "sha256": sha256_file(BASELINE_ZIP)},
}
print(json.dumps(input_archives, indent=2))


# Cell 4
PAIR_EMBEDDINGS_PATH = CACHE_INPUT_DIR / "aligned_pair_side_embeddings.npy"
PAIR_METADATA_PATH = CACHE_INPUT_DIR / "aligned_pair_side_metadata.csv"
UNIQUE_EMBEDDINGS_PATH = CACHE_INPUT_DIR / "unique_aligned_embeddings.npy"
UNIQUE_METADATA_PATH = CACHE_INPUT_DIR / "unique_aligned_metadata.csv"
BASELINE_RECORD_PATH = BASELINE_INPUT_DIR / "experiment_record.json"

required_paths = [
    PAIR_EMBEDDINGS_PATH, PAIR_METADATA_PATH,
    UNIQUE_EMBEDDINGS_PATH, UNIQUE_METADATA_PATH, BASELINE_RECORD_PATH,
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise RuntimeError("Required files are missing after extraction:\n" + "\n".join(missing))

pair_embeddings = np.load(PAIR_EMBEDDINGS_PATH, allow_pickle=False)
pair_metadata = pd.read_csv(PAIR_METADATA_PATH)
unique_embeddings = np.load(UNIQUE_EMBEDDINGS_PATH, allow_pickle=False)
unique_metadata = pd.read_csv(UNIQUE_METADATA_PATH)
with open(BASELINE_RECORD_PATH, "r", encoding="utf-8") as file:
    baseline_record = json.load(file)

if pair_embeddings.shape != (EXPECTED_PAIR_SIDES, EMBEDDING_DIMENSION):
    raise RuntimeError(f"Unexpected pair embedding shape: {pair_embeddings.shape}")
if unique_embeddings.shape != (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION):
    raise RuntimeError(f"Unexpected unique embedding shape: {unique_embeddings.shape}")
if len(pair_metadata) != EXPECTED_PAIR_SIDES or len(unique_metadata) != EXPECTED_UNIQUE_IMAGES:
    raise RuntimeError("Embedding and metadata row counts do not agree.")
if not np.isfinite(pair_embeddings).all() or not np.isfinite(unique_embeddings).all():
    raise RuntimeError("The cache contains NaN or infinite embedding values.")
if not np.allclose(np.linalg.norm(pair_embeddings, axis=1), 1.0, atol=1e-4):
    raise RuntimeError("Pair-side embeddings are not L2-normalized.")
if not np.allclose(np.linalg.norm(unique_embeddings, axis=1), 1.0, atol=1e-4):
    raise RuntimeError("Unique embeddings are not L2-normalized.")

expected_bin_index = np.arange(EXPECTED_PAIR_SIDES)
expected_pair_id = np.repeat(np.arange(1, EXPECTED_PAIRS + 1), 2)
expected_side = np.tile(np.array(["a", "b"]), EXPECTED_PAIRS)
if not np.array_equal(pair_metadata["bin_index"].to_numpy(), expected_bin_index):
    raise RuntimeError("Pair-side bin_index ordering is invalid.")
if not np.array_equal(pair_metadata["pair_id"].to_numpy(), expected_pair_id):
    raise RuntimeError("Each pair_id must have exactly two adjacent sides.")
if not np.array_equal(pair_metadata["side"].astype(str).to_numpy(), expected_side):
    raise RuntimeError("Pair-side ordering must be a,b for every pair.")

labels_a = pair_metadata.iloc[0::2]["label"].to_numpy(dtype=np.int8)
labels_b = pair_metadata.iloc[1::2]["label"].to_numpy(dtype=np.int8)
folds_a = pair_metadata.iloc[0::2]["fold"].to_numpy(dtype=np.int8)
folds_b = pair_metadata.iloc[1::2]["fold"].to_numpy(dtype=np.int8)
if not np.array_equal(labels_a, labels_b) or not set(np.unique(labels_a)).issubset({0, 1}):
    raise RuntimeError("Pair labels are invalid or disagree between sides.")
if not np.array_equal(folds_a, folds_b):
    raise RuntimeError("Fold IDs disagree between pair sides.")

fold_audit_rows = []
for fold in range(1, EXPECTED_FOLDS + 1):
    fold_labels = labels_a[folds_a == fold]
    fold_audit_rows.append({
        "fold": fold,
        "pairs": len(fold_labels),
        "genuine": int(fold_labels.sum()),
        "impostor": int((fold_labels == 0).sum()),
    })
fold_audit_df = pd.DataFrame(fold_audit_rows)
if not (
    fold_audit_df["pairs"].eq(EXPECTED_PAIRS_PER_FOLD).all()
    and fold_audit_df["genuine"].eq(EXPECTED_GENUINE_PER_FOLD).all()
    and fold_audit_df["impostor"].eq(EXPECTED_IMPOSTOR_PER_FOLD).all()
):
    raise RuntimeError("The frozen CFP-FP ten-fold protocol audit failed.")

unique_ids = pair_metadata["unique_image_id"].to_numpy(dtype=np.int64)
if unique_ids.min() != 0 or unique_ids.max() != EXPECTED_UNIQUE_IMAGES - 1:
    raise RuntimeError("Occurrence-to-unique-image IDs are out of range.")
if not np.allclose(pair_embeddings, unique_embeddings[unique_ids], atol=1e-6):
    raise RuntimeError("Pair embeddings do not reproduce from the unique-image mapping.")

labels = labels_a
folds = folds_a
pair_ids = np.arange(1, EXPECTED_PAIRS + 1)

baseline_accuracy = float(baseline_record["results"]["accuracy"][0]["mean_fold_accuracy"])
baseline_auc = float(baseline_record["results"]["roc_eer"][0]["auc"])
baseline_eer = float(baseline_record["results"]["roc_eer"][0]["eer"])
baseline_tar = float(baseline_record["results"]["tar_at_far_1e-3"][0]["pooled_tar"])
baseline_observed_far = float(
    baseline_record["results"]["tar_at_far_1e-3"][0]["pooled_observed_far"]
)

cache_audit = {
    "pair_embedding_shape": list(pair_embeddings.shape),
    "unique_embedding_shape": list(unique_embeddings.shape),
    "pair_embeddings_sha256": sha256_file(PAIR_EMBEDDINGS_PATH),
    "pair_metadata_sha256": sha256_file(PAIR_METADATA_PATH),
    "unique_embeddings_sha256": sha256_file(UNIQUE_EMBEDDINGS_PATH),
    "unique_metadata_sha256": sha256_file(UNIQUE_METADATA_PATH),
    "l2_normalized": True,
    "fold_protocol_valid": True,
    "occurrence_to_unique_mapping_valid": True,
}
with open(OUTPUT_DIR / "input_cache_audit.json", "w", encoding="utf-8") as file:
    json.dump(cache_audit, file, indent=2)
fold_audit_df.to_csv(OUTPUT_DIR / "fold_protocol_audit.csv", index=False)

display(fold_audit_df)
print("Cache and protocol audit passed.")
print(
    f"E01-R4 baseline — accuracy {baseline_accuracy:.3%}, AUC {baseline_auc:.3%}, "
    f"EER {baseline_eer:.3%}, TAR@FAR 1e-3 {baseline_tar:.3%}"
)


# Cell 6
E02_ZIP = find_one_zip(["e02", "128bit", "results"], "E02 128-bit results")
E02_INPUT_DIR = INPUT_STAGE_DIR / "e02"
E02_INPUT_DIR.mkdir(exist_ok=True)
safe_extract_zip(E02_ZIP, E02_INPUT_DIR)
input_archives["e02_results_zip"] = {"path": str(E02_ZIP), "sha256": sha256_file(E02_ZIP)}
e02_record = json.loads((E02_INPUT_DIR / "experiment_record.json").read_text())
e02_scores = pd.read_csv(E02_INPUT_DIR / "same_key_pair_scores_all_keys.csv")
e02_keys = pd.read_csv(E02_INPUT_DIR / "verification_metrics_by_key.csv")
for field in ["pair_embeddings_sha256", "pair_metadata_sha256", "unique_embeddings_sha256", "unique_metadata_sha256"]:
    if cache_audit[field] != e02_record["cache_audit"][field]:
        raise RuntimeError("E02 input mismatch: " + field)
if e02_record["protection"]["key_seeds"] != KEY_SEEDS:
    raise RuntimeError("E02 seeds differ.")
print("E02 input provenance matches.")


# Cell 8
def select_accuracy_threshold(scores, labels):
    """Exact maximum-accuracy threshold in O(n log n), choosing the highest threshold on ties."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n = len(labels)
    correct = int((labels == 0).sum())  # threshold above max: predict all impostor
    best_accuracy = correct / n
    best_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        group = sorted_labels[i:j]
        correct += int((group == 1).sum()) - int((group == 0).sum())
        accuracy = correct / n
        threshold = float(sorted_scores[i])
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            best_threshold = threshold
        i = j
    return best_threshold, float(best_accuracy)


def evaluate_at_threshold(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    predictions = (scores >= threshold).astype(np.int8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "pairs": len(labels), "genuine_pairs": tp + fn, "impostor_pairs": tn + fp,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "tar": tp / (tp + fn), "far": fp / (tn + fp), "frr": fn / (tp + fn),
        "predictions": predictions,
    }


def calculate_auc_eer(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    auc_value = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossings = np.where(difference[:-1] * difference[1:] <= 0)[0]
    if len(crossings):
        index = min(crossings, key=lambda i: abs(difference[i]) + abs(difference[i + 1]))
        denominator = difference[index + 1] - difference[index]
        weight = 0.5 if denominator == 0 else -difference[index] / denominator
        eer = fpr[index] + weight * (fpr[index + 1] - fpr[index])
        if np.isfinite(thresholds[index]) and np.isfinite(thresholds[index + 1]):
            eer_threshold = thresholds[index] + weight * (thresholds[index + 1] - thresholds[index])
        else:
            closest = int(np.argmin(np.abs(difference)))
            eer_threshold = thresholds[closest]
    else:
        closest = int(np.argmin(np.abs(difference)))
        eer = (fpr[closest] + fnr[closest]) / 2.0
        eer_threshold = thresholds[closest]
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    return {
        "pairs": len(labels), "genuine_pairs": len(genuine), "impostor_pairs": len(impostor),
        "auc": auc_value, "eer": float(eer), "eer_threshold": float(eer_threshold),
        "genuine_mean": float(genuine.mean()), "genuine_std": float(genuine.std(ddof=1)),
        "impostor_mean": float(impostor.mean()), "impostor_std": float(impostor.std(ddof=1)),
    }


def select_threshold_at_far(scores, labels, target_far):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    impostor = scores[labels == 0]
    allowed_fp = int(np.floor(target_far * len(impostor)))
    descending = np.sort(impostor)[::-1]
    threshold = np.nextafter(descending[allowed_fp], np.inf)
    achieved_fp = int((impostor >= threshold).sum())
    achieved_far = achieved_fp / len(impostor)
    if achieved_far > target_far:
        raise RuntimeError("Training FAR exceeded the target.")
    return {
        "threshold": float(threshold),
        "training_impostor_pairs": len(impostor),
        "training_allowed_fp": allowed_fp,
        "training_achieved_fp": achieved_fp,
        "training_achieved_far": achieved_far,
    }


def pooled_counts(frame):
    tp, tn, fp, fn = [int(frame[column].sum()) for column in ["tp", "tn", "fp", "fn"]]
    genuine, impostor = tp + fn, tn + fp
    return tp, tn, fp, fn, genuine, impostor, genuine + impostor


print("Evaluation functions defined.")


# Cell 10
raw_scores = np.sum(pair_embeddings[0::2] * pair_embeddings[1::2], axis=1, dtype=np.float64)
raw_rows = []
for f in range(1, 11):
    train, test = folds != f, folds == f
    cal = select_threshold_at_far(raw_scores[train], labels[train], TARGET_FAR)
    row = evaluate_at_threshold(raw_scores[test], labels[test], cal["threshold"])
    row.pop("predictions")
    raw_rows.append({"fold": f, **cal, **row})
raw_df = pd.DataFrame(raw_rows)
tp, tn, fp, fn, ng, ni, nt = pooled_counts(raw_df)
if not np.isclose(tp/ng, baseline_tar, atol=1e-12) or not np.isclose(fp/ni, baseline_observed_far, atol=1e-12):
    raise RuntimeError("Raw low-FAR baseline mismatch: investigate before comparing dimensions.")
raw_df.to_csv(ROOT_OUTPUT_DIR / "raw_baseline_recheck.csv", index=False)
print("Raw TAR/FAR reproduced:", tp/ng, fp/ni)


# Cell 12
def run_dimension(TEMPLATE_BITS):
    OUTPUT_DIR = ROOT_OUTPUT_DIR / f"bits_{TEMPLATE_BITS}"
    CACHE_OUTPUT_DIR = ROOT_CACHE_DIR / f"bits_{TEMPLATE_BITS}"
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_OUTPUT_DIR.mkdir(exist_ok=True)
    def make_projection(seed, input_dimension=EMBEDDING_DIMENSION, output_dimension=TEMPLATE_BITS):
        rng = np.random.default_rng(seed)
        gaussian = rng.standard_normal((input_dimension, output_dimension), dtype=np.float64)
        q, r = np.linalg.qr(gaussian, mode="reduced")
        # Fix QR sign ambiguity so the same input and seed reproduce exactly.
        diagonal_sign = np.where(np.diag(r) < 0.0, -1.0, 1.0)
        q = q * diagonal_sign[None, :]
        return q.astype(np.float32)


    def protect_embeddings(embeddings, projection):
        return (embeddings @ projection >= 0.0)


    def binary_entropy(probability_one):
        p = np.clip(np.asarray(probability_one, dtype=np.float64), 1e-12, 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


    protected_unique_bits = np.empty(
        (len(KEY_SEEDS), EXPECTED_UNIQUE_IMAGES, TEMPLATE_BITS), dtype=np.bool_
    )
    template_quality_rows = []
    projection_checksums = {}

    for key_index, seed in enumerate(KEY_SEEDS):
        projection = make_projection(seed)
        np.save(CACHE_OUTPUT_DIR / f"projection_{seed}.npy", projection)
        orthogonality_error = float(
            np.max(np.abs(projection.T @ projection - np.eye(TEMPLATE_BITS)))
        )
        if orthogonality_error > 2e-5:
            raise RuntimeError(f"Projection orthogonality audit failed for seed {seed}.")
        bits = protect_embeddings(unique_embeddings, projection)
        protected_unique_bits[key_index] = bits
        bit_one_rates = bits.mean(axis=0)
        entropy = binary_entropy(bit_one_rates)
        centered = bits.astype(np.float32) - bit_one_rates[None, :]
        correlation = np.corrcoef(centered, rowvar=False)
        off_diagonal = np.abs(correlation[~np.eye(TEMPLATE_BITS, dtype=bool)])
        projection_checksums[str(seed)] = hashlib.sha256(projection.tobytes()).hexdigest()
        template_quality_rows.append({
            "key_index": key_index,
            "key_seed": seed,
            "orthogonality_max_abs_error": orthogonality_error,
            "mean_bit_one_rate": float(bit_one_rates.mean()),
            "minimum_bit_one_rate": float(bit_one_rates.min()),
            "maximum_bit_one_rate": float(bit_one_rates.max()),
            "mean_bit_entropy": float(entropy.mean()),
            "minimum_bit_entropy": float(entropy.min()),
            "mean_absolute_interbit_correlation": float(np.nanmean(off_diagonal)),
        })

    # Exact reproducibility check for the first key.
    first_projection_again = make_projection(KEY_SEEDS[0])
    first_bits_again = protect_embeddings(unique_embeddings, first_projection_again)
    if not np.array_equal(first_bits_again, protected_unique_bits[0]):
        raise RuntimeError("Key reproducibility audit failed.")

    template_quality_df = pd.DataFrame(template_quality_rows)
    template_quality_df.to_csv(OUTPUT_DIR / "template_quality_by_key.csv", index=False)
    display(template_quality_df)
    print("All protected-template generation audits passed.")

    accuracy_fold_rows = []
    roc_fold_rows = []
    low_far_fold_rows = []
    key_summary_rows = []
    pair_score_rows = []
    representative_scores = None

    for key_index, seed in enumerate(KEY_SEEDS):
        occurrence_bits = protected_unique_bits[key_index][unique_ids]
        bits_a = occurrence_bits[0::2]
        bits_b = occurrence_bits[1::2]
        scores = np.mean(bits_a == bits_b, axis=1, dtype=np.float64)
        if representative_scores is None:
            representative_scores = scores.copy()

        for pair_id, fold, label, score in zip(pair_ids, folds, labels, scores):
            pair_score_rows.append({
                "key_index": key_index, "key_seed": seed, "pair_id": int(pair_id),
                "fold": int(fold), "label": int(label),
                "pair_type": "genuine" if label else "impostor",
                "hamming_similarity": float(score),
            })

        key_accuracy_rows = []
        key_roc_rows = []
        key_low_far_rows = []
        for test_fold in range(1, EXPECTED_FOLDS + 1):
            train_mask = folds != test_fold
            test_mask = folds == test_fold

            accuracy_threshold, training_accuracy = select_accuracy_threshold(
                scores[train_mask], labels[train_mask]
            )
            accuracy_metrics = evaluate_at_threshold(
                scores[test_mask], labels[test_mask], accuracy_threshold
            )
            accuracy_metrics.pop("predictions")
            accuracy_row = {
                "key_index": key_index, "key_seed": seed, "fold": test_fold,
                "threshold": accuracy_threshold, "training_accuracy": training_accuracy,
                **accuracy_metrics,
            }
            accuracy_fold_rows.append(accuracy_row)
            key_accuracy_rows.append(accuracy_row)

            roc_row = {
                "key_index": key_index, "key_seed": seed, "fold": test_fold,
                **calculate_auc_eer(labels[test_mask], scores[test_mask]),
            }
            roc_fold_rows.append(roc_row)
            key_roc_rows.append(roc_row)

            calibration = select_threshold_at_far(scores[train_mask], labels[train_mask], TARGET_FAR)
            low_far_metrics = evaluate_at_threshold(
                scores[test_mask], labels[test_mask], calibration["threshold"]
            )
            low_far_metrics.pop("predictions")
            low_far_row = {
                "key_index": key_index, "key_seed": seed, "fold": test_fold,
                "target_training_far": TARGET_FAR, **calibration, **low_far_metrics,
            }
            low_far_fold_rows.append(low_far_row)
            key_low_far_rows.append(low_far_row)

        key_accuracy_df = pd.DataFrame(key_accuracy_rows)
        key_roc_df = pd.DataFrame(key_roc_rows)
        key_low_far_df = pd.DataFrame(key_low_far_rows)
        tp, tn, fp, fn, genuine, impostor, total = pooled_counts(key_accuracy_df)
        ltp, ltn, lfp, lfn, lgenuine, limpostor, ltotal = pooled_counts(key_low_far_df)
        pooled_roc = calculate_auc_eer(labels, scores)
        key_summary_rows.append({
            "key_index": key_index,
            "key_seed": seed,
            "mean_accuracy_threshold": float(key_accuracy_df["threshold"].mean()),
            "threshold_std": float(key_accuracy_df["threshold"].std(ddof=1)),
            "mean_fold_accuracy": float(key_accuracy_df["accuracy"].mean()),
            "fold_accuracy_std": float(key_accuracy_df["accuracy"].std(ddof=1)),
            "pooled_accuracy": (tp + tn) / total,
            "pooled_tar_at_accuracy_threshold": tp / genuine,
            "pooled_far_at_accuracy_threshold": fp / impostor,
            "pooled_frr_at_accuracy_threshold": fn / genuine,
            "pooled_auc": pooled_roc["auc"],
            "pooled_eer": pooled_roc["eer"],
            "pooled_eer_threshold": pooled_roc["eer_threshold"],
            "genuine_score_mean": pooled_roc["genuine_mean"],
            "genuine_score_std": pooled_roc["genuine_std"],
            "impostor_score_mean": pooled_roc["impostor_mean"],
            "impostor_score_std": pooled_roc["impostor_std"],
            "mean_low_far_threshold": float(key_low_far_df["threshold"].mean()),
            "low_far_threshold_std": float(key_low_far_df["threshold"].std(ddof=1)),
            "pooled_tar_at_far_1e_3": ltp / lgenuine,
            "pooled_observed_far_at_far_1e_3": lfp / limpostor,
            "pooled_frr_at_far_1e_3": lfn / lgenuine,
            "mean_fold_tar_at_far_1e_3": float(key_low_far_df["tar"].mean()),
            "fold_tar_std_at_far_1e_3": float(key_low_far_df["tar"].std(ddof=1)),
        })
        print(f"Completed key {key_index + 1}/{len(KEY_SEEDS)} (seed {seed})")

    accuracy_fold_df = pd.DataFrame(accuracy_fold_rows)
    roc_fold_df = pd.DataFrame(roc_fold_rows)
    low_far_fold_df = pd.DataFrame(low_far_fold_rows)
    key_summary_df = pd.DataFrame(key_summary_rows)
    pair_scores_df = pd.DataFrame(pair_score_rows)

    accuracy_fold_df.to_csv(OUTPUT_DIR / "accuracy_fold_results_all_keys.csv", index=False)
    roc_fold_df.to_csv(OUTPUT_DIR / "roc_fold_metrics_all_keys.csv", index=False)
    low_far_fold_df.to_csv(OUTPUT_DIR / "tar_at_far_1e-3_fold_results_all_keys.csv", index=False)
    key_summary_df.to_csv(OUTPUT_DIR / "verification_metrics_by_key.csv", index=False)
    pair_scores_df.to_csv(OUTPUT_DIR / "same_key_pair_scores_all_keys.csv", index=False)

    display(key_summary_df)

    def summarize_across_keys(frame, column):
        values = frame[column].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        # Student-t critical value for 10 keys and 95% two-sided CI (df=9).
        t_critical = 2.2621571627409915
        half_width = t_critical * std / math.sqrt(len(values))
        return mean, std, mean - half_width, mean + half_width


    metric_map = {
        "mean_fold_accuracy": "Mean fold accuracy",
        "pooled_auc": "Pooled AUC",
        "pooled_eer": "Pooled EER",
        "pooled_tar_at_far_1e_3": "TAR at training FAR 1e-3",
        "pooled_observed_far_at_far_1e_3": "Observed FAR at training FAR 1e-3",
    }
    aggregate_rows = []
    for column, label in metric_map.items():
        mean, std, ci_low, ci_high = summarize_across_keys(key_summary_df, column)
        aggregate_rows.append({
            "metric": label, "column": column, "keys": len(KEY_SEEDS),
            "mean": mean, "std_across_keys": std,
            "ci95_low": ci_low, "ci95_high": ci_high,
        })
    aggregate_summary_df = pd.DataFrame(aggregate_rows)

    protected_accuracy = float(key_summary_df["mean_fold_accuracy"].mean())
    protected_auc = float(key_summary_df["pooled_auc"].mean())
    protected_eer = float(key_summary_df["pooled_eer"].mean())
    protected_tar = float(key_summary_df["pooled_tar_at_far_1e_3"].mean())
    protected_far = float(key_summary_df["pooled_observed_far_at_far_1e_3"].mean())

    utility_comparison_df = pd.DataFrame([
        {"metric": "Mean fold accuracy", "E01_R4_raw": baseline_accuracy,
         "E03_protected_mean": protected_accuracy, "delta_protected_minus_raw": protected_accuracy - baseline_accuracy},
        {"metric": "Pooled AUC", "E01_R4_raw": baseline_auc,
         "E03_protected_mean": protected_auc, "delta_protected_minus_raw": protected_auc - baseline_auc},
        {"metric": "Pooled EER", "E01_R4_raw": baseline_eer,
         "E03_protected_mean": protected_eer, "delta_protected_minus_raw": protected_eer - baseline_eer},
        {"metric": "TAR at training FAR 1e-3", "E01_R4_raw": baseline_tar,
         "E03_protected_mean": protected_tar, "delta_protected_minus_raw": protected_tar - baseline_tar},
        {"metric": "Observed FAR at training FAR 1e-3", "E01_R4_raw": baseline_observed_far,
         "E03_protected_mean": protected_far, "delta_protected_minus_raw": protected_far - baseline_observed_far},
    ])

    aggregate_summary_df.to_csv(OUTPUT_DIR / "protected_metrics_across_keys_summary.csv", index=False)
    utility_comparison_df.to_csv(OUTPUT_DIR / "utility_delta_vs_E01_R4.csv", index=False)

    print("Protected results across keys:")
    display(aggregate_summary_df)
    print("Utility comparison:")
    display(utility_comparison_df)

    unlinkability_rows = []
    unlinkability_sample_rows = []
    pooled_link_labels = []
    pooled_link_scores = []

    for first_key_index in range(len(KEY_SEEDS)):
        for second_key_index in range(first_key_index + 1, len(KEY_SEEDS)):
            first_bits = protected_unique_bits[first_key_index]
            second_bits = protected_unique_bits[second_key_index]
            same_image_scores = np.mean(first_bits == second_bits, axis=1, dtype=np.float64)

            rng = np.random.default_rng(
                UNLINKABILITY_SAMPLING_SEED + 1000 * first_key_index + second_key_index
            )
            shift = int(rng.integers(1, EXPECTED_UNIQUE_IMAGES))
            different_indices = np.roll(np.arange(EXPECTED_UNIQUE_IMAGES), shift)
            different_image_scores = np.mean(
                first_bits == second_bits[different_indices], axis=1, dtype=np.float64
            )

            link_labels = np.concatenate([
                np.ones(EXPECTED_UNIQUE_IMAGES, dtype=np.int8),
                np.zeros(EXPECTED_UNIQUE_IMAGES, dtype=np.int8),
            ])
            link_scores = np.concatenate([same_image_scores, different_image_scores])
            raw_auc = float(roc_auc_score(link_labels, link_scores))
            attacker_auc = max(raw_auc, 1.0 - raw_auc)
            mean_gap = float(same_image_scores.mean() - different_image_scores.mean())

            unlinkability_rows.append({
                "first_key_index": first_key_index,
                "first_key_seed": KEY_SEEDS[first_key_index],
                "second_key_index": second_key_index,
                "second_key_seed": KEY_SEEDS[second_key_index],
                "unique_images_per_class": EXPECTED_UNIQUE_IMAGES,
                "nonmated_circular_shift": shift,
                "same_image_mean_hamming_similarity": float(same_image_scores.mean()),
                "same_image_std_hamming_similarity": float(same_image_scores.std(ddof=1)),
                "different_image_mean_hamming_similarity": float(different_image_scores.mean()),
                "different_image_std_hamming_similarity": float(different_image_scores.std(ddof=1)),
                "same_minus_different_mean_gap": mean_gap,
                "raw_linkability_auc": raw_auc,
                "attacker_oriented_auc": attacker_auc,
            })
            pooled_link_labels.append(link_labels)
            pooled_link_scores.append(link_scores)

            # A compact, deterministic sample is saved for plots and audit.
            sample_size = min(1000, EXPECTED_UNIQUE_IMAGES)
            sample_indices = rng.choice(EXPECTED_UNIQUE_IMAGES, size=sample_size, replace=False)
            for index in sample_indices:
                unlinkability_sample_rows.append({
                    "first_key_seed": KEY_SEEDS[first_key_index],
                    "second_key_seed": KEY_SEEDS[second_key_index],
                    "comparison": "same_image",
                    "hamming_similarity": float(same_image_scores[index]),
                })
                unlinkability_sample_rows.append({
                    "first_key_seed": KEY_SEEDS[first_key_index],
                    "second_key_seed": KEY_SEEDS[second_key_index],
                    "comparison": "different_image",
                    "hamming_similarity": float(different_image_scores[index]),
                })

    unlinkability_df = pd.DataFrame(unlinkability_rows)
    unlinkability_sample_df = pd.DataFrame(unlinkability_sample_rows)
    pooled_link_labels = np.concatenate(pooled_link_labels)
    pooled_link_scores = np.concatenate(pooled_link_scores)
    pooled_raw_linkability_auc = float(roc_auc_score(pooled_link_labels, pooled_link_scores))
    pooled_attacker_auc = max(pooled_raw_linkability_auc, 1.0 - pooled_raw_linkability_auc)

    unlinkability_summary = {
        "key_pairs": int(len(unlinkability_df)),
        "comparison_boundary": "same encoded aligned image across keys versus different encoded images",
        "mean_same_image_hamming_similarity": float(
            unlinkability_df["same_image_mean_hamming_similarity"].mean()
        ),
        "mean_different_image_hamming_similarity": float(
            unlinkability_df["different_image_mean_hamming_similarity"].mean()
        ),
        "mean_absolute_same_minus_different_gap": float(
            unlinkability_df["same_minus_different_mean_gap"].abs().mean()
        ),
        "mean_attacker_oriented_auc_across_key_pairs": float(
            unlinkability_df["attacker_oriented_auc"].mean()
        ),
        "worst_attacker_oriented_auc": float(
            unlinkability_df["attacker_oriented_auc"].max()
        ),
        "pooled_raw_linkability_auc": pooled_raw_linkability_auc,
        "pooled_attacker_oriented_auc": pooled_attacker_auc,
    }

    unlinkability_df.to_csv(OUTPUT_DIR / "cross_key_unlinkability_by_key_pair.csv", index=False)
    unlinkability_sample_df.to_csv(OUTPUT_DIR / "cross_key_unlinkability_score_sample.csv", index=False)
    with open(OUTPUT_DIR / "cross_key_unlinkability_summary.json", "w", encoding="utf-8") as file:
        json.dump(unlinkability_summary, file, indent=2)

    display(unlinkability_df)
    print(json.dumps(unlinkability_summary, indent=2))

    packed_unique_templates = np.packbits(protected_unique_bits, axis=-1, bitorder="big")
    expected_packed_shape = (len(KEY_SEEDS), EXPECTED_UNIQUE_IMAGES, TEMPLATE_BITS // 8)
    if packed_unique_templates.shape != expected_packed_shape:
        raise RuntimeError(f"Unexpected packed-template shape: {packed_unique_templates.shape}")
    if not np.array_equal(
        np.unpackbits(packed_unique_templates, axis=-1, count=TEMPLATE_BITS, bitorder="big").astype(bool),
        protected_unique_bits,
    ):
        raise RuntimeError("Packed-template round-trip audit failed.")

    benchmark_embeddings = unique_embeddings[:256]
    benchmark_projection = make_projection(KEY_SEEDS[0])
    _ = protect_embeddings(benchmark_embeddings, benchmark_projection)  # warm-up
    latency_ms_per_template = []
    for _ in range(20):
        started = time.perf_counter()
        _ = protect_embeddings(benchmark_embeddings, benchmark_projection)
        elapsed = time.perf_counter() - started
        latency_ms_per_template.append(1000.0 * elapsed / len(benchmark_embeddings))

    raw_bytes_per_template = EMBEDDING_DIMENSION * np.dtype(np.float32).itemsize
    protected_bytes_per_template = TEMPLATE_BITS // 8
    efficiency_summary = {
        "raw_embedding_bytes_per_template": raw_bytes_per_template,
        "protected_packed_bytes_per_template": protected_bytes_per_template,
        "storage_reduction_factor": raw_bytes_per_template / protected_bytes_per_template,
        "benchmark_batch_size": len(benchmark_embeddings),
        "transform_latency_ms_per_template_p50": float(np.percentile(latency_ms_per_template, 50)),
        "transform_latency_ms_per_template_p95": float(np.percentile(latency_ms_per_template, 95)),
        "latency_note": "NumPy batch projection plus sign binarization on the current runtime; excludes embedding extraction.",
    }
    pd.DataFrame([efficiency_summary]).to_csv(OUTPUT_DIR / "efficiency_summary.csv", index=False)
    print(json.dumps(efficiency_summary, indent=2))

    np.save(CACHE_OUTPUT_DIR / "templates_packed.npy", packed_unique_templates)
    config = {"bits": TEMPLATE_BITS, "seeds": KEY_SEEDS, "projection_sha256": projection_checksums,
              "matrix_layout": "independent 512 x d row-major Gaussian for each dimension; not nested",
              "binarization": ">=0", "bitorder": "big", "input_cache_audit": cache_audit}
    (CACHE_OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    if TEMPLATE_BITS == 128:
        score_columns = ["key_seed", "pair_id", "fold", "label", "hamming_similarity"]
        left = pair_scores_df[score_columns].sort_values(["key_seed", "pair_id"]).reset_index(drop=True)
        right = e02_scores[score_columns].sort_values(["key_seed", "pair_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
        metric_columns = ["mean_fold_accuracy", "pooled_auc", "pooled_eer", "pooled_tar_at_far_1e_3", "pooled_observed_far_at_far_1e_3"]
        a = key_summary_df.sort_values("key_seed")[metric_columns].to_numpy()
        b = e02_keys.sort_values("key_seed")[metric_columns].to_numpy()
        if not np.allclose(a, b, atol=1e-12, rtol=0):
            raise RuntimeError("128-bit E02 metric reproduction failed.")
        (ROOT_OUTPUT_DIR / "e02_control_check.json").write_text(json.dumps({"passed": True, "all_70000_pair_scores_exact": True, "metric_tolerance": 1e-12}, indent=2))
        print("PASS: all 70,000 E02 scores reproduced exactly.")
    summary = {"bits": TEMPLATE_BITS, "packed_bytes": TEMPLATE_BITS // 8, "storage_reduction": 2048/(TEMPLATE_BITS//8),
               "tar_delta_vs_raw_pp": 100*(protected_tar-baseline_tar),
               "cross_key_mean_oriented_auc": unlinkability_summary["mean_attacker_oriented_auc_across_key_pairs"],
               "cross_key_worst_oriented_auc": unlinkability_summary["worst_attacker_oriented_auc"],
               "cross_key_pooled_oriented_auc": pooled_attacker_auc,
               "latency_ms_batch_per_template_p50": efficiency_summary["transform_latency_ms_per_template_p50"]}
    for _, r in aggregate_summary_df.iterrows():
        for stat in ["mean", "std_across_keys", "ci95_low", "ci95_high"]:
            summary[r["column"] + "_" + stat] = float(r[stat])
    key_summary_df["bits"] = TEMPLATE_BITS
    return summary, key_summary_df


# Cell 14
dimension_rows, all_key_frames = [], []
for bits in DIMENSIONS:
    print("\nRunning", bits, "bits", flush=True)
    summary, keys = run_dimension(bits)
    dimension_rows.append(summary)
    all_key_frames.append(keys)
summary_df = pd.DataFrame(dimension_rows)
all_keys_df = pd.concat(all_key_frames, ignore_index=True)
ref_tar = float(summary_df.loc[summary_df.bits == 128, "pooled_tar_at_far_1e_3_mean"].iloc[0])
summary_df["tar_delta_vs_128_pp"] = 100*(summary_df["pooled_tar_at_far_1e_3_mean"]-ref_tar)
summary_df.to_csv(ROOT_OUTPUT_DIR / "dimension_summary.csv", index=False)
all_keys_df.to_csv(ROOT_OUTPUT_DIR / "all_dimensions_key_metrics.csv", index=False)
display(summary_df[["bits", "packed_bytes", "pooled_tar_at_far_1e_3_mean", "pooled_observed_far_at_far_1e_3_mean", "tar_delta_vs_128_pp", "cross_key_worst_oriented_auc"]])


# Cell 16
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
x = summary_df.bits
for ax, column, title in [(axes[0,0], "pooled_tar_at_far_1e_3", "Genuine comparisons accepted"), (axes[0,1], "pooled_eer", "Equal error rate")]:
    y = summary_df[column+"_mean"] * 100
    err = (summary_df[column+"_ci95_high"]-summary_df[column+"_mean"]) * 100
    ax.errorbar(x, y, yerr=err, marker="o", capsize=4)
    ax.set_title(title + " (95% key intervals)")
    ax.set_ylabel("Percent")
axes[0,0].axhline(100*baseline_tar, color="gray", linestyle="--", label="Raw baseline")
axes[0,0].legend()
axes[1,0].plot(x, 100*summary_df.pooled_observed_far_at_far_1e_3_mean, marker="o")
axes[1,0].axhline(100*TARGET_FAR, color="gray", linestyle="--", label="Training target")
axes[1,0].set_title("Observed held-out FAR, mean across keys")
axes[1,0].set_ylabel("Percent")
axes[1,0].legend()
axes[1,1].plot(x, summary_df.cross_key_mean_oriented_auc, marker="o", label="Mean key-pair AUC")
axes[1,1].plot(x, summary_df.cross_key_worst_oriented_auc, marker="s", label="Worst key-pair AUC")
axes[1,1].axhline(.5, color="gray", linestyle="--")
axes[1,1].set_title("Same-image cross-key linkage (descriptive)")
axes[1,1].set_ylabel("Oriented AUC")
axes[1,1].legend()
for ax in axes.flat:
    ax.set_xticks(DIMENSIONS)
    ax.set_xlabel("Template bits")
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(ROOT_OUTPUT_DIR / "dimension_tradeoff.png", dpi=180)
plt.show()


# Cell 18
tracker = []
lines = ["# CFP-E03-DIM results", "", "128-bit control: all 70,000 E02 pair scores reproduced exactly.", "",
         "| Bits | Bytes | TAR (%) | Observed FAR (%) | EER (%) | TAR change vs 128 (pp) | Worst cross-key AUC |",
         "|---|---|---|---|---|---|---|"]
for _, r in summary_df.iterrows():
    tar, far, eer = [100*r[c+"_mean"] for c in ["pooled_tar_at_far_1e_3", "pooled_observed_far_at_far_1e_3", "pooled_eer"]]
    result = f"TAR {tar:.3f}%; observed FAR {far:.3f}%; EER {eer:.3f}%; {int(r.packed_bytes)} bytes; TAR change vs 128 bits {r.tar_delta_vs_128_pp:+.3f} percentage points; worst cross-key AUC {r.cross_key_worst_oriented_auc:.4f}"
    tracker.append({"Experiment": f"CFP-E03-DIM-{int(r.bits)}", "Main question": "How does template length affect cross-pose utility, compactness and same-image cross-key linkage?", "Important metric": "TAR at training FAR 0.1%; observed FAR; EER; bytes; cross-key AUC", "Actual result": result, "Simple interpretation": "Exploratory comparison across ten seeds on fixed pairs; does not establish stronger security or identity unlinkability."})
    lines.append(f"| {int(r.bits)} | {int(r.packed_bytes)} | {tar:.3f} | {far:.3f} | {eer:.3f} | {r.tar_delta_vs_128_pp:+.3f} | {r.cross_key_worst_oriented_auc:.4f} |")
lines += ["", f"Raw E01-R4 TAR: {baseline_tar:.3%}; observed FAR: {baseline_observed_far:.3%}.", "",
          "TAR is measured using thresholds learned on nine folds; actual test FAR can exceed the 0.1% training target.",
          "All dimensions are reported. No final deployment dimension is selected by this notebook.",
          "Intervals in dimension_summary.csv describe key variability, not uncertainty over new identities or datasets.",
          "Cross-key scores compare identical encoded images against different encoded images. Different images are not verified different people.",
          "Length-specific projections are not nested. At 512 bits sign binarization still discards information.",
          "This is an ablation supporting method design, not a standalone novelty or security claim."]
brief = "\n".join(lines)
(ROOT_OUTPUT_DIR / "CFP-E03-DIM_brief.md").write_text(brief)
pd.DataFrame(tracker).to_csv(ROOT_OUTPUT_DIR / "tracker_rows.csv", index=False)
print(brief)
record = {"experiment": EXPERIMENT_ID, "started_utc": RUN_STARTED_UTC, "completed_utc": datetime.now(timezone.utc).isoformat(),
          "dimensions": DIMENSIONS, "key_seeds": KEY_SEEDS, "unlinkability_sampling_seed": UNLINKABILITY_SAMPLING_SEED,
          "inputs": input_archives, "cache_audit": cache_audit,
          "python": platform.python_version(), "platform": platform.platform(),
          "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
          "thread_environment": {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
          "protocol": "Frozen E01-R4 cached pair folds; 9-fold threshold training; no identity-disjoint claim",
          "target_training_far": TARGET_FAR, "projection_design": "E02 exact generator independently shaped at each dimension; not nested",
          "interpretation": lines[-7:], "results": json.loads(summary_df.to_json(orient="records"))}
(ROOT_OUTPUT_DIR / "experiment_record.json").write_text(json.dumps(record, indent=2))
pair_metadata.to_csv(ROOT_CACHE_DIR / "aligned_pair_side_metadata.csv", index=False)
unique_metadata.to_csv(ROOT_CACHE_DIR / "unique_aligned_metadata.csv", index=False)
for directory in [ROOT_OUTPUT_DIR, ROOT_CACHE_DIR]:
    manifest = [{"path": p.relative_to(directory).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
for name, directory in [("CFP-E03-DIM-results", ROOT_OUTPUT_DIR), ("CFP-E03-DIM-protected-template-cache", ROOT_CACHE_DIR)]:
    archive = Path(shutil.make_archive(str(WORKING_ROOT / name), "zip", root_dir=directory))
    display(FileLink(str(archive)))
print("Download both ZIPs and save the executed notebook.")
