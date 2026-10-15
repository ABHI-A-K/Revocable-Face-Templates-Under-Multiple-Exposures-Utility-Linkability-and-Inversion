# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 3
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

EXPERIMENT_ID = "CFP-FP-E02-seeded-RO-128bit"
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()

ON_KAGGLE = Path("/kaggle/input").exists()
KAGGLE_INPUT_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
WORKING_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUTPUT_DIR = WORKING_ROOT / "experiments" / EXPERIMENT_ID
CACHE_OUTPUT_DIR = WORKING_ROOT / "experiments" / "CFP-FP-E02-protected-template-cache"
INPUT_STAGE_DIR = WORKING_ROOT / "experiments" / "CFP-FP-E02-input-stage"

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


# Cell 5
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
RUN_PURPOSE = (
    "Measure same-key frontal-to-profile verification utility and cross-key same-image "
    "unlinkability after converting frozen 512-D CFP-FP MobileFaceNet embeddings into "
    "128-bit seeded Random Orthogonal templates."
)
HYPOTHESIS = (
    "The 128-bit templates will preserve most of the aligned E01-R4 verification utility, "
    "while independent keys will make same-image cross-key Hamming scores statistically "
    "similar to different-image cross-key scores."
)
RUN_NOTES = "Fresh CFP-FP-E02 run using the verified CFP-FP-E01-R4 cache."

print(RUN_PURPOSE)


# Cell 7
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


# Cell 9
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


# Cell 11
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


# Cell 13
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


# Cell 15
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


# Cell 17
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
     "E02_protected_mean": protected_accuracy, "delta_protected_minus_raw": protected_accuracy - baseline_accuracy},
    {"metric": "Pooled AUC", "E01_R4_raw": baseline_auc,
     "E02_protected_mean": protected_auc, "delta_protected_minus_raw": protected_auc - baseline_auc},
    {"metric": "Pooled EER", "E01_R4_raw": baseline_eer,
     "E02_protected_mean": protected_eer, "delta_protected_minus_raw": protected_eer - baseline_eer},
    {"metric": "TAR at training FAR 1e-3", "E01_R4_raw": baseline_tar,
     "E02_protected_mean": protected_tar, "delta_protected_minus_raw": protected_tar - baseline_tar},
    {"metric": "Observed FAR at training FAR 1e-3", "E01_R4_raw": baseline_observed_far,
     "E02_protected_mean": protected_far, "delta_protected_minus_raw": protected_far - baseline_observed_far},
])

aggregate_summary_df.to_csv(OUTPUT_DIR / "protected_metrics_across_keys_summary.csv", index=False)
utility_comparison_df.to_csv(OUTPUT_DIR / "utility_delta_vs_E01_R4.csv", index=False)

print("Protected results across keys:")
display(aggregate_summary_df)
print("Utility comparison:")
display(utility_comparison_df)


# Cell 19
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


# Cell 21
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


# Cell 23
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Utility: raw versus mean protected values.
metric_names = ["Accuracy", "AUC", "TAR@FAR 1e-3"]
raw_values = [baseline_accuracy, baseline_auc, baseline_tar]
protected_values = [protected_accuracy, protected_auc, protected_tar]
x = np.arange(len(metric_names))
width = 0.36
axes[0, 0].bar(x - width / 2, 100 * np.array(raw_values), width, label="E01-R4 raw")
axes[0, 0].bar(x + width / 2, 100 * np.array(protected_values), width, label="E02 protected mean")
axes[0, 0].set_xticks(x, metric_names)
axes[0, 0].set_ylabel("percent")
axes[0, 0].set_title("Raw versus protected utility")
axes[0, 0].legend()

# Same-key score separation for the first key.
axes[0, 1].hist(representative_scores[labels == 0], bins=40, alpha=0.7, density=True, label="impostor")
axes[0, 1].hist(representative_scores[labels == 1], bins=40, alpha=0.7, density=True, label="genuine")
axes[0, 1].set_xlabel("Hamming similarity")
axes[0, 1].set_ylabel("density")
axes[0, 1].set_title(f"Same-key scores (seed {KEY_SEEDS[0]})")
axes[0, 1].legend()

# Low-FAR TAR stability across keys.
axes[1, 0].plot(
    key_summary_df["key_index"] + 1,
    100 * key_summary_df["pooled_tar_at_far_1e_3"],
    marker="o",
)
axes[1, 0].axhline(100 * baseline_tar, linestyle="--", color="black", label="E01-R4 raw")
axes[1, 0].set_xticks(range(1, len(KEY_SEEDS) + 1))
axes[1, 0].set_xlabel("key number")
axes[1, 0].set_ylabel("TAR (%)")
axes[1, 0].set_title("TAR@FAR 1e-3 across keys")
axes[1, 0].legend()

# Cross-key score overlap using the saved balanced sample.
same_sample = unlinkability_sample_df.loc[
    unlinkability_sample_df["comparison"] == "same_image", "hamming_similarity"
]
different_sample = unlinkability_sample_df.loc[
    unlinkability_sample_df["comparison"] == "different_image", "hamming_similarity"
]
axes[1, 1].hist(different_sample, bins=40, alpha=0.7, density=True, label="different image")
axes[1, 1].hist(same_sample, bins=40, alpha=0.7, density=True, label="same image")
axes[1, 1].axvline(0.5, linestyle="--", color="black")
axes[1, 1].set_xlabel("cross-key Hamming similarity")
axes[1, 1].set_ylabel("density")
axes[1, 1].set_title(f"Cross-key unlinkability (attacker AUC {pooled_attacker_auc:.3f})")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cfp_fp_e02_diagnostics.png", dpi=180, bbox_inches="tight")
plt.show()


# Cell 25
def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


RUN_COMPLETED_UTC = datetime.now(timezone.utc).isoformat()
runtime_audit = {
    "python": platform.python_version(),
    "operating_system": platform.platform(),
    "packages": {
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "matplotlib": package_version("matplotlib"),
    },
}

protection_config = {
    "input_dimension": EMBEDDING_DIMENSION,
    "output_bits": TEMPLATE_BITS,
    "projection": "Gaussian matrix followed by reduced QR; orthonormal columns",
    "binarization": "projected value >= 0",
    "similarity": "fraction of equal bits (Hamming similarity)",
    "key_seeds": KEY_SEEDS,
    "projection_sha256_by_seed": projection_checksums,
    "packed_bitorder": "big",
    "security_boundary": (
        "Seeds are public reproducibility identifiers. E02 evaluates verification utility "
        "and same-image cross-key unlinkability, not cryptographic irreversibility under key exposure."
    ),
}
with open(OUTPUT_DIR / "protection_config.json", "w", encoding="utf-8") as file:
    json.dump(protection_config, file, indent=2)

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "title": "Seeded Random Orthogonal 128-bit protected templates on aligned CFP-FP",
    "researcher": RESEARCHER_NAME,
    "institution": INSTITUTION,
    "run_started_utc": RUN_STARTED_UTC,
    "run_completed_utc": RUN_COMPLETED_UTC,
    "purpose": RUN_PURPOSE,
    "hypothesis": HYPOTHESIS,
    "input_archives": input_archives,
    "cache_audit": cache_audit,
    "frozen_protocol": {
        "pairs": EXPECTED_PAIRS,
        "pair_sides": EXPECTED_PAIR_SIDES,
        "unique_encoded_images": EXPECTED_UNIQUE_IMAGES,
        "folds": EXPECTED_FOLDS,
        "pairs_per_fold": EXPECTED_PAIRS_PER_FOLD,
        "genuine_per_fold": EXPECTED_GENUINE_PER_FOLD,
        "impostor_per_fold": EXPECTED_IMPOSTOR_PER_FOLD,
        "threshold_training": "nine folds; held-out fold used only for testing",
        "target_training_far": TARGET_FAR,
        "preprocessing": "reuse E01-R4 aligned embeddings; no SCRFD or re-alignment",
    },
    "protection": protection_config,
    "runtime": runtime_audit,
    "baseline_E01_R4": {
        "mean_fold_accuracy": baseline_accuracy,
        "pooled_auc": baseline_auc,
        "pooled_eer": baseline_eer,
        "pooled_tar_at_far_1e_3": baseline_tar,
        "pooled_observed_far_at_far_1e_3": baseline_observed_far,
    },
    "protected_results_across_keys": json.loads(aggregate_summary_df.to_json(orient="records")),
    "utility_delta_vs_E01_R4": json.loads(utility_comparison_df.to_json(orient="records")),
    "unlinkability": unlinkability_summary,
    "efficiency": efficiency_summary,
    "limitations": [
        "The experiment begins from pre-aligned E01-R4 embeddings and does not measure raw-image detection.",
        "The cache has no verified person identities, so unlinkability is tested for the same encoded image across keys, not different images of the same person.",
        "E02 does not test embedding reconstruction, inversion, or a key-exposure attacker.",
        "A seeded NumPy generator provides experimental reproducibility; it is not a production key-management design.",
        "The held-out fold FAR granularity is approximately 1/350 = 0.286%.",
    ],
}
with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)


def pct(value):
    return f"{100 * float(value):.3f}%"


accuracy_stats = aggregate_summary_df.loc[
    aggregate_summary_df["column"] == "mean_fold_accuracy"
].iloc[0]
auc_stats = aggregate_summary_df.loc[aggregate_summary_df["column"] == "pooled_auc"].iloc[0]
eer_stats = aggregate_summary_df.loc[aggregate_summary_df["column"] == "pooled_eer"].iloc[0]
tar_stats = aggregate_summary_df.loc[
    aggregate_summary_df["column"] == "pooled_tar_at_far_1e_3"
].iloc[0]

brief = f"""# CFP-FP-E02 — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

Can frozen 512-D MobileFaceNet embeddings be converted into compact, key-controlled 128-bit templates while preserving frontal-to-profile verification utility and reducing cross-key linkability?

## Controlled design

- Reused the verified E01-R4 aligned cache: 14,000 pair sides, 7,000 pairs, 5,890 unique encoded images.
- Ten official contiguous folds; each fold has 350 genuine and 350 impostor pairs.
- Ten independent seeded Random Orthogonal projections, 512-D to 128 values, followed by sign binarization.
- Same-key Hamming similarity for verification.
- Every threshold learned from nine folds and evaluated on the held-out fold.
- No SCRFD, image decoding, re-alignment, embedding re-extraction, or model retraining.

## Main results

| Metric | E01-R4 raw | E02 protected mean across keys | 95% CI across keys |
|---|---:|---:|---:|
| Mean fold accuracy | {pct(baseline_accuracy)} | {pct(accuracy_stats['mean'])} | {pct(accuracy_stats['ci95_low'])}–{pct(accuracy_stats['ci95_high'])} |
| Pooled AUC | {pct(baseline_auc)} | {pct(auc_stats['mean'])} | {pct(auc_stats['ci95_low'])}–{pct(auc_stats['ci95_high'])} |
| Pooled EER | {pct(baseline_eer)} | {pct(eer_stats['mean'])} | {pct(eer_stats['ci95_low'])}–{pct(eer_stats['ci95_high'])} |
| TAR at training FAR 0.1% | {pct(baseline_tar)} | {pct(tar_stats['mean'])} | {pct(tar_stats['ci95_low'])}–{pct(tar_stats['ci95_high'])} |

## Cross-key unlinkability

- Key pairs evaluated: {unlinkability_summary['key_pairs']}
- Mean same-image cross-key Hamming similarity: {unlinkability_summary['mean_same_image_hamming_similarity']:.4f}
- Mean different-image cross-key Hamming similarity: {unlinkability_summary['mean_different_image_hamming_similarity']:.4f}
- Pooled attacker-oriented linkability AUC: {unlinkability_summary['pooled_attacker_oriented_auc']:.4f}
- Worst key-pair attacker-oriented AUC: {unlinkability_summary['worst_attacker_oriented_auc']:.4f}

Values near 0.5 for cross-key Hamming similarity and linkability AUC indicate that changing the key makes same-image templates difficult to distinguish from unrelated-image templates.

## Compactness

- Raw float32 embedding: {efficiency_summary['raw_embedding_bytes_per_template']:,} bytes
- Packed protected template: {efficiency_summary['protected_packed_bytes_per_template']} bytes
- Storage reduction: {efficiency_summary['storage_reduction_factor']:.0f}×

## Interpretation boundary

This experiment establishes protected-template **utility and same-image cross-key unlinkability** on the aligned CFP-FP protocol. It does not establish cryptographic irreversibility, resistance to key exposure, reconstruction resistance, or identity-disjoint attack performance.
"""
(OUTPUT_DIR / "CFP-FP-E02_one_page_brief.md").write_text(brief, encoding="utf-8")
(OUTPUT_DIR / "RESULTS_README.md").write_text(brief + f"\n\n## Run notes\n\n{RUN_NOTES}\n", encoding="utf-8")

tracker_row_df = pd.DataFrame([{
    "Experiment": "CFP-FP-E02",
    "Main question": "Can 512-D CFP-FP embeddings become revocable 128-bit RO templates while retaining utility and resisting cross-key linkage?",
    "Important metric": "TAR@FAR 0.1%; EER; AUC; accuracy; cross-key linkability AUC",
    "Actual result": (
        f"Protected TAR {pct(protected_tar)}; EER {pct(protected_eer)}; "
        f"AUC {pct(protected_auc)}; accuracy {pct(protected_accuracy)}; "
        f"pooled attacker linkability AUC {unlinkability_summary['pooled_attacker_oriented_auc']:.4f}"
    ),
    "Simple interpretation": (
        "Measures the utility cost of 128-bit RO protection and whether independently keyed "
        "templates of the same aligned image look unlinkable."
    ),
}])
tracker_row_df.to_csv(OUTPUT_DIR / "result_tracker_row.csv", index=False)

print(brief)
display(tracker_row_df)


# Cell 27
np.save(CACHE_OUTPUT_DIR / "protected_unique_templates_packed.npy", packed_unique_templates)
np.save(CACHE_OUTPUT_DIR / "key_seeds.npy", np.asarray(KEY_SEEDS, dtype=np.int64))
pair_metadata.to_csv(CACHE_OUTPUT_DIR / "aligned_pair_side_metadata.csv", index=False)
unique_metadata.to_csv(CACHE_OUTPUT_DIR / "unique_aligned_metadata.csv", index=False)
with open(CACHE_OUTPUT_DIR / "protection_config.json", "w", encoding="utf-8") as file:
    json.dump(protection_config, file, indent=2)


def file_manifest(root):
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


# Create manifests after all substantive files exist. The manifest does not hash itself.
file_manifest(OUTPUT_DIR).to_csv(OUTPUT_DIR / "result_file_manifest.csv", index=False)
file_manifest(CACHE_OUTPUT_DIR).to_csv(CACHE_OUTPUT_DIR / "cache_file_manifest.csv", index=False)

results_archive = Path(shutil.make_archive(
    str(WORKING_ROOT / "CFP-FP-E02-seeded-RO-128bit-results"),
    "zip", root_dir=OUTPUT_DIR,
))
cache_archive = Path(shutil.make_archive(
    str(WORKING_ROOT / "CFP-FP-E02-protected-template-cache"),
    "zip", root_dir=CACHE_OUTPUT_DIR,
))

os.chdir(WORKING_ROOT)
print("Results archive:", results_archive, f"({results_archive.stat().st_size / 1024**2:.2f} MiB)")
display(FileLink(results_archive.name))
print("Protected-template cache:", cache_archive, f"({cache_archive.stat().st_size / 1024**2:.2f} MiB)")
display(FileLink(cache_archive.name))
print("If a link does not open, download the ZIP from Kaggle's Output/Files panel.")
