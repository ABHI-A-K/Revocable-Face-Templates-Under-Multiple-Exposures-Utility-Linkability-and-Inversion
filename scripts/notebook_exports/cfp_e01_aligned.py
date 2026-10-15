# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 3
import subprocess
import sys

print("Python executable:", sys.executable)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "insightface==1.0.1", "huggingface_hub>=0.26,<2",
])

# InsightFace may pull the CPU ONNX Runtime wheel. Remove both variants, then
# install the GPU wheel last so it owns the shared `onnxruntime` module.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
    check=False,
    stdout=subprocess.DEVNULL,
)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "onnxruntime-gpu==1.26.0",
])

print("Installation complete.")
if "onnxruntime" in sys.modules:
    print("IMPORTANT: ONNX Runtime was already imported. Restart the session, then Run All.")


# Cell 5
import hashlib
import json
import os
import pickle
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import FileLink, display
from sklearn.metrics import roc_auc_score, roc_curve

EXPERIMENT_ID = "CFP-FP-E01-R4-aligned-baseline"
PIPELINE_REVISION = "R4"
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
CACHE_DIR = Path("/kaggle/working/experiments/CFP-FP-E01-R4-aligned-cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PAIRS = 7000
EXPECTED_SIDES = 2
EXPECTED_OCCURRENCES = EXPECTED_PAIRS * EXPECTED_SIDES
EXPECTED_FOLDS = 10
EXPECTED_PAIRS_PER_FOLD = 700
EXPECTED_GENUINE_PER_FOLD = 350
EXPECTED_IMPOSTOR_PER_FOLD = 350
EXPECTED_IMAGE_SIZE = (112, 112)
EMBEDDING_DIMENSION = 512
TARGET_FAR = 1e-3
BATCH_SIZE = 256
CHECKPOINT_INTERVAL = 1000

# Frozen methodological decisions.
USE_FLIP_AUGMENTATION = False
SIMILARITY = "cosine"
FOLD_RULE = "contiguous official order: 700 pairs per fold"
FAILURE_RULE = "block all metrics unless all 14,000 pair-side images decode and embed"

# Prior raw-pipeline observations are recorded as context, not recomputed here.
PRIOR_RAW_TRANSFER = {
    "R1_image_coverage": 0.0573,
    "R2_image_coverage": 0.4766,
    "R3_image_coverage": 0.4970,
    "interpretation": "raw detector/alignment transfer failure; not a MobileFaceNet recognition score",
}

print("Experiment:", EXPERIMENT_ID)
print("Results:", OUTPUT_DIR)
print("Cache:", CACHE_DIR)


# Cell 7
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
RUN_PURPOSE = (
    "Measure the frozen MobileFaceNet recognition baseline on the standard aligned CFP-FP "
    "frontal-to-profile verification package after raw preprocessing transfer failed."
)
HYPOTHESIS = (
    "Extreme pose will reduce recognition utility relative to LFW, while complete aligned input "
    "coverage will permit a valid benchmark independent of raw face-detection failure."
)
RUN_NOTES = "Fresh R4 run. R1-R3 caches intentionally excluded."

research_log = f"""# Research log — {EXPERIMENT_ID}

- Started (UTC): {RUN_STARTED_UTC}
- Researcher: {RESEARCHER_NAME}
- Institution: {INSTITUTION}
- Purpose: {RUN_PURPOSE}
- Hypothesis: {HYPOTHESIS}
- Notes: {RUN_NOTES}
"""
(OUTPUT_DIR / "research_log.md").write_text(research_log, encoding="utf-8")
print(research_log)


# Cell 9
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot see a CUDA GPU. Select a Kaggle T4 accelerator, restart, and Run All."
    )

# Loading PyTorch first makes its compatible CUDA/cuDNN libraries available to ONNX Runtime.
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

import insightface

package_names = [
    "numpy", "pandas", "opencv-python", "Pillow", "scikit-learn",
    "matplotlib", "onnx", "onnxruntime-gpu", "insightface",
]
package_versions = {}
for package_name in package_names:
    try:
        package_versions[package_name] = version(package_name)
    except PackageNotFoundError:
        package_versions[package_name] = "NOT INSTALLED"

runtime_audit = {
    "python": sys.version.split()[0],
    "operating_system": platform.platform(),
    "gpu": torch.cuda.get_device_name(0),
    "pytorch_cuda_available": bool(torch.cuda.is_available()),
    "pytorch_cuda_build": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "onnxruntime_providers": ort.get_available_providers(),
    "packages": package_versions,
}
print(json.dumps(runtime_audit, indent=2))

if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError(
        "The T4 is visible, but ONNX Runtime did not load CUDAExecutionProvider. "
        "Restart the Kaggle session and Run All from the beginning."
    )
print("CUDAExecutionProvider audit passed.")


# Cell 11
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


candidates = sorted(KAGGLE_INPUT_ROOT.rglob("cfp_fp.bin"))
print("Candidates:")
for candidate in candidates:
    print(" -", candidate)

source_kind = "attached_kaggle_input"
published_fallback_sha256 = "76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370"

if len(candidates) == 0:
    print("No attached cfp_fp.bin found; using the pinned verified fallback.")
    from huggingface_hub import hf_hub_download

    # Use the ordinary HTTPS download route; no Hugging Face account or token is required.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    downloaded_path = hf_hub_download(
        repo_id="Icar/val_sets",
        filename="cfp_fp.bin",
        repo_type="dataset",
        revision="1f7800d9d5ba295554573a4e292e870b64a49b74",
        local_dir="/kaggle/working/cfp_fp_verified_source",
    )
    candidates = [Path(downloaded_path)]
    source_kind = "huggingface:Icar/val_sets@1f7800d9d5ba295554573a4e292e870b64a49b74"

if len(candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one cfp_fp.bin below {KAGGLE_INPUT_ROOT}, found {len(candidates)}. "
        "Attach only the Arcface Dataset input, or remove duplicate inputs."
    )

CFP_FP_BIN = candidates[0]
observed_bin_sha256 = sha256_file(CFP_FP_BIN)
if source_kind.startswith("huggingface") and observed_bin_sha256 != published_fallback_sha256:
    raise RuntimeError(
        "Downloaded cfp_fp.bin failed its published SHA-256 check. "
        f"Observed {observed_bin_sha256}; expected {published_fallback_sha256}. File not loaded."
    )

try:
    with open(CFP_FP_BIN, "rb") as file:
        bins, issame_list = pickle.load(file)
except UnicodeDecodeError:
    with open(CFP_FP_BIN, "rb") as file:
        bins, issame_list = pickle.load(file, encoding="bytes")

labels = np.asarray(issame_list, dtype=np.int8)
if len(labels) != EXPECTED_PAIRS:
    raise RuntimeError(f"Expected {EXPECTED_PAIRS} labels, found {len(labels)}.")
if len(bins) != EXPECTED_OCCURRENCES:
    raise RuntimeError(f"Expected {EXPECTED_OCCURRENCES} encoded images, found {len(bins)}.")
if not set(np.unique(labels)).issubset({0, 1}):
    raise RuntimeError("Labels are not binary genuine/impostor values.")

fold_rows = []
for fold in range(1, EXPECTED_FOLDS + 1):
    start = (fold - 1) * EXPECTED_PAIRS_PER_FOLD
    stop = fold * EXPECTED_PAIRS_PER_FOLD
    fold_labels = labels[start:stop]
    fold_rows.append({
        "fold": fold,
        "pairs": len(fold_labels),
        "genuine": int(fold_labels.sum()),
        "impostor": int((fold_labels == 0).sum()),
    })

fold_audit_df = pd.DataFrame(fold_rows)
display(fold_audit_df)
if not (
    fold_audit_df["pairs"].eq(EXPECTED_PAIRS_PER_FOLD).all()
    and fold_audit_df["genuine"].eq(EXPECTED_GENUINE_PER_FOLD).all()
    and fold_audit_df["impostor"].eq(EXPECTED_IMPOSTOR_PER_FOLD).all()
):
    raise RuntimeError(
        "The cfp_fp.bin ordering does not match the expected official 10-fold layout. Metrics blocked."
    )

bin_audit = {
    "path": str(CFP_FP_BIN),
    "source_kind": source_kind,
    "size_bytes": CFP_FP_BIN.stat().st_size,
    "sha256": observed_bin_sha256,
    "published_fallback_sha256": published_fallback_sha256,
    "pairs": int(len(labels)),
    "encoded_pair_sides": int(len(bins)),
    "genuine_pairs": int(labels.sum()),
    "impostor_pairs": int((labels == 0).sum()),
    "folds": EXPECTED_FOLDS,
    "fold_rule": FOLD_RULE,
}
with open(OUTPUT_DIR / "aligned_bin_audit.json", "w", encoding="utf-8") as file:
    json.dump(bin_audit, file, indent=2)
fold_audit_df.to_csv(OUTPUT_DIR / "fold_protocol_audit.csv", index=False)
print(json.dumps(bin_audit, indent=2))
print("Aligned CFP-FP package audit passed.")


# Cell 13
from insightface.app import FaceAnalysis

MODEL_ROOT = Path("/kaggle/working/insightface_models")
MODEL_DIR = MODEL_ROOT / "models" / "buffalo_sc"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

model_pack = FaceAnalysis(
    name="buffalo_sc",
    root=str(MODEL_ROOT),
    allowed_modules=["detection", "recognition"],
    providers=providers,
)
model_pack.prepare(ctx_id=0, det_thresh=0.50, det_size=(640, 640))
mobilefacenet = model_pack.models["recognition"]

model_path = MODEL_DIR / "w600k_mbf.onnx"
expected_hash = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"
if not model_path.is_file():
    raise FileNotFoundError(f"Missing frozen recognition model: {model_path}")
observed_hash = sha256_file(model_path)
if observed_hash != expected_hash:
    raise RuntimeError(
        f"MobileFaceNet hash mismatch. Observed {observed_hash}; expected {expected_hash}."
    )
session_providers = mobilefacenet.session.get_providers()
if "CUDAExecutionProvider" not in session_providers:
    raise RuntimeError(f"MobileFaceNet is not using CUDA: {session_providers}")

model_audit = {
    "model_pack": "buffalo_sc",
    "recognizer": "MobileFaceNet w600k_mbf",
    "model_sha256": observed_hash,
    "input": "pre-aligned 112x112 BGR decoded from cfp_fp.bin",
    "embedding_dimension": EMBEDDING_DIMENSION,
    "l2_normalization": True,
    "flip_augmentation": USE_FLIP_AUGMENTATION,
    "similarity": SIMILARITY,
    "session_providers": session_providers,
}
print(json.dumps(model_audit, indent=2))
print("Frozen MobileFaceNet audit passed.")


# Cell 15
def decode_aligned(encoded):
    array = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[:2] != EXPECTED_IMAGE_SIZE:
        raise RuntimeError(
            f"Expected aligned size {EXPECTED_IMAGE_SIZE}, found {image.shape[:2]}. "
            "Automatic resizing is intentionally disabled."
        )
    return image


selected_pair_indices = [
    int(np.flatnonzero(labels == 1)[0]),
    int(np.flatnonzero(labels == 0)[0]),
]
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
smoke_rows = []

for row, pair_index in enumerate(selected_pair_indices):
    pair_images = []
    for side in range(2):
        bin_index = 2 * pair_index + side
        image = decode_aligned(bins[bin_index])
        if image is None:
            raise RuntimeError(f"Smoke-test decode failed at bin index {bin_index}.")
        pair_images.append(image)
        axes[row, side].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, side].set_title(
            f"{'genuine' if labels[pair_index] else 'impostor'} | side {'AB'[side]}\n"
            f"pair {pair_index + 1}, fold {pair_index // 700 + 1}"
        )
        axes[row, side].axis("off")

    raw_features = mobilefacenet.get_feat(pair_images).astype(np.float32)
    norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
    pair_embeddings = raw_features / np.clip(norms, 1e-12, None)
    cosine = float(pair_embeddings[0] @ pair_embeddings[1])
    smoke_rows.append({
        "pair_id": pair_index + 1,
        "fold": pair_index // 700 + 1,
        "label": int(labels[pair_index]),
        "pair_type": "genuine" if labels[pair_index] else "impostor",
        "cosine_similarity": cosine,
    })

plt.suptitle("Decoded aligned CFP-FP smoke test")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aligned_smoke_test.png", dpi=180, bbox_inches="tight")
plt.show()

smoke_test_df = pd.DataFrame(smoke_rows)
smoke_test_df.to_csv(OUTPUT_DIR / "smoke_test_results.csv", index=False)
display(smoke_test_df)
print("Smoke test passed.")


# Cell 17
embeddings_path = CACHE_DIR / "aligned_pair_side_embeddings.npy"
metadata_path = CACHE_DIR / "aligned_pair_side_metadata.csv"

embeddings = np.full((EXPECTED_OCCURRENCES, EMBEDDING_DIMENSION), np.nan, dtype=np.float32)
metadata_rows = []
extraction_started = time.perf_counter()

for batch_start in range(0, EXPECTED_OCCURRENCES, BATCH_SIZE):
    batch_stop = min(batch_start + BATCH_SIZE, EXPECTED_OCCURRENCES)
    decoded_images = []
    batch_records = []

    for bin_index in range(batch_start, batch_stop):
        pair_index = bin_index // 2
        side = "a" if bin_index % 2 == 0 else "b"
        encoded = bins[bin_index]
        image = decode_aligned(encoded)
        status = "valid" if image is not None else "decode_failure"
        batch_records.append({
            "bin_index": bin_index,
            "pair_id": pair_index + 1,
            "side": side,
            "fold": pair_index // EXPECTED_PAIRS_PER_FOLD + 1,
            "label": int(labels[pair_index]),
            "pair_type": "genuine" if labels[pair_index] else "impostor",
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            "status": status,
        })
        decoded_images.append(image)

    valid_positions = [i for i, image in enumerate(decoded_images) if image is not None]
    if valid_positions:
        valid_images = [decoded_images[i] for i in valid_positions]
        raw_features = mobilefacenet.get_feat(valid_images).astype(np.float32)
        norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
        normalized = raw_features / np.clip(norms, 1e-12, None)
        for local_position, embedding in zip(valid_positions, normalized):
            embeddings[batch_start + local_position] = embedding

    metadata_rows.extend(batch_records)
    if batch_stop % CHECKPOINT_INTERVAL < BATCH_SIZE or batch_stop == EXPECTED_OCCURRENCES:
        print(f"Processed {batch_stop:,} / {EXPECTED_OCCURRENCES:,} pair-side images")

extraction_seconds = time.perf_counter() - extraction_started
metadata_df = pd.DataFrame(metadata_rows)
valid_embedding_mask = np.isfinite(embeddings).all(axis=1)
metadata_df.loc[~valid_embedding_mask, "status"] = "embedding_failure"

coverage = float(valid_embedding_mask.mean())
print(f"Pair-side embedding coverage: {coverage:.2%}")
print(f"Extraction time: {extraction_seconds:.2f} seconds")

# A standard aligned benchmark must be complete. Unlike R3, no partial-pair
# operational score is created from failed inputs.
if embeddings.shape != (EXPECTED_OCCURRENCES, EMBEDDING_DIMENSION):
    raise RuntimeError("The complete aligned embedding matrix shape check failed.")
if coverage != 1.0:
    failures = metadata_df.loc[~valid_embedding_mask].head(20)
    display(failures)
    raise RuntimeError(
        f"Aligned pair-side coverage is {coverage:.2%}; expected 100%. Metrics are blocked."
    )
if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
    raise RuntimeError("At least one embedding is not L2-normalized.")

np.save(embeddings_path, embeddings)
metadata_df.to_csv(metadata_path, index=False)

# Deduplicate repeated images by encoded-byte hash. This is useful for storage
# and auditing, but hashes are not identity labels.
first_index_by_hash = {}
unique_indices = []
occurrence_to_unique = []
for bin_index, encoded_hash in enumerate(metadata_df["encoded_sha256"]):
    if encoded_hash not in first_index_by_hash:
        first_index_by_hash[encoded_hash] = len(unique_indices)
        unique_indices.append(bin_index)
    occurrence_to_unique.append(first_index_by_hash[encoded_hash])

unique_embeddings = embeddings[np.asarray(unique_indices)]
unique_metadata_df = metadata_df.iloc[unique_indices].copy().reset_index(drop=True)
unique_metadata_df.insert(0, "unique_image_id", np.arange(len(unique_metadata_df)))
metadata_df["unique_image_id"] = occurrence_to_unique

np.save(CACHE_DIR / "unique_aligned_embeddings.npy", unique_embeddings)
unique_metadata_df.to_csv(CACHE_DIR / "unique_aligned_metadata.csv", index=False)
metadata_df.to_csv(metadata_path, index=False)

extraction_summary = {
    "pair_side_images": EXPECTED_OCCURRENCES,
    "valid_embeddings": int(valid_embedding_mask.sum()),
    "coverage": coverage,
    "unique_encoded_images": int(len(unique_indices)),
    "embedding_dimension": EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "seconds": extraction_seconds,
    "mean_ms_per_pair_side": 1000 * extraction_seconds / EXPECTED_OCCURRENCES,
}
with open(OUTPUT_DIR / "embedding_extraction_summary.json", "w", encoding="utf-8") as file:
    json.dump(extraction_summary, file, indent=2)
print(json.dumps(extraction_summary, indent=2))


# Cell 19
embeddings_a = embeddings[0::2]
embeddings_b = embeddings[1::2]
scores = np.sum(embeddings_a * embeddings_b, axis=1).astype(np.float64)

pair_scores_df = pd.DataFrame({
    "pair_id": np.arange(1, EXPECTED_PAIRS + 1),
    "fold": np.arange(EXPECTED_PAIRS) // EXPECTED_PAIRS_PER_FOLD + 1,
    "label": labels,
    "pair_type": np.where(labels == 1, "genuine", "impostor"),
    "cosine_similarity": scores,
})
if len(pair_scores_df) != EXPECTED_PAIRS or not np.isfinite(scores).all():
    raise RuntimeError("Complete finite scores were not produced for all 7,000 pairs.")

pair_scores_df.to_csv(OUTPUT_DIR / "pair_scores.csv", index=False)
coverage_by_fold_df = (
    pair_scores_df.groupby(["fold", "pair_type"], as_index=False)
    .size().rename(columns={"size": "evaluable_pairs"})
)
coverage_by_fold_df.to_csv(OUTPUT_DIR / "pair_coverage_by_fold_and_class.csv", index=False)
display(coverage_by_fold_df)
print("All 7,000 official aligned pairs are evaluable.")


# Cell 21
def select_accuracy_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.nextafter(candidates.max(), np.inf))
    predictions = scores[:, None] >= candidates[None, :]
    accuracies = (predictions == labels[:, None]).mean(axis=0)
    maximum_accuracy = float(accuracies.max())
    best = candidates[np.isclose(accuracies, maximum_accuracy, rtol=0, atol=1e-12)]
    return float(best.max()), maximum_accuracy


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


print("Evaluation functions defined.")


# Cell 23
accuracy_fold_rows = []
oof_rows = []
roc_fold_rows = []
low_far_fold_rows = []

for test_fold in range(1, EXPECTED_FOLDS + 1):
    train_df = pair_scores_df[pair_scores_df["fold"] != test_fold]
    test_df = pair_scores_df[pair_scores_df["fold"] == test_fold]

    threshold, training_accuracy = select_accuracy_threshold(
        train_df["cosine_similarity"], train_df["label"]
    )
    accuracy_metrics = evaluate_at_threshold(
        test_df["cosine_similarity"], test_df["label"], threshold
    )
    predictions = accuracy_metrics.pop("predictions")
    accuracy_fold_rows.append({
        "evaluation": "aligned_benchmark", "fold": test_fold,
        "threshold": threshold, "training_accuracy": training_accuracy, **accuracy_metrics,
    })
    for pair_id, label, score, prediction in zip(
        test_df["pair_id"], test_df["label"], test_df["cosine_similarity"], predictions
    ):
        oof_rows.append({
            "pair_id": int(pair_id), "fold": test_fold, "label": int(label),
            "score": float(score), "threshold": threshold, "prediction": int(prediction),
        })

    roc_fold_rows.append({
        "evaluation": "aligned_benchmark", "fold": test_fold,
        **calculate_auc_eer(test_df["label"], test_df["cosine_similarity"]),
    })

    calibration = select_threshold_at_far(
        train_df["cosine_similarity"], train_df["label"], TARGET_FAR
    )
    low_far_metrics = evaluate_at_threshold(
        test_df["cosine_similarity"], test_df["label"], calibration["threshold"]
    )
    low_far_metrics.pop("predictions")
    low_far_fold_rows.append({
        "evaluation": "aligned_benchmark", "fold": test_fold,
        "target_training_far": TARGET_FAR, **calibration, **low_far_metrics,
    })

accuracy_fold_df = pd.DataFrame(accuracy_fold_rows)
accuracy_oof_df = pd.DataFrame(oof_rows).sort_values("pair_id")
roc_fold_df = pd.DataFrame(roc_fold_rows)
low_far_fold_df = pd.DataFrame(low_far_fold_rows)

def pooled_confusion_summary(frame):
    tp, tn, fp, fn = [int(frame[column].sum()) for column in ["tp", "tn", "fp", "fn"]]
    genuine, impostor = tp + fn, tn + fp
    total = genuine + impostor
    return tp, tn, fp, fn, genuine, impostor, total

tp, tn, fp, fn, genuine, impostor, total = pooled_confusion_summary(accuracy_fold_df)
accuracy_summary_df = pd.DataFrame([{
    "evaluation": "aligned_benchmark", "evaluated_pairs": total,
    "mean_threshold": accuracy_fold_df["threshold"].mean(),
    "threshold_std": accuracy_fold_df["threshold"].std(ddof=1),
    "mean_fold_accuracy": accuracy_fold_df["accuracy"].mean(),
    "fold_accuracy_std": accuracy_fold_df["accuracy"].std(ddof=1),
    "pooled_accuracy": (tp + tn) / total, "pooled_tar": tp / genuine,
    "pooled_far": fp / impostor, "pooled_frr": fn / genuine,
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
}])

pooled_roc = calculate_auc_eer(pair_scores_df["label"], pair_scores_df["cosine_similarity"])
roc_summary_df = pd.DataFrame([{
    "evaluation": "aligned_benchmark", **pooled_roc,
    "mean_fold_auc": roc_fold_df["auc"].mean(),
    "fold_auc_std": roc_fold_df["auc"].std(ddof=1),
    "mean_fold_eer": roc_fold_df["eer"].mean(),
    "fold_eer_std": roc_fold_df["eer"].std(ddof=1),
}])

tp, tn, fp, fn, genuine, impostor, total = pooled_confusion_summary(low_far_fold_df)
low_far_summary_df = pd.DataFrame([{
    "evaluation": "aligned_benchmark", "target_training_far": TARGET_FAR,
    "evaluated_pairs": total, "genuine_pairs": genuine, "impostor_pairs": impostor,
    "mean_threshold": low_far_fold_df["threshold"].mean(),
    "threshold_std": low_far_fold_df["threshold"].std(ddof=1),
    "pooled_tar": tp / genuine, "pooled_observed_far": fp / impostor,
    "pooled_frr": fn / genuine,
    "mean_fold_tar": low_far_fold_df["tar"].mean(),
    "fold_tar_std": low_far_fold_df["tar"].std(ddof=1),
    "test_far_granularity_approx": 1 / EXPECTED_IMPOSTOR_PER_FOLD,
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
}])

accuracy_fold_df.to_csv(OUTPUT_DIR / "accuracy_fold_results.csv", index=False)
accuracy_oof_df.to_csv(OUTPUT_DIR / "accuracy_out_of_fold_decisions.csv", index=False)
accuracy_summary_df.to_csv(OUTPUT_DIR / "accuracy_summary.csv", index=False)
roc_fold_df.to_csv(OUTPUT_DIR / "roc_fold_metrics.csv", index=False)
roc_summary_df.to_csv(OUTPUT_DIR / "roc_summary.csv", index=False)
low_far_fold_df.to_csv(OUTPUT_DIR / "tar_at_far_1e-3_fold_results.csv", index=False)
low_far_summary_df.to_csv(OUTPUT_DIR / "tar_at_far_1e-3_summary.csv", index=False)

print("--- ACCURACY ---")
display(accuracy_summary_df)
print("--- ROC AUC AND EER ---")
display(roc_summary_df)
print("--- TAR AT TRAINING FAR 1e-3 ---")
display(low_far_summary_df)


# Cell 25
plt.style.use("seaborn-v0_8-whitegrid")
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].hist(impostor_scores, bins=50, alpha=0.70, density=True, label="impostor")
axes[0].hist(genuine_scores, bins=50, alpha=0.70, density=True, label="genuine")
axes[0].set_title("Aligned CFP-FP cosine scores")
axes[0].set_xlabel("cosine similarity")
axes[0].set_ylabel("density")
axes[0].legend()

fpr, tpr, _ = roc_curve(labels, scores)
axes[1].plot(fpr, tpr, label=f"AUC = {roc_summary_df.iloc[0]['auc']:.4f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_title("ROC curve")
axes[1].set_xlabel("false acceptance rate")
axes[1].set_ylabel("true acceptance rate")
axes[1].legend()

axes[2].bar(accuracy_fold_df["fold"], 100 * accuracy_fold_df["accuracy"])
axes[2].set_title("Held-out accuracy by fold")
axes[2].set_xlabel("fold")
axes[2].set_ylabel("accuracy (%)")
axes[2].set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cfp_fp_r4_diagnostics.png", dpi=180, bbox_inches="tight")
plt.show()


# Cell 27
RUN_COMPLETED_UTC = datetime.now(timezone.utc).isoformat()

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "pipeline_revision": PIPELINE_REVISION,
    "title": "Frozen MobileFaceNet aligned CFP-FP verification baseline",
    "researcher": RESEARCHER_NAME,
    "institution": INSTITUTION,
    "run_started_utc": RUN_STARTED_UTC,
    "run_completed_utc": RUN_COMPLETED_UTC,
    "purpose": RUN_PURPOSE,
    "hypothesis": HYPOTHESIS,
    "dataset": bin_audit,
    "recognition_pipeline": model_audit,
    "runtime": runtime_audit,
    "embedding_extraction": extraction_summary,
    "prior_raw_transfer_diagnostic": PRIOR_RAW_TRANSFER,
    "evaluation": {
        "boundary": "recognition-only aligned CFP-FP benchmark",
        "threshold_training": "nine folds only; held-out fold used only for testing",
        "target_training_far": TARGET_FAR,
        "failure_rule": FAILURE_RULE,
        "test_fold_far_granularity": 1 / EXPECTED_IMPOSTOR_PER_FOLD,
    },
    "results": {
        "accuracy": json.loads(accuracy_summary_df.to_json(orient="records")),
        "roc_eer": json.loads(roc_summary_df.to_json(orient="records")),
        "tar_at_far_1e-3": json.loads(low_far_summary_df.to_json(orient="records")),
    },
    "limitations": [
        "R4 does not measure raw-image face detection or end-to-end operational acceptance.",
        "The aligned-bin package does not provide verified identity labels or original filenames.",
        "The R4 cache alone is therefore not sufficient for identity-disjoint E05-ID or identity-grouped E07.",
        "The FAR target is trained on nine folds; each held-out fold has only 350 impostor pairs.",
        "No flip augmentation is used, matching the frozen no-augmentation decision for this project.",
    ],
}
with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)

accuracy = accuracy_summary_df.iloc[0]
roc = roc_summary_df.iloc[0]
low_far = low_far_summary_df.iloc[0]

def pct(value):
    return f"{100 * float(value):.3f}%"

brief = f"""# CFP-FP-E01-R4 — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

How well does the frozen 512-D MobileFaceNet verify frontal-to-profile pairs when supplied with the standard aligned 112×112 CFP-FP crops?

## Why R4 was necessary

- Raw R1 coverage: {pct(PRIOR_RAW_TRANSFER['R1_image_coverage'])}
- R2 recovery coverage: {pct(PRIOR_RAW_TRANSFER['R2_image_coverage'])}
- R3 multi-scale/tiled coverage: {pct(PRIOR_RAW_TRANSFER['R3_image_coverage'])}
- R3 remained below the frozen 90% validity gate, so no R3 recognition metrics were allowed.

## Controlled design

- Standard aligned package checksum: `{bin_audit['sha256']}`
- 7,000 pairs, 10 contiguous folds, 350 genuine and 350 impostor pairs per fold.
- Frozen `w600k_mbf` MobileFaceNet, 512-D L2-normalized embedding, cosine similarity.
- No flip augmentation.
- Every threshold learned from nine folds and tested on the held-out fold.
- Complete 14,000/14,000 pair-side embedding coverage required.

## Main results

| Metric | Aligned benchmark |
|---|---:|
| Evaluated pairs | {int(low_far['evaluated_pairs']):,} |
| Mean fold accuracy | {pct(accuracy['mean_fold_accuracy'])} |
| Pooled ROC AUC | {pct(roc['auc'])} |
| Pooled EER | {pct(roc['eer'])} |
| TAR at training FAR ≤ 0.1% | {pct(low_far['pooled_tar'])} |
| Observed pooled FAR | {pct(low_far['pooled_observed_far'])} |
| FRR | {pct(low_far['pooled_frr'])} |

## Interpretation boundary

R4 measures **recognition under pose variation after standard alignment**. R3 separately measures **raw preprocessing transfer**. These numbers must not be merged or labelled as the same pipeline.

The cache can support pair-based protected-template utility and unlinkability experiments. Before E05-ID or identity-grouped E07, a separately verified identity-to-aligned-image mapping is required; the notebook does not infer identities from pair labels.
"""
(OUTPUT_DIR / "CFP-FP-E01-R4_one_page_brief.md").write_text(brief, encoding="utf-8")
(OUTPUT_DIR / "RESULTS_README.md").write_text(brief + f"\n\n## Run notes\n\n{RUN_NOTES}\n", encoding="utf-8")

tracker_row_df = pd.DataFrame([{
    "Experiment": "CFP-FP-E01-R4",
    "Main question": "Frozen MobileFaceNet recognition under frontal-to-profile pose after standard alignment",
    "Important metric": "TAR@FAR 0.1%; EER; AUC; mean fold accuracy",
    "Actual result": (
        f"TAR {pct(low_far['pooled_tar'])}; EER {pct(roc['eer'])}; "
        f"AUC {pct(roc['auc'])}; accuracy {pct(accuracy['mean_fold_accuracy'])}"
    ),
    "Simple interpretation": "Valid aligned recognition baseline; raw detector transfer remains a separate failure result.",
}])
tracker_row_df.to_csv(OUTPUT_DIR / "result_tracker_row.csv", index=False)

print(brief)
display(tracker_row_df)


# Cell 29
def file_manifest(root):
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)

file_manifest(OUTPUT_DIR).to_csv(OUTPUT_DIR / "result_file_manifest.csv", index=False)
file_manifest(CACHE_DIR).to_csv(CACHE_DIR / "cache_file_manifest.csv", index=False)

results_archive = Path(shutil.make_archive(
    "/kaggle/working/CFP-FP-E01-R4-aligned-baseline-results",
    "zip", root_dir=OUTPUT_DIR,
))
cache_archive = Path(shutil.make_archive(
    "/kaggle/working/CFP-FP-E01-R4-aligned-embedding-cache",
    "zip", root_dir=CACHE_DIR,
))

os.chdir("/kaggle/working")
print("Results archive:", results_archive, f"({results_archive.stat().st_size / 1024**2:.2f} MiB)")
display(FileLink(results_archive.name))
print("Aligned cache archive:", cache_archive, f"({cache_archive.stat().st_size / 1024**2:.2f} MiB)")
display(FileLink(cache_archive.name))
print("If a link does not open, download the ZIP from Kaggle's Output/Files panel.")
