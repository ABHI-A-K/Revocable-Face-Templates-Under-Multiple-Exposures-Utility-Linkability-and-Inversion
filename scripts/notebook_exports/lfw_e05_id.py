# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 3
import subprocess
import sys
import json

print("Python executable:", sys.executable)

if "onnxruntime" in sys.modules:
    raise RuntimeError(
        "ONNX Runtime is already loaded in this kernel. Restart the Kaggle session, "
        "then choose Run All so the repaired GPU installation is imported cleanly."
    )

# Install InsightFace first. It may temporarily install the CPU ONNX Runtime dependency.
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "insightface==1.0.1",
    ]
)

# Remove both distributions after InsightFace dependency resolution. This clears
# overlapping onnxruntime Python files that otherwise make provider selection random.
subprocess.run(
    [
        sys.executable, "-m", "pip", "uninstall", "-y",
        "onnxruntime", "onnxruntime-gpu",
    ],
    check=False,
    stdout=subprocess.DEVNULL,
)

subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "--force-reinstall", "--no-deps", "onnxruntime-gpu==1.26.0",
    ]
)

# Probe in a separate process: it cannot reuse a stale module from this notebook kernel.
probe_code = (
    "import json, onnxruntime as ort; "
    "print(json.dumps({'version': ort.__version__, "
    "'device': ort.get_device(), 'providers': ort.get_available_providers()}))"
)
probe = subprocess.run(
    [sys.executable, "-c", probe_code],
    capture_output=True,
    text=True,
)

if probe.returncode != 0:
    raise RuntimeError(
        "The repaired ONNX Runtime installation could not be imported in a clean "
        f"process. Diagnostic output:\n{probe.stderr.strip()}"
    )

provider_probe = json.loads(probe.stdout.strip().splitlines()[-1])
print("Clean-process ONNX Runtime audit:", json.dumps(provider_probe, indent=2))

if "CUDAExecutionProvider" not in provider_probe["providers"]:
    raise RuntimeError(
        "onnxruntime-gpu installed, but CUDAExecutionProvider is still absent. "
        "Confirm Kaggle Settings -> Accelerator -> GPU, restart the session, and "
        "run this notebook from Step 1. Providers found: "
        f"{provider_probe['providers']}"
    )

print("Installation complete: CUDAExecutionProvider verified.")


# Cell 5
from pathlib import Path

print("--- SEARCHING FOR E01 CACHE ---")

found = False

for embeddings_file in Path("/kaggle/input").rglob("raw_embeddings.npy"):
    metadata_file = embeddings_file.with_name("embedding_metadata.csv")

    print("\nEmbedding file:", embeddings_file)
    print("Metadata exists:", metadata_file.is_file())
    print("Cache directory:", embeddings_file.parent)

    if metadata_file.is_file():
        found = True

if not found:
    print("\nComplete E01 cache not found.")


# Cell 6
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ID = "LFW-E05-ID-identity-disjoint-multi-exposure-RO"
EXPERIMENT_TITLE = "Identity-Disjoint Multi-Exposure Attacker Using RO"

DATASET_ROOT = Path("/kaggle/input/datasets/jessicali9530/lfw-dataset")
IMAGE_ROOT = DATASET_ROOT / "lfw-deepfunneled" / "lfw-deepfunneled"
PAIRS_PATH = DATASET_ROOT / "pairs.csv"

CACHE_DIR = Path("/kaggle/working/experiments/LFW-E01-raw-baseline")
OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Change only this path if your attached frozen cache is stored elsewhere.
PRECOMPUTED_CACHE_DIR = Path(
    "/kaggle/input/datasets/abhiak0709/cache-embeddings-from-lfw-e03"
)

EMBEDDING_DIMENSION = 512
TEMPLATE_BITS = 128

# Frozen from E02-E05.
ORIGINAL_RO_SEEDS = [
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
]
E05_EXTENSION_RO_SEEDS = list(range(2026081106, 2026081121))
RO_SEEDS = ORIGINAL_RO_SEEDS + E05_EXTENSION_RO_SEEDS

EXPOSURE_COUNTS = [1, 2, 5, 10]
MAX_EXPOSURES = max(EXPOSURE_COUNTS)
KEY_TRIAL_RANDOM_SEED = 2026082105
ATTACK_MODEL_RANDOM_SEED = 2026082106
BOOTSTRAP_RANDOM_SEED = 2026082107
BOOTSTRAP_REPETITIONS = 5000

# New E05-ID controls, fixed before scoring.
IDENTITY_SPLIT_RANDOM_SEED = 2026082501
PAIR_SAMPLING_RANDOM_SEED = 2026082502
IDENTITY_FOLDS = 10
PAIRS_PER_CLASS_PER_FOLD = 300
MAX_GENUINE_PAIRS_PER_IDENTITY = 10

DETECTION_THRESHOLD = 0.50
DETECTION_SIZE = (640, 640)
CHECKPOINT_INTERVAL = 500
RUN_PILOT_IF_EXTRACTION_IS_NEEDED = True
PILOT_SIZE = 100
PILOT_RANDOM_SEED = 2026
STRICT_FROZEN_COUNTS = True

EXPECTED_SOURCE_TOTAL_PAIRS = 6000
EXPECTED_SOURCE_GENUINE_PAIRS = 3000
EXPECTED_SOURCE_IMPOSTOR_PAIRS = 3000
EXPECTED_UNIQUE_IMAGES = 7701
EXPECTED_VALID_EMBEDDINGS = 7682
EXPECTED_TOTAL_PAIRS = IDENTITY_FOLDS * PAIRS_PER_CLASS_PER_FOLD * 2
EXPECTED_GENUINE_PAIRS = IDENTITY_FOLDS * PAIRS_PER_CLASS_PER_FOLD
EXPECTED_IMPOSTOR_PAIRS = IDENTITY_FOLDS * PAIRS_PER_CLASS_PER_FOLD
EXPECTED_EVALUABLE_PAIRS = EXPECTED_TOTAL_PAIRS

# Frozen E05 exploratory security guardrails; retained for direct comparability.
MIN_WORST_SAME_KEY_AUC = 0.98
MAX_MEAN_LEARNED_ATTACKER_AUC = 0.55
MAX_WORST_LEARNED_ATTACKER_AUC = 0.60
MAX_MEAN_K10_DSYS = 0.10
MAX_MEAN_AUC_GROWTH_K10_MINUS_K1 = 0.02
MAX_MEAN_K10_BALANCED_ACCURACY = 0.55

if len(RO_SEEDS) != 20 or len(set(RO_SEEDS)) != 20:
    raise RuntimeError("E05-ID requires exactly the 20 unique E05 RO seeds.")
if MAX_EXPOSURES >= len(RO_SEEDS):
    raise RuntimeError("At least one key must remain unseen after selecting exposures.")

print("Experiment:", EXPERIMENT_ID)
print("Output directory:", OUTPUT_DIR)
print("Dataset root exists:", DATASET_ROOT.is_dir())
print("Pairs file exists:", PAIRS_PATH.is_file())
print("RO keys:", len(RO_SEEDS))
print("Exposure counts:", EXPOSURE_COUNTS)
print("Identity folds:", IDENTITY_FOLDS)
print("Pairs per fold:", PAIRS_PER_CLASS_PER_FOLD * 2)


# Cell 8
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
RUN_PURPOSE = (
    "Test whether E05 multi-exposure linkability generalises to identities that are "
    "completely absent from attacker training."
)
HYPOTHESIS = (
    "If the E05 signal reflects identity-independent structure in RO-protected "
    "templates, the attacker will remain above chance on held-out identities; "
    "additional exposures should still show little K=1 to K=10 growth."
)
RUN_NOTES = """
- The official LFW pairs are used only to reconstruct the frozen 7,701-image cache order.
- Attack pairs are rebuilt after embedding validation using ten identity-disjoint folds.
- Every identity, image, and derived pair belongs to exactly one fold.
- Each fold contains 300 genuine and 300 impostor pairs made only from valid embeddings.
- Genuine sampling is spread across identities and capped at 10 pairs per identity.
- Frozen protection: seeded Gaussian RO, reduced QR, 128 bits, threshold at zero.
- The exact E05 20-key bank and balanced query-key schedule are preserved.
- Exposure sets remain nested at K=1,2,5,10; both A-to-B and B-to-A are tested.
- The same logistic attacker is trained on nine identity folds and tested on the tenth.
- The attack does not use raw embeddings, face images, RO matrices, or secret seeds.
- Reconstruction, inversion, and impersonation remain outside E05-ID.
""".strip()

RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()

initial_log = f"""# {EXPERIMENT_ID} research log

**Researcher:** {RESEARCHER_NAME}  
**Institution:** {INSTITUTION}  
**Run started (UTC):** {RUN_STARTED_UTC}

## Purpose

{RUN_PURPOSE}

## Hypothesis

{HYPOTHESIS}

## Identity-disjoint rule

All images of one identity are assigned to exactly one of ten folds. The attacker trains
on nine folds and is tested on the remaining fold, so train/test identity overlap must
equal zero in every cross-validation round.

## Frozen exploratory security guardrails

- Worst same-key AUC across the 20-key validity check >= {MIN_WORST_SAME_KEY_AUC:.2f}
- Mean learned-attacker AUC at every K <= {MAX_MEAN_LEARNED_ATTACKER_AUC:.2f}
- Worst learned-attacker AUC over any trial/direction/K <= {MAX_WORST_LEARNED_ATTACKER_AUC:.2f}
- Mean learned-attacker D_sys at K=10 <= {MAX_MEAN_K10_DSYS:.2f}
- Mean learned-attacker AUC growth from K=1 to K=10 <= {MAX_MEAN_AUC_GROWTH_K10_MINUS_K1:.2f}
- Mean learned-attacker balanced accuracy at K=10 <= {MAX_MEAN_K10_BALANCED_ACCURACY:.2f}

## Notes

{RUN_NOTES}
"""

(OUTPUT_DIR / "research_log.md").write_text(initial_log, encoding="utf-8")
print(initial_log)


# Cell 10
import insightface
import onnxruntime as ort

PACKAGE_NAMES = [
    "numpy", "pandas", "opencv-python", "Pillow", "scikit-learn",
    "matplotlib", "torch", "onnx", "onnxruntime", "onnxruntime-gpu", "insightface",
]

package_versions = {}
for package_name in PACKAGE_NAMES:
    try:
        package_versions[package_name] = version(package_name)
    except PackageNotFoundError:
        package_versions[package_name] = "NOT INSTALLED"

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is not enabled. In Kaggle: Settings -> Accelerator -> GPU, "
        "restart the session, and Run All again."
    )

runtime_audit = {
    "python": sys.version.split()[0],
    "operating_system": platform.platform(),
    "packages": package_versions,
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "torch_cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    "onnxruntime_device": ort.get_device(),
    "onnxruntime_providers": ort.get_available_providers(),
}

print(json.dumps(runtime_audit, indent=2))

if package_versions["onnxruntime"] != "NOT INSTALLED":
    raise RuntimeError(
        "Conflicting CPU onnxruntime distribution is installed. Restart the Kaggle "
        "session and run Step 1 from the updated notebook."
    )

if "CUDAExecutionProvider" not in ort.get_available_providers():
    print(
        "NOTICE: ONNX CUDA provider is unavailable. This is harmless only when "
        "the complete frozen embedding cache is found; the cache step will enforce this."
    )


# Cell 12
if not PAIRS_PATH.is_file():
    raise FileNotFoundError(
        "pairs.csv was not found. Add jessicali9530/lfw-dataset to the notebook."
    )

source_pairs_df = pd.read_csv(PAIRS_PATH)


def make_image_path(person, image_number):
    image_number = int(float(image_number))
    return IMAGE_ROOT / str(person) / f"{person}_{image_number:04d}.jpg"


source_rows = []
for index, row in source_pairs_df.iterrows():
    is_genuine = pd.isna(row["Unnamed: 3"])
    if is_genuine:
        person_a = str(row["name"])
        person_b = person_a
        image_a = make_image_path(person_a, row["imagenum1"])
        image_b = make_image_path(person_b, row["imagenum2"])
        label = 1
    else:
        person_a = str(row["name"])
        person_b = str(row["imagenum2"])
        image_a = make_image_path(person_a, row["imagenum1"])
        image_b = make_image_path(person_b, row["Unnamed: 3"])
        label = 0

    source_rows.append(
        {
            "source_pair_id": index + 1,
            "source_fold": (index // 600) + 1,
            "label": label,
            "person_a": person_a,
            "person_b": person_b,
            "image_a": str(image_a),
            "image_b": str(image_b),
        }
    )

source_pair_manifest_df = pd.DataFrame(source_rows)
unique_image_paths = sorted(
    set(source_pair_manifest_df["image_a"]) | set(source_pair_manifest_df["image_b"])
)
missing_images = [path for path in unique_image_paths if not Path(path).is_file()]

source_inventory_checks = {
    "official_pairs_used_for_cache_index_only": len(source_pair_manifest_df),
    "official_genuine_pairs": int((source_pair_manifest_df["label"] == 1).sum()),
    "official_impostor_pairs": int((source_pair_manifest_df["label"] == 0).sum()),
    "unique_cache_images": len(unique_image_paths),
    "missing_images": len(missing_images),
}

expected_source_inventory = {
    "official_pairs_used_for_cache_index_only": EXPECTED_SOURCE_TOTAL_PAIRS,
    "official_genuine_pairs": EXPECTED_SOURCE_GENUINE_PAIRS,
    "official_impostor_pairs": EXPECTED_SOURCE_IMPOSTOR_PAIRS,
    "unique_cache_images": EXPECTED_UNIQUE_IMAGES,
    "missing_images": 0,
}

if source_inventory_checks != expected_source_inventory:
    raise RuntimeError(
        f"Frozen cache inventory validation failed. Obtained {source_inventory_checks}; "
        f"expected {expected_source_inventory}."
    )

source_pair_manifest_df.to_csv(
    OUTPUT_DIR / "official_pairs_cache_index_only.csv", index=False
)
print(json.dumps(source_inventory_checks, indent=2))
print("Frozen cache image order rebuilt. Official pairs will not be attack samples.")


# Cell 14
local_embeddings_path = CACHE_DIR / "raw_embeddings.npy"
local_metadata_path = CACHE_DIR / "embedding_metadata.csv"

if PRECOMPUTED_CACHE_DIR is not None:
    selected_cache_dir = Path(PRECOMPUTED_CACHE_DIR)
    embeddings_path = selected_cache_dir / "raw_embeddings.npy"
    metadata_path = selected_cache_dir / "embedding_metadata.csv"
    cache_is_writable = False
else:
    selected_cache_dir = CACHE_DIR
    embeddings_path = local_embeddings_path
    metadata_path = local_metadata_path
    cache_is_writable = True

if embeddings_path.exists() != metadata_path.exists():
    raise RuntimeError(
        f"Incomplete cache in {selected_cache_dir}: both raw_embeddings.npy and "
        "embedding_metadata.csv must exist together."
    )

if embeddings_path.exists():
    print("Loading frozen cache:", selected_cache_dir)
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)
    if embeddings.shape != (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected cached embedding shape: {embeddings.shape}")
    if metadata_df["image_path"].astype(str).tolist() != unique_image_paths:
        raise RuntimeError("Cached image order does not match the frozen manifest.")
else:
    if not cache_is_writable:
        raise FileNotFoundError(
            f"The attached cache folder is missing the required files: {selected_cache_dir}"
        )
    print("No cache found; initialising resumable frozen extraction.")
    embeddings = np.full(
        (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION), np.nan, dtype=np.float32
    )
    metadata_df = pd.DataFrame(
        {
            "image_id": np.arange(EXPECTED_UNIQUE_IMAGES),
            "image_path": unique_image_paths,
            "identity": [Path(path).parent.name for path in unique_image_paths],
            "status": "pending",
            "faces_detected": 0,
            "detection_score": np.nan,
            "bbox_x1": np.nan,
            "bbox_y1": np.nan,
            "bbox_x2": np.nan,
            "bbox_y2": np.nan,
            "embedding_norm": np.nan,
            "latency_ms": np.nan,
        }
    )

pending_indices = metadata_df.index[metadata_df["status"] == "pending"].tolist()
if pending_indices and not cache_is_writable:
    raise RuntimeError("The attached read-only cache contains pending rows.")

EXTRACTION_NEEDED = bool(pending_indices)
print("Cache directory:", selected_cache_dir)
print("Completed rows:", EXPECTED_UNIQUE_IMAGES - len(pending_indices))
print("Pending rows:", len(pending_indices))
print("GPU extraction needed:", EXTRACTION_NEEDED)


# Cell 16
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


expected_model_hashes = {
    "det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    "w600k_mbf.onnx": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
}

app = None
observed_model_hashes = {}
providers_used = []

if EXTRACTION_NEEDED:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "The embedding cache is incomplete and ONNX CUDA is unavailable. "
            "Rerun Step 1 in a fresh GPU session or attach the frozen E01 cache."
        )

    from insightface.app import FaceAnalysis

    MODEL_ROOT = Path("/kaggle/working/insightface_models")
    MODEL_DIR = MODEL_ROOT / "models" / "buffalo_sc"
    providers_used = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODEL_ROOT),
        allowed_modules=["detection", "recognition"],
        providers=providers_used,
    )
    app.prepare(
        ctx_id=0,
        det_thresh=DETECTION_THRESHOLD,
        det_size=DETECTION_SIZE,
    )

    for filename, expected_hash in expected_model_hashes.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing frozen model file: {model_path}")
        observed_hash = sha256_file(model_path)
        observed_model_hashes[filename] = observed_hash
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Model hash mismatch for {filename}: {observed_hash} != {expected_hash}"
            )
    print("Frozen models loaded and hash-verified on CUDA.")
else:
    print("Complete frozen cache found; model loading correctly skipped.")

model_audit = {
    "model_pack": "buffalo_sc",
    "detector": "SCRFD-500MF",
    "recognizer": "MobileFaceNet w600k_mbf",
    "detection_threshold": DETECTION_THRESHOLD,
    "detection_size": list(DETECTION_SIZE),
    "expected_model_sha256": expected_model_hashes,
    "observed_model_sha256_this_run": observed_model_hashes,
    "providers_used_for_new_extraction": providers_used,
    "inference_skipped_due_to_complete_cache": not EXTRACTION_NEEDED,
}

print(json.dumps(model_audit, indent=2))


# Cell 18
def extract_largest_face(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, 0, None
    faces = app.get(image)
    if not faces:
        return None, 0, None
    selected_face = max(
        faces,
        key=lambda face: (
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )
    return selected_face.normed_embedding.astype(np.float32), len(faces), selected_face


if EXTRACTION_NEEDED:
    for pair_kind, label in [("genuine", 1), ("impostor", 0)]:
        pair = pair_manifest_df[pair_manifest_df["label"] == label].iloc[0]
        first, count_a, _ = extract_largest_face(pair["image_a"])
        second, count_b, _ = extract_largest_face(pair["image_b"])
        if first is None or second is None:
            raise RuntimeError(f"Smoke test failed for the {pair_kind} pair.")
        print(
            f"{pair_kind.title()} smoke pair | faces=({count_a},{count_b}) | "
            f"cosine={float(np.dot(first, second)):.6f}"
        )

    if RUN_PILOT_IF_EXTRACTION_IS_NEEDED:
        rng = np.random.default_rng(PILOT_RANDOM_SEED)
        pilot_paths = rng.choice(
            np.asarray(unique_image_paths, dtype=object),
            size=PILOT_SIZE,
            replace=False,
        ).tolist()

        warmup_image = cv2.imread(pilot_paths[0])
        if warmup_image is None:
            raise RuntimeError("Could not read the pilot warm-up image.")
        _ = app.get(warmup_image)

        pilot_rows = []
        for image_path in pilot_paths:
            started = time.perf_counter()
            embedding, face_count, face = extract_largest_face(image_path)
            latency_ms = (time.perf_counter() - started) * 1000
            pilot_rows.append(
                {
                    "image_path": str(image_path),
                    "status": "valid" if embedding is not None else "failure",
                    "faces_detected": face_count,
                    "detection_score": np.nan if face is None else float(face.det_score),
                    "embedding_norm": np.nan if embedding is None else float(np.linalg.norm(embedding)),
                    "latency_ms": latency_ms,
                }
            )
        pilot_df = pd.DataFrame(pilot_rows)
        pilot_df.to_csv(OUTPUT_DIR / "pilot_runtime.csv", index=False)
        print("Pilot valid:", int((pilot_df["status"] == "valid").sum()), "/", len(pilot_df))
        print("Pilot p50/p95 ms:", pilot_df["latency_ms"].median(), pilot_df["latency_ms"].quantile(0.95))
else:
    print("Smoke test and pilot skipped because the frozen cache is complete.")


# Cell 20
def save_embedding_checkpoint():
    if not cache_is_writable:
        return
    temporary_embeddings = CACHE_DIR / "raw_embeddings.tmp"
    temporary_metadata = CACHE_DIR / "embedding_metadata.tmp"
    with open(temporary_embeddings, "wb") as file:
        np.save(file, embeddings)
    os.replace(temporary_embeddings, local_embeddings_path)
    metadata_df.to_csv(temporary_metadata, index=False)
    os.replace(temporary_metadata, local_metadata_path)


extraction_started = time.perf_counter()
for progress, row_index in enumerate(pending_indices, start=1):
    image_path = metadata_df.at[row_index, "image_path"]
    image_started = time.perf_counter()
    image = cv2.imread(image_path)

    if image is None:
        metadata_df.at[row_index, "status"] = "read_failure"
    else:
        faces = app.get(image)
        metadata_df.at[row_index, "faces_detected"] = len(faces)
        if not faces:
            metadata_df.at[row_index, "status"] = "detection_failure"
        else:
            selected_face = max(
                faces,
                key=lambda face: (
                    (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                ),
            )
            embedding = selected_face.normed_embedding.astype(np.float32)
            embedding_norm = float(np.linalg.norm(embedding))
            embedding_valid = (
                embedding.shape == (EMBEDDING_DIMENSION,)
                and np.isfinite(embedding).all()
                and np.isclose(embedding_norm, 1.0, atol=1e-4)
            )
            if embedding_valid:
                embeddings[row_index] = embedding
                status = "valid"
            else:
                status = "invalid_embedding"

            bbox = selected_face.bbox.astype(float)
            metadata_df.at[row_index, "status"] = status
            metadata_df.at[row_index, "detection_score"] = float(selected_face.det_score)
            metadata_df.loc[
                row_index, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
            ] = bbox
            metadata_df.at[row_index, "embedding_norm"] = embedding_norm

    metadata_df.at[row_index, "latency_ms"] = (
        time.perf_counter() - image_started
    ) * 1000

    if progress % CHECKPOINT_INTERVAL == 0 or progress == len(pending_indices):
        save_embedding_checkpoint()
        print(f"Checkpoint: {progress}/{len(pending_indices)} pending images processed")

save_embedding_checkpoint()

valid_embedding_mask = (
    metadata_df["status"].eq("valid").to_numpy()
    & np.isfinite(embeddings).all(axis=1)
)
status_counts = metadata_df["status"].value_counts().to_dict()
extraction_summary = {
    "total_unique_images": len(embeddings),
    "valid_embeddings": int(valid_embedding_mask.sum()),
    "read_failures": int(status_counts.get("read_failure", 0)),
    "detection_failures": int(status_counts.get("detection_failure", 0)),
    "invalid_embeddings": int(status_counts.get("invalid_embedding", 0)),
    "multiple_face_images": int((metadata_df["faces_detected"] > 1).sum()),
    "embedding_shape": list(embeddings.shape),
    "new_rows_processed": len(pending_indices),
    "minutes_this_cell": (time.perf_counter() - extraction_started) / 60,
}

print(json.dumps(extraction_summary, indent=2))

if embeddings.shape != (EXPECTED_UNIQUE_IMAGES, EMBEDDING_DIMENSION):
    raise RuntimeError("Frozen embedding shape check failed.")
if STRICT_FROZEN_COUNTS and int(valid_embedding_mask.sum()) != EXPECTED_VALID_EMBEDDINGS:
    raise RuntimeError(
        "Valid embedding count differs from frozen E01/E02. Review the cache/models before continuing."
    )


# Cell 22
failed_images_df = metadata_df[~valid_embedding_mask].copy()
failed_images_df.to_csv(OUTPUT_DIR / "failed_images_excluded_before_pairing.csv", index=False)

valid_rows_df = metadata_df.loc[
    valid_embedding_mask, ["image_id", "image_path", "identity"]
].copy()
valid_rows_df["image_id"] = valid_rows_df["image_id"].astype(int)
valid_rows_df["identity"] = valid_rows_df["identity"].astype(str)

identity_to_image_ids = {
    identity: sorted(group["image_id"].astype(int).tolist())
    for identity, group in valid_rows_df.groupby("identity", sort=True)
}
image_id_to_path = valid_rows_df.set_index("image_id")["image_path"].astype(str).to_dict()

split_rng = np.random.default_rng(IDENTITY_SPLIT_RANDOM_SEED)
identity_stat_rows = []
for identity, image_ids in identity_to_image_ids.items():
    image_count = len(image_ids)
    raw_capacity = image_count * (image_count - 1) // 2
    identity_stat_rows.append(
        {
            "identity": identity,
            "valid_images": image_count,
            "raw_genuine_pair_capacity": raw_capacity,
            "capped_genuine_pair_capacity": min(
                raw_capacity, MAX_GENUINE_PAIRS_PER_IDENTITY
            ),
            "split_tie_breaker": float(split_rng.random()),
        }
    )

identity_stats_df = pd.DataFrame(identity_stat_rows)
fold_state = {
    fold: {"genuine_capacity": 0, "valid_images": 0, "identity_count": 0}
    for fold in range(1, IDENTITY_FOLDS + 1)
}
identity_to_fold = {}

# Assign multi-image identities first, largest capped capacity first.
multi_identity_df = identity_stats_df[
    identity_stats_df["capped_genuine_pair_capacity"] > 0
].sort_values(
    ["capped_genuine_pair_capacity", "valid_images", "split_tie_breaker", "identity"],
    ascending=[False, False, True, True],
)

for row in multi_identity_df.itertuples(index=False):
    chosen_fold = min(
        fold_state,
        key=lambda fold: (
            fold_state[fold]["genuine_capacity"],
            fold_state[fold]["valid_images"],
            fold_state[fold]["identity_count"],
            fold,
        ),
    )
    identity_to_fold[row.identity] = chosen_fold
    fold_state[chosen_fold]["genuine_capacity"] += int(
        row.capped_genuine_pair_capacity
    )
    fold_state[chosen_fold]["valid_images"] += int(row.valid_images)
    fold_state[chosen_fold]["identity_count"] += 1

# Single-image identities add impostor diversity but no genuine-pair capacity.
single_identity_df = identity_stats_df[
    identity_stats_df["capped_genuine_pair_capacity"] == 0
].sort_values(["split_tie_breaker", "identity"], ascending=[True, True])

for row in single_identity_df.itertuples(index=False):
    chosen_fold = min(
        fold_state,
        key=lambda fold: (
            fold_state[fold]["identity_count"],
            fold_state[fold]["valid_images"],
            fold,
        ),
    )
    identity_to_fold[row.identity] = chosen_fold
    fold_state[chosen_fold]["valid_images"] += int(row.valid_images)
    fold_state[chosen_fold]["identity_count"] += 1

identity_fold_assignment_df = identity_stats_df.drop(
    columns=["split_tie_breaker"]
).copy()
identity_fold_assignment_df["fold"] = identity_fold_assignment_df["identity"].map(
    identity_to_fold
)
identity_fold_assignment_df = identity_fold_assignment_df.sort_values(
    ["fold", "identity"]
).reset_index(drop=True)

if identity_fold_assignment_df["fold"].isna().any():
    raise RuntimeError("At least one identity was not assigned to a fold.")
if identity_fold_assignment_df["identity"].duplicated().any():
    raise RuntimeError("An identity appears more than once in the fold assignment.")

pair_rng = np.random.default_rng(PAIR_SAMPLING_RANDOM_SEED)
attack_pair_rows = []
fold_protocol_rows = []

for fold in range(1, IDENTITY_FOLDS + 1):
    fold_identities = sorted(
        identity_fold_assignment_df.loc[
            identity_fold_assignment_df["fold"] == fold, "identity"
        ].tolist()
    )

    # Round-robin genuine selection gives many identities a first pair before any
    # identity contributes a second pair.
    genuine_queues = {}
    for identity in fold_identities:
        candidates = list(combinations(identity_to_image_ids[identity], 2))
        pair_rng.shuffle(candidates)
        if candidates:
            genuine_queues[identity] = candidates[:MAX_GENUINE_PAIRS_PER_IDENTITY]

    genuine_identity_order = sorted(genuine_queues)
    pair_rng.shuffle(genuine_identity_order)
    genuine_selected = []
    round_index = 0
    while len(genuine_selected) < PAIRS_PER_CLASS_PER_FOLD:
        added_this_round = 0
        for identity in genuine_identity_order:
            queue = genuine_queues[identity]
            if round_index < len(queue):
                image_a_id, image_b_id = queue[round_index]
                if bool(pair_rng.integers(0, 2)):
                    image_a_id, image_b_id = image_b_id, image_a_id
                genuine_selected.append((identity, image_a_id, image_b_id))
                added_this_round += 1
                if len(genuine_selected) == PAIRS_PER_CLASS_PER_FOLD:
                    break
        if added_this_round == 0:
            break
        round_index += 1

    if len(genuine_selected) < PAIRS_PER_CLASS_PER_FOLD:
        raise RuntimeError(
            f"Fold {fold} has only {len(genuine_selected)} capped genuine candidates; "
            f"{PAIRS_PER_CLASS_PER_FOLD} are required."
        )

    identity_pair_candidates = list(combinations(fold_identities, 2))
    pair_rng.shuffle(identity_pair_candidates)
    if len(identity_pair_candidates) < PAIRS_PER_CLASS_PER_FOLD:
        raise RuntimeError(f"Fold {fold} lacks enough distinct impostor identity pairs.")
    impostor_identity_pairs = identity_pair_candidates[:PAIRS_PER_CLASS_PER_FOLD]

    for identity, image_a_id, image_b_id in genuine_selected:
        attack_pair_rows.append(
            {
                "fold": fold,
                "pair_type": "mated_same_identity",
                "label": 1,
                "person_a": identity,
                "person_b": identity,
                "image_a": image_id_to_path[image_a_id],
                "image_b": image_id_to_path[image_b_id],
            }
        )

    for identity_a, identity_b in impostor_identity_pairs:
        image_a_id = int(pair_rng.choice(identity_to_image_ids[identity_a]))
        image_b_id = int(pair_rng.choice(identity_to_image_ids[identity_b]))
        if bool(pair_rng.integers(0, 2)):
            identity_a, identity_b = identity_b, identity_a
            image_a_id, image_b_id = image_b_id, image_a_id
        attack_pair_rows.append(
            {
                "fold": fold,
                "pair_type": "non_mated_different_identity",
                "label": 0,
                "person_a": identity_a,
                "person_b": identity_b,
                "image_a": image_id_to_path[image_a_id],
                "image_b": image_id_to_path[image_b_id],
            }
        )

    fold_protocol_rows.append(
        {
            "fold": fold,
            "identities": len(fold_identities),
            "valid_images": int(
                identity_fold_assignment_df.loc[
                    identity_fold_assignment_df["fold"] == fold, "valid_images"
                ].sum()
            ),
            "capped_genuine_capacity": int(
                identity_fold_assignment_df.loc[
                    identity_fold_assignment_df["fold"] == fold,
                    "capped_genuine_pair_capacity",
                ].sum()
            ),
            "genuine_pairs": PAIRS_PER_CLASS_PER_FOLD,
            "impostor_pairs": PAIRS_PER_CLASS_PER_FOLD,
        }
    )

pair_manifest_df = pd.DataFrame(attack_pair_rows)
pair_manifest_df.insert(0, "pair_id", np.arange(1, len(pair_manifest_df) + 1))
fold_protocol_df = pd.DataFrame(fold_protocol_rows)

fold_identity_sets = {
    fold: set(
        identity_fold_assignment_df.loc[
            identity_fold_assignment_df["fold"] == fold, "identity"
        ]
    )
    for fold in range(1, IDENTITY_FOLDS + 1)
}
overlap_rows = []
for test_fold in range(1, IDENTITY_FOLDS + 1):
    test_identities = fold_identity_sets[test_fold]
    train_identities = set().union(
        *[
            fold_identity_sets[fold]
            for fold in range(1, IDENTITY_FOLDS + 1)
            if fold != test_fold
        ]
    )
    overlap_rows.append(
        {
            "test_fold": test_fold,
            "train_identities": len(train_identities),
            "test_identities": len(test_identities),
            "train_test_identity_overlap": len(train_identities & test_identities),
        }
    )

identity_overlap_audit_df = pd.DataFrame(overlap_rows)
unordered_image_pairs = pair_manifest_df.apply(
    lambda row: "|".join(sorted([str(row["image_a"]), str(row["image_b"])])), axis=1
)

protocol_checks = {
    "total_pairs": len(pair_manifest_df),
    "genuine_pairs": int((pair_manifest_df["label"] == 1).sum()),
    "impostor_pairs": int((pair_manifest_df["label"] == 0).sum()),
    "folds": int(pair_manifest_df["fold"].nunique()),
    "valid_identities": int(identity_fold_assignment_df["identity"].nunique()),
    "valid_images_available": len(valid_rows_df),
    "max_train_test_identity_overlap": int(
        identity_overlap_audit_df["train_test_identity_overlap"].max()
    ),
    "duplicate_unordered_image_pairs": int(unordered_image_pairs.duplicated().sum()),
    "genuine_identity_rule_pass": bool(
        pair_manifest_df.loc[pair_manifest_df["label"] == 1, "person_a"].eq(
            pair_manifest_df.loc[pair_manifest_df["label"] == 1, "person_b"]
        ).all()
    ),
    "impostor_identity_rule_pass": bool(
        pair_manifest_df.loc[pair_manifest_df["label"] == 0, "person_a"].ne(
            pair_manifest_df.loc[pair_manifest_df["label"] == 0, "person_b"]
        ).all()
    ),
}

per_fold_class_counts = pair_manifest_df.groupby(["fold", "label"]).size().unstack()
expected_class_counts = np.full(
    (IDENTITY_FOLDS, 2), PAIRS_PER_CLASS_PER_FOLD, dtype=int
)

if len(pair_manifest_df) != EXPECTED_TOTAL_PAIRS:
    raise RuntimeError("Identity-disjoint pair count validation failed.")
if not np.array_equal(per_fold_class_counts[[0, 1]].to_numpy(), expected_class_counts):
    raise RuntimeError("Every fold must contain exactly 300 impostor and 300 genuine pairs.")
if protocol_checks["max_train_test_identity_overlap"] != 0:
    raise RuntimeError("Identity leakage detected between training and test folds.")
if protocol_checks["duplicate_unordered_image_pairs"] != 0:
    raise RuntimeError("A duplicate unordered image pair was generated.")
if not protocol_checks["genuine_identity_rule_pass"]:
    raise RuntimeError("A genuine pair contains different identities.")
if not protocol_checks["impostor_identity_rule_pass"]:
    raise RuntimeError("An impostor pair contains the same identity.")

identity_fold_assignment_df.to_csv(
    OUTPUT_DIR / "identity_fold_assignment.csv", index=False
)
pair_manifest_df.to_csv(
    OUTPUT_DIR / "identity_disjoint_pair_manifest.csv", index=False
)
fold_protocol_df.to_csv(OUTPUT_DIR / "identity_fold_protocol_summary.csv", index=False)
identity_overlap_audit_df.to_csv(
    OUTPUT_DIR / "identity_overlap_audit.csv", index=False
)
with open(OUTPUT_DIR / "identity_disjoint_protocol_audit.json", "w", encoding="utf-8") as file:
    json.dump(protocol_checks, file, indent=2)

failure_audit = {
    "failed_unique_images_excluded_before_pairing": len(failed_images_df),
    "attack_pairs_affected_by_failed_images": 0,
    "evaluable_pairs": len(pair_manifest_df),
    "evaluable_mated_pairs": int((pair_manifest_df["label"] == 1).sum()),
    "evaluable_non_mated_pairs": int((pair_manifest_df["label"] == 0).sum()),
}

print(json.dumps(protocol_checks, indent=2))
print(json.dumps(failure_audit, indent=2))
display(fold_protocol_df)
display(identity_overlap_audit_df)
print("Identity-disjoint protocol validation passed: every overlap count is zero.")


# Cell 24
def l2_normalize_rows(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def create_seeded_ro_matrix(embedding_dim, template_bits, seed):
    if template_bits > embedding_dim:
        raise ValueError("template_bits cannot exceed embedding_dim for reduced QR.")
    rng = np.random.default_rng(seed)
    gaussian_matrix = rng.standard_normal((embedding_dim, template_bits))
    q, r = np.linalg.qr(gaussian_matrix, mode="reduced")
    diagonal_signs = np.sign(np.diag(r))
    diagonal_signs[diagonal_signs == 0] = 1.0
    return (q * diagonal_signs).astype(np.float32)


torch.backends.cuda.matmul.allow_tf32 = False
valid_embeddings = l2_normalize_rows(embeddings[valid_embedding_mask])
valid_embeddings_gpu = torch.from_numpy(valid_embeddings).to("cuda")

biohashes_by_seed = {}
ro_validation = {}
template_generation_started = time.perf_counter()

for seed in RO_SEEDS:
    projection = create_seeded_ro_matrix(EMBEDDING_DIMENSION, TEMPLATE_BITS, seed)
    projection_gpu = torch.from_numpy(projection).to("cuda")

    torch.cuda.synchronize()
    started = time.perf_counter()
    projected_gpu = valid_embeddings_gpu @ projection_gpu
    valid_biohashes = (projected_gpu >= 0.0).cpu().numpy()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    biohashes = np.zeros((len(embeddings), TEMPLATE_BITS), dtype=bool)
    biohashes[valid_embedding_mask] = valid_biohashes
    biohashes_by_seed[seed] = biohashes

    orthogonality_error = float(
        np.max(
            np.abs(
                projection.T @ projection
                - np.eye(TEMPLATE_BITS, dtype=np.float32)
            )
        )
    )
    packed = np.packbits(biohashes, axis=1, bitorder="big")
    projection_path = OUTPUT_DIR / f"ro_projection_seed_{seed}.npy"
    template_path = OUTPUT_DIR / f"biohashes_packed_seed_{seed}.npy"
    np.save(projection_path, projection)
    np.save(template_path, packed)

    ro_validation[str(seed)] = {
        "projection_shape": list(projection.shape),
        "protected_template_shape": list(biohashes.shape),
        "valid_templates": int(valid_biohashes.shape[0]),
        "orthogonality_max_error": orthogonality_error,
        "overall_one_rate": float(valid_biohashes.mean()),
        "gpu_generation_ms": elapsed_ms,
        "packed_template_sha256": sha256_file(template_path),
    }

    del projection_gpu, projected_gpu

del valid_embeddings_gpu
torch.cuda.empty_cache()

template_generation_summary = {
    "device": runtime_audit["gpu_name"],
    "seed_sets": len(RO_SEEDS),
    "total_elapsed_ms": (time.perf_counter() - template_generation_started) * 1000,
    "raw_float32_bytes_per_template": EMBEDDING_DIMENSION * 4,
    "protected_packed_bytes_per_template": TEMPLATE_BITS // 8,
    "storage_reduction_factor": EMBEDDING_DIMENSION * 4 / (TEMPLATE_BITS // 8),
}

with open(OUTPUT_DIR / "ro_validation.json", "w", encoding="utf-8") as file:
    json.dump(ro_validation, file, indent=2)

metadata_df[["image_id", "image_path", "identity", "status"]].to_csv(
    OUTPUT_DIR / "template_index.csv", index=False
)

ro_validation_df = pd.DataFrame.from_dict(ro_validation, orient="index")
ro_validation_df.index.name = "seed"
display(
    ro_validation_df[
        ["valid_templates", "orthogonality_max_error", "overall_one_rate", "gpu_generation_ms"]
    ]
)
print(json.dumps(template_generation_summary, indent=2))

if not all(item["orthogonality_max_error"] < 1e-5 for item in ro_validation.values()):
    raise RuntimeError("At least one RO projection failed orthogonality validation.")
if set(biohashes_by_seed) != set(RO_SEEDS):
    raise RuntimeError("Not all 20 protected-template sets were generated.")


# Cell 26
path_to_id = dict(zip(metadata_df["image_path"], metadata_df["image_id"]))
image_a_ids = pair_manifest_df["image_a"].map(path_to_id).to_numpy(dtype=int)
image_b_ids = pair_manifest_df["image_b"].map(path_to_id).to_numpy(dtype=int)
matcher_evaluable = valid_embedding_mask[image_a_ids] & valid_embedding_mask[image_b_ids]

evaluable_pairs_df = pair_manifest_df.loc[matcher_evaluable].reset_index(drop=True)
a_ids = image_a_ids[matcher_evaluable]
b_ids = image_b_ids[matcher_evaluable]
pair_labels = evaluable_pairs_df["label"].to_numpy(dtype=np.int8)
pair_folds = evaluable_pairs_df["fold"].to_numpy(dtype=np.int8)


def hamming_similarity(template_a, template_b):
    return 1.0 - np.mean(np.logical_xor(template_a, template_b), axis=1)


# Balanced schedule: every key is the fresh query exactly once.
trial_rng = np.random.default_rng(KEY_TRIAL_RANDOM_SEED)
query_order = np.asarray(RO_SEEDS, dtype=np.int64).copy()
trial_rng.shuffle(query_order)

key_trial_rows = []
for trial_id, query_seed in enumerate(query_order, start=1):
    remaining = np.asarray(
        [seed for seed in RO_SEEDS if seed != int(query_seed)], dtype=np.int64
    )
    trial_rng.shuffle(remaining)
    exposure_order = remaining[:MAX_EXPOSURES].tolist()
    row = {
        "trial_id": trial_id,
        "query_seed": int(query_seed),
    }
    row.update(
        {
            f"exposure_seed_{position}": int(seed)
            for position, seed in enumerate(exposure_order, start=1)
        }
    )
    key_trial_rows.append(row)

key_trials_df = pd.DataFrame(key_trial_rows)
key_trials_df.to_csv(OUTPUT_DIR / "key_trial_schedule.csv", index=False)

if key_trials_df["query_seed"].nunique() != len(RO_SEEDS):
    raise RuntimeError("Each key must serve exactly once as the fresh query key.")
for _, row in key_trials_df.iterrows():
    exposures = [int(row[f"exposure_seed_{i}"]) for i in range(1, MAX_EXPOSURES + 1)]
    if len(set(exposures)) != MAX_EXPOSURES or int(row["query_seed"]) in exposures:
        raise RuntimeError("Invalid trial: exposures must be unique and exclude the query key.")

print("Key trials:", len(key_trials_df))
display(key_trials_df)


same_key_rows = []
for seed in RO_SEEDS:
    scores = hamming_similarity(
        biohashes_by_seed[seed][a_ids],
        biohashes_by_seed[seed][b_ids],
    )
    fpr, tpr, thresholds = roc_curve(pair_labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    same_key_rows.append(
        {
            "seed": seed,
            "key_group": "original_e02_e04" if seed in ORIGINAL_RO_SEEDS else "e05_extension",
            "evaluated_pairs": len(scores),
            "auc": float(roc_auc_score(pair_labels, scores)),
            "approximate_eer": float(0.5 * (fpr[eer_index] + fnr[eer_index])),
            "eer_threshold": float(thresholds[eer_index]),
            "genuine_mean": float(scores[pair_labels == 1].mean()),
            "impostor_mean": float(scores[pair_labels == 0].mean()),
            "genuine_impostor_mean_margin": float(
                scores[pair_labels == 1].mean() - scores[pair_labels == 0].mean()
            ),
        }
    )

same_key_sanity_df = pd.DataFrame(same_key_rows)
same_key_sanity_df.to_csv(OUTPUT_DIR / "same_key_recognition_sanity.csv", index=False)
worst_same_key_auc = float(same_key_sanity_df["auc"].min())

print("Worst same-key AUC across 20 keys:", worst_same_key_auc)
display(same_key_sanity_df)


# Cell 28
def standard_unlinkability_metrics(mated_scores, non_mated_scores, omega=1.0, n_bins=100):
    # Gómez-Barrero et al. local D(s) and global D_sys, identical to E03.
    mated_scores = np.asarray(mated_scores, dtype=np.float64)
    non_mated_scores = np.asarray(non_mated_scores, dtype=np.float64)
    lower = min(float(mated_scores.min()), float(non_mated_scores.min()))
    upper = max(float(mated_scores.max()), float(non_mated_scores.max()))
    if lower == upper:
        upper = np.nextafter(upper, np.inf)

    bin_edges = np.linspace(lower, upper, num=n_bins + 1, endpoint=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    mated_density = np.histogram(mated_scores, bins=bin_edges, density=True)[0]
    non_mated_density = np.histogram(non_mated_scores, bins=bin_edges, density=True)[0]

    likelihood_ratio = np.divide(
        mated_density,
        non_mated_density,
        out=np.ones_like(mated_density),
        where=non_mated_density != 0,
    )
    local_d = 2.0 * (omega * likelihood_ratio / (1.0 + omega * likelihood_ratio)) - 1.0
    local_d[omega * likelihood_ratio <= 1.0] = 0.0
    local_d[non_mated_density == 0] = 1.0

    integrate = getattr(np, "trapezoid", np.trapz)
    d_sys = float(integrate(local_d * mated_density, x=bin_centers))
    d_sys = float(np.clip(d_sys, 0.0, 1.0))
    return d_sys


def direction_aware_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    raw_auc = float(roc_auc_score(labels, scores))
    attacker_auc = max(raw_auc, 1.0 - raw_auc)
    mated = scores[labels == 1]
    non_mated = scores[labels == 0]
    return {
        "raw_auc": raw_auc,
        "attacker_auc": attacker_auc,
        "attacker_direction": "high_score_is_mated" if raw_auc >= 0.5 else "low_score_is_mated",
        "auc_linkability_index": 2.0 * (attacker_auc - 0.5),
        "d_sys": standard_unlinkability_metrics(mated, non_mated, omega=1.0, n_bins=100),
        "mated_mean": float(mated.mean()),
        "non_mated_mean": float(non_mated.mean()),
        "absolute_mean_gap": float(abs(mated.mean() - non_mated.mean())),
    }


def make_order_invariant_features(score_matrix, per_bit_agreement_rates):
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    per_bit_agreement_rates = np.asarray(per_bit_agreement_rates, dtype=np.float64)
    if per_bit_agreement_rates.shape[0] != score_matrix.shape[0]:
        raise ValueError("Score and per-bit feature row counts must match.")
    if per_bit_agreement_rates.shape[1] != TEMPLATE_BITS:
        raise ValueError("Per-bit agreement features must contain 128 columns.")
    sorted_scores = np.sort(score_matrix, axis=1)
    summaries = np.column_stack(
        [
            score_matrix.mean(axis=1),
            score_matrix.std(axis=1, ddof=0),
            score_matrix.min(axis=1),
            score_matrix.max(axis=1),
            np.median(score_matrix, axis=1),
            np.ptp(score_matrix, axis=1),
        ]
    )
    return np.column_stack([per_bit_agreement_rates, sorted_scores, summaries])


def select_linkage_rule(training_labels, training_scores):
    labels = np.asarray(training_labels, dtype=np.int8)
    scores = np.asarray(training_scores, dtype=np.float64)
    candidates = []

    for direction_sign, direction_name in [
        (1.0, "high_score_is_mated"),
        (-1.0, "low_score_is_mated"),
    ]:
        transformed = direction_sign * scores
        fpr, tpr, thresholds = roc_curve(
            labels, transformed, pos_label=1, drop_intermediate=False
        )
        balanced_accuracies = 0.5 * (tpr + (1.0 - fpr))
        finite = np.isfinite(thresholds)
        best_value = float(balanced_accuracies[finite].max())
        best_thresholds = thresholds[
            finite & np.isclose(balanced_accuracies, best_value, rtol=0, atol=1e-12)
        ]
        candidates.append(
            {
                "direction_sign": direction_sign,
                "direction": direction_name,
                "threshold_transformed": float(best_thresholds.max()),
                "training_balanced_accuracy": best_value,
            }
        )

    candidates.sort(
        key=lambda item: (item["training_balanced_accuracy"], item["threshold_transformed"]),
        reverse=True,
    )
    return candidates[0]


def apply_linkage_rule(labels, scores, rule):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (
        rule["direction_sign"] * scores >= rule["threshold_transformed"]
    ).astype(np.int8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return {
        "samples": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "link_detection_rate": tpr,
        "false_link_rate": 1.0 - tnr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "predictions": predictions,
    }


def pooled_binary_attack_summary(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "link_detection_rate": tpr,
        "false_link_rate": 1.0 - tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_mean_ci(values, seed, repetitions=BOOTSTRAP_REPETITIONS):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# Cell 30
setting_metric_rows = []
fold_attack_rows = []
oof_score_frames = []

attack_started = time.perf_counter()

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
        query_templates = biohashes_by_seed[query_seed][query_ids]
        all_bit_agreements = np.stack(
            [
                np.equal(
                    biohashes_by_seed[exposure_seed][gallery_ids],
                    query_templates,
                )
                for exposure_seed in exposure_order
            ],
            axis=1,
        )
        all_exposure_scores = all_bit_agreements.mean(axis=2)

        for exposure_count in EXPOSURE_COUNTS:
            used_seeds = exposure_order[:exposure_count]
            score_matrix = all_exposure_scores[:, :exposure_count]
            per_bit_agreement_rates = all_bit_agreements[
                :, :exposure_count, :
            ].mean(axis=1)
            mean_scores = score_matrix.mean(axis=1)
            maximum_scores = score_matrix.max(axis=1)
            attack_features = make_order_invariant_features(
                score_matrix, per_bit_agreement_rates
            )

            mean_metrics = direction_aware_metrics(pair_labels, mean_scores)
            maximum_metrics = direction_aware_metrics(pair_labels, maximum_scores)

            learned_oof_probability = np.full(len(pair_labels), np.nan, dtype=np.float64)
            learned_oof_prediction = np.full(len(pair_labels), -1, dtype=np.int8)

            for test_fold in sorted(np.unique(pair_folds)):
                train_mask = pair_folds != test_fold
                test_mask = pair_folds == test_fold

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
                model.fit(attack_features[train_mask], pair_labels[train_mask])

                training_probability = model.predict_proba(attack_features[train_mask])[:, 1]
                rule = select_linkage_rule(pair_labels[train_mask], training_probability)
                test_probability = model.predict_proba(attack_features[test_mask])[:, 1]
                fold_evaluation = apply_linkage_rule(
                    pair_labels[test_mask], test_probability, rule
                )
                test_predictions = fold_evaluation.pop("predictions")

                learned_oof_probability[test_mask] = test_probability
                learned_oof_prediction[test_mask] = test_predictions

                fold_attack_rows.append(
                    {
                        "trial_id": trial_id,
                        "orientation": orientation,
                        "exposure_count": exposure_count,
                        "query_seed": query_seed,
                        "exposure_seeds": "|".join(map(str, used_seeds)),
                        "test_fold": int(test_fold),
                        **rule,
                        **fold_evaluation,
                    }
                )

            if not np.isfinite(learned_oof_probability).all():
                raise RuntimeError("At least one out-of-fold probability is missing.")
            if not np.isin(learned_oof_prediction, [0, 1]).all():
                raise RuntimeError("At least one out-of-fold prediction is missing.")

            learned_metrics = direction_aware_metrics(
                pair_labels, learned_oof_probability
            )
            learned_binary = pooled_binary_attack_summary(
                pair_labels, learned_oof_prediction
            )

            setting_metric_rows.append(
                {
                    "trial_id": trial_id,
                    "orientation": orientation,
                    "exposure_count": exposure_count,
                    "query_seed": query_seed,
                    "exposure_seeds": "|".join(map(str, used_seeds)),
                    "feature_count": int(attack_features.shape[1]),
                    **{f"mean_{key}": value for key, value in mean_metrics.items()},
                    **{f"maximum_{key}": value for key, value in maximum_metrics.items()},
                    **{f"learned_{key}": value for key, value in learned_metrics.items()},
                    **{f"learned_{key}": value for key, value in learned_binary.items()},
                }
            )

            oof_frame = evaluable_pairs_df[["pair_id", "fold", "label"]].copy()
            oof_frame.insert(0, "trial_id", trial_id)
            oof_frame.insert(1, "orientation", orientation)
            oof_frame.insert(2, "exposure_count", exposure_count)
            oof_frame.insert(3, "query_seed", query_seed)
            oof_frame.insert(4, "exposure_seeds", "|".join(map(str, used_seeds)))
            oof_frame["mean_score"] = mean_scores
            oof_frame["maximum_score"] = maximum_scores
            oof_frame["learned_oof_probability"] = learned_oof_probability
            oof_frame["learned_oof_prediction"] = learned_oof_prediction
            oof_score_frames.append(oof_frame)

    elapsed_minutes = (time.perf_counter() - attack_started) / 60
    print(
        f"Completed key trial {trial_position + 1}/{len(key_trials_df)} "
        f"| elapsed {elapsed_minutes:.2f} min"
    )

setting_metrics_df = pd.DataFrame(setting_metric_rows)
fold_attack_df = pd.DataFrame(fold_attack_rows)
oof_scores_df = pd.concat(oof_score_frames, ignore_index=True)

setting_metrics_df.to_csv(OUTPUT_DIR / "cumulative_metrics_per_trial.csv", index=False)
fold_attack_df.to_csv(OUTPUT_DIR / "learned_attacker_fold_results.csv", index=False)
oof_scores_df.to_csv(
    OUTPUT_DIR / "cumulative_attack_out_of_fold_scores.csv.gz",
    index=False,
    compression="gzip",
)

expected_settings = len(key_trials_df) * 2 * len(EXPOSURE_COUNTS)
expected_fold_rows = expected_settings * 10
expected_oof_rows = expected_settings * EXPECTED_EVALUABLE_PAIRS

attack_audit = {
    "key_trials": len(key_trials_df),
    "directions": 2,
    "exposure_counts": EXPOSURE_COUNTS,
    "settings": len(setting_metrics_df),
    "fold_models": len(fold_attack_df),
    "out_of_fold_rows": len(oof_scores_df),
    "expected_settings": expected_settings,
    "expected_fold_models": expected_fold_rows,
    "expected_out_of_fold_rows": expected_oof_rows,
    "elapsed_minutes": (time.perf_counter() - attack_started) / 60,
}

print(json.dumps(attack_audit, indent=2))
if attack_audit["settings"] != expected_settings:
    raise RuntimeError("Cumulative setting count validation failed.")
if attack_audit["fold_models"] != expected_fold_rows:
    raise RuntimeError("Fold-model count validation failed.")
if attack_audit["out_of_fold_rows"] != expected_oof_rows:
    raise RuntimeError("Out-of-fold row count validation failed.")


# Cell 32
summary_rows = []
attacker_prefixes = {
    "mean_score": "mean",
    "maximum_score": "maximum",
    "learned": "learned",
}

for exposure_count in EXPOSURE_COUNTS:
    subset = setting_metrics_df[
        setting_metrics_df["exposure_count"] == exposure_count
    ]
    for attacker_name, prefix in attacker_prefixes.items():
        auc_values = subset[f"{prefix}_attacker_auc"].to_numpy(dtype=float)
        dsys_values = subset[f"{prefix}_d_sys"].to_numpy(dtype=float)
        auc_ci_low, auc_ci_high = bootstrap_mean_ci(
            auc_values,
            BOOTSTRAP_RANDOM_SEED + exposure_count * 10 + len(summary_rows),
        )
        dsys_ci_low, dsys_ci_high = bootstrap_mean_ci(
            dsys_values,
            BOOTSTRAP_RANDOM_SEED + exposure_count * 100 + len(summary_rows),
        )
        row = {
            "exposure_count": exposure_count,
            "attacker": attacker_name,
            "settings": len(subset),
            "mean_attacker_auc": float(auc_values.mean()),
            "auc_ci95_low": auc_ci_low,
            "auc_ci95_high": auc_ci_high,
            "std_attacker_auc": float(auc_values.std(ddof=1)),
            "worst_attacker_auc": float(auc_values.max()),
            "mean_d_sys": float(dsys_values.mean()),
            "d_sys_ci95_low": dsys_ci_low,
            "d_sys_ci95_high": dsys_ci_high,
            "worst_d_sys": float(dsys_values.max()),
        }
        if prefix == "learned":
            balanced = subset["learned_balanced_accuracy"].to_numpy(dtype=float)
            balanced_ci_low, balanced_ci_high = bootstrap_mean_ci(
                balanced,
                BOOTSTRAP_RANDOM_SEED + exposure_count * 1000,
            )
            row.update(
                {
                    "mean_balanced_accuracy": float(balanced.mean()),
                    "balanced_accuracy_ci95_low": balanced_ci_low,
                    "balanced_accuracy_ci95_high": balanced_ci_high,
                    "worst_balanced_accuracy": float(balanced.max()),
                }
            )
        summary_rows.append(row)

cumulative_summary_df = pd.DataFrame(summary_rows)
cumulative_summary_df.to_csv(
    OUTPUT_DIR / "cumulative_exposure_summary.csv", index=False
)

print("--- PRIMARY LEARNED ATTACKER ---")
display(cumulative_summary_df[cumulative_summary_df["attacker"] == "learned"])
print("--- TRANSPARENT MEAN/MAXIMUM BASELINES ---")
display(cumulative_summary_df[cumulative_summary_df["attacker"] != "learned"])


# Cell 34
growth_rows = []

for (trial_id, orientation), group in setting_metrics_df.groupby(
    ["trial_id", "orientation"], sort=False
):
    indexed = group.set_index("exposure_count")
    baseline = indexed.loc[1]
    for exposure_count in [2, 5, 10]:
        current = indexed.loc[exposure_count]
        growth_rows.append(
            {
                "trial_id": int(trial_id),
                "orientation": orientation,
                "exposure_count": exposure_count,
                "query_seed": int(current["query_seed"]),
                "learned_auc_growth_from_k1": float(
                    current["learned_attacker_auc"] - baseline["learned_attacker_auc"]
                ),
                "learned_d_sys_growth_from_k1": float(
                    current["learned_d_sys"] - baseline["learned_d_sys"]
                ),
                "learned_balanced_accuracy_growth_from_k1": float(
                    current["learned_balanced_accuracy"]
                    - baseline["learned_balanced_accuracy"]
                ),
            }
        )

growth_per_setting_df = pd.DataFrame(growth_rows)
growth_per_setting_df.to_csv(
    OUTPUT_DIR / "exposure_growth_per_setting.csv", index=False
)

growth_summary_rows = []
for exposure_count, group in growth_per_setting_df.groupby("exposure_count", sort=True):
    row = {"exposure_count": int(exposure_count), "paired_settings": len(group)}
    for metric in [
        "learned_auc_growth_from_k1",
        "learned_d_sys_growth_from_k1",
        "learned_balanced_accuracy_growth_from_k1",
    ]:
        values = group[metric].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            BOOTSTRAP_RANDOM_SEED + int(exposure_count) * 10000 + len(growth_summary_rows),
        )
        row[f"mean_{metric}"] = float(values.mean())
        row[f"{metric}_ci95_low"] = ci_low
        row[f"{metric}_ci95_high"] = ci_high
        row[f"worst_{metric}"] = float(values.max())
    growth_summary_rows.append(row)

growth_summary_df = pd.DataFrame(growth_summary_rows)
growth_summary_df.to_csv(OUTPUT_DIR / "exposure_growth_summary.csv", index=False)

print("Positive values mean that extra exposures helped the attacker.")
display(growth_summary_df)


# Cell 36
learned_summary_df = cumulative_summary_df[
    cumulative_summary_df["attacker"] == "learned"
].set_index("exposure_count")

maximum_mean_learned_auc = float(learned_summary_df["mean_attacker_auc"].max())
worst_learned_auc = float(setting_metrics_df["learned_attacker_auc"].max())
mean_k10_d_sys = float(learned_summary_df.loc[10, "mean_d_sys"])
mean_k10_balanced_accuracy = float(
    learned_summary_df.loc[10, "mean_balanced_accuracy"]
)
mean_k10_auc_growth = float(
    growth_summary_df.loc[
        growth_summary_df["exposure_count"] == 10,
        "mean_learned_auc_growth_from_k1",
    ].iloc[0]
)

identity_generalisation_by_k = {
    str(k): {
        "mean_auc": float(learned_summary_df.loc[k, "mean_attacker_auc"]),
        "key_setting_ci95_low": float(learned_summary_df.loc[k, "auc_ci95_low"]),
        "key_setting_ci95_high": float(learned_summary_df.loc[k, "auc_ci95_high"]),
        "clearly_above_chance": bool(learned_summary_df.loc[k, "auc_ci95_low"] > 0.50),
    }
    for k in EXPOSURE_COUNTS
}

if all(item["clearly_above_chance"] for item in identity_generalisation_by_k.values()):
    identity_generalisation_conclusion = "ABOVE_CHANCE_ON_UNSEEN_IDENTITIES_AT_ALL_TESTED_K"
elif any(item["clearly_above_chance"] for item in identity_generalisation_by_k.values()):
    identity_generalisation_conclusion = "ABOVE_CHANCE_ON_UNSEEN_IDENTITIES_AT_SOME_K"
else:
    identity_generalisation_conclusion = "NO_CLEAR_ABOVE_CHANCE_SIGNAL_ON_UNSEEN_IDENTITIES"

guardrail_checks = {
    "same_key_validity_pass": worst_same_key_auc >= MIN_WORST_SAME_KEY_AUC,
    "mean_learned_auc_pass": maximum_mean_learned_auc <= MAX_MEAN_LEARNED_ATTACKER_AUC,
    "worst_learned_auc_pass": worst_learned_auc <= MAX_WORST_LEARNED_ATTACKER_AUC,
    "k10_mean_d_sys_pass": mean_k10_d_sys <= MAX_MEAN_K10_DSYS,
    "k10_auc_growth_pass": mean_k10_auc_growth <= MAX_MEAN_AUC_GROWTH_K10_MINUS_K1,
    "k10_balanced_accuracy_pass": (
        mean_k10_balanced_accuracy <= MAX_MEAN_K10_BALANCED_ACCURACY
    ),
}

security_guardrail_conclusion = (
    "SUPPORTED_WITHIN_UNCHANGED_E05_GUARDRAILS"
    if all(guardrail_checks.values())
    else "ONE_OR_MORE_UNCHANGED_E05_GUARDRAILS_FAILED"
)

interpretation_summary = {
    "identity_generalisation_conclusion": identity_generalisation_conclusion,
    "security_guardrail_conclusion": security_guardrail_conclusion,
    "identity_generalisation_by_k": identity_generalisation_by_k,
    "worst_same_key_auc": worst_same_key_auc,
    "maximum_mean_learned_attacker_auc_across_k": maximum_mean_learned_auc,
    "worst_learned_attacker_auc": worst_learned_auc,
    "mean_k10_d_sys": mean_k10_d_sys,
    "mean_k10_balanced_accuracy": mean_k10_balanced_accuracy,
    "mean_k10_minus_k1_auc_growth": mean_k10_auc_growth,
    "guardrail_checks": guardrail_checks,
    "identity_overlap_maximum": protocol_checks["max_train_test_identity_overlap"],
    "scope": (
        "LFW identities assigned to 10 disjoint folds; 6,000 balanced generated pairs; "
        "20 deterministic RO keys; protected-template-only attacker; 128-bit templates."
    ),
}

with open(OUTPUT_DIR / "interpretation_summary.json", "w", encoding="utf-8") as file:
    json.dump(interpretation_summary, file, indent=2)

print(json.dumps(interpretation_summary, indent=2))


# Cell 38
attacker_colors = {
    "mean_score": "tab:blue",
    "maximum_score": "tab:orange",
    "learned": "tab:purple",
}

plt.figure(figsize=(8.8, 5.4))
for attacker_name in ["mean_score", "maximum_score", "learned"]:
    plot_df = cumulative_summary_df[
        cumulative_summary_df["attacker"] == attacker_name
    ].sort_values("exposure_count")
    lower_error = plot_df["mean_attacker_auc"] - plot_df["auc_ci95_low"]
    upper_error = plot_df["auc_ci95_high"] - plot_df["mean_attacker_auc"]
    plt.errorbar(
        plot_df["exposure_count"],
        plot_df["mean_attacker_auc"],
        yerr=np.vstack([lower_error, upper_error]),
        marker="o",
        capsize=4,
        linewidth=2,
        label=attacker_name.replace("_", " ").title(),
        color=attacker_colors[attacker_name],
    )
plt.axhline(0.5, color="black", linestyle="--", label="Chance = 0.50")
plt.axhline(
    MAX_MEAN_LEARNED_ATTACKER_AUC,
    color="tab:red",
    linestyle=":",
    label=f"Mean-AUC guardrail = {MAX_MEAN_LEARNED_ATTACKER_AUC:.2f}",
)
plt.xticks(EXPOSURE_COUNTS)
plt.xlabel("Number of exposed templates (K)")
plt.ylabel("Direction-aware attacker AUC")
plt.title("Identity-disjoint linkability as protected templates are exposed")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "attacker_auc_vs_exposures.png", dpi=180)
plt.show()

learned_plot_df = cumulative_summary_df[
    cumulative_summary_df["attacker"] == "learned"
].sort_values("exposure_count")
plt.figure(figsize=(8.2, 5.0))
plt.errorbar(
    learned_plot_df["exposure_count"],
    learned_plot_df["mean_d_sys"],
    yerr=np.vstack(
        [
            learned_plot_df["mean_d_sys"] - learned_plot_df["d_sys_ci95_low"],
            learned_plot_df["d_sys_ci95_high"] - learned_plot_df["mean_d_sys"],
        ]
    ),
    marker="o",
    capsize=4,
    linewidth=2,
    color="tab:green",
)
plt.axhline(0.0, color="black", linestyle="--", label="Ideal D_sys = 0")
plt.axhline(
    MAX_MEAN_K10_DSYS,
    color="tab:red",
    linestyle=":",
    label=f"K=10 mean guardrail = {MAX_MEAN_K10_DSYS:.2f}",
)
plt.xticks(EXPOSURE_COUNTS)
plt.xlabel("Number of exposed templates (K)")
plt.ylabel("Learned-attacker D_sys")
plt.title("Identity-disjoint D_sys under cumulative exposure")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learned_dsys_vs_exposures.png", dpi=180)
plt.show()

plt.figure(figsize=(7.8, 4.8))
growth_plot = growth_summary_df.sort_values("exposure_count")
plt.bar(
    growth_plot["exposure_count"].astype(str),
    growth_plot["mean_learned_auc_growth_from_k1"],
    color="tab:cyan",
    alpha=0.85,
)
plt.axhline(0.0, color="black", linestyle="--", label="No change from K=1")
plt.axhline(
    MAX_MEAN_AUC_GROWTH_K10_MINUS_K1,
    color="tab:red",
    linestyle=":",
    label=f"K=10 growth guardrail = {MAX_MEAN_AUC_GROWTH_K10_MINUS_K1:.2f}",
)
plt.xlabel("Number of exposed templates (K)")
plt.ylabel("Mean learned-AUC growth from K=1")
plt.title("Identity-disjoint AUC growth caused by extra exposures")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learned_auc_growth_from_k1.png", dpi=180)
plt.show()

same_key_plot_df = same_key_sanity_df.copy()
same_key_plot_df["seed_suffix"] = same_key_plot_df["seed"].astype(str).str[-4:]
colors = np.where(
    same_key_plot_df["key_group"] == "original_e02_e04",
    "tab:blue",
    "tab:gray",
)
plt.figure(figsize=(11, 4.8))
plt.bar(same_key_plot_df["seed_suffix"], same_key_plot_df["auc"], color=colors)
plt.axhline(
    MIN_WORST_SAME_KEY_AUC,
    color="tab:red",
    linestyle=":",
    label=f"Validity guardrail = {MIN_WORST_SAME_KEY_AUC:.2f}",
)
plt.ylim(max(0.90, worst_same_key_auc - 0.02), 1.0)
plt.xlabel("RO key seed suffix")
plt.ylabel("Same-key matcher AUC")
plt.title("Same-key validity on the identity-disjoint pair corpus")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "same_key_auc_by_seed.png", dpi=180)
plt.show()


# Cell 40
RUN_COMPLETED_UTC = datetime.now(timezone.utc).isoformat()

experiment_record = {
    "experiment_id": EXPERIMENT_ID,
    "title": EXPERIMENT_TITLE,
    "researcher": RESEARCHER_NAME,
    "institution": INSTITUTION,
    "run_started_utc": RUN_STARTED_UTC,
    "run_completed_utc": RUN_COMPLETED_UTC,
    "purpose": RUN_PURPOSE,
    "hypothesis": HYPOTHESIS,
    "threat_model": {
        "attacker_assets": "K historical protected bit strings and a fresh-key query bit string",
        "exposure_counts": EXPOSURE_COUNTS,
        "attacker_features": (
            "128 per-bit agreement rates, sorted cross-key Hamming scores, and "
            "order-invariant score summaries"
        ),
        "algorithm_public": True,
        "raw_embeddings_available_to_attacker": False,
        "face_images_available_to_attacker": False,
        "projection_matrices_used_by_attacker": False,
        "secret_seed_or_key_material_used_by_attacker": False,
        "reconstruction_or_impersonation_tested": False,
    },
    "source_cache_inventory": source_inventory_checks,
    "identity_disjoint_protocol": {
        **protocol_checks,
        "identity_folds": IDENTITY_FOLDS,
        "identity_split_random_seed": IDENTITY_SPLIT_RANDOM_SEED,
        "pair_sampling_random_seed": PAIR_SAMPLING_RANDOM_SEED,
        "pairs_per_class_per_fold": PAIRS_PER_CLASS_PER_FOLD,
        "max_genuine_pairs_per_identity": MAX_GENUINE_PAIRS_PER_IDENTITY,
        "fold_summary": json.loads(fold_protocol_df.to_json(orient="records")),
        "overlap_audit": json.loads(identity_overlap_audit_df.to_json(orient="records")),
    },
    "runtime": runtime_audit,
    "frozen_face_pipeline": model_audit,
    "embedding_extraction": extraction_summary,
    "failure_audit": failure_audit,
    "protected_template": {
        "method": "seeded Gaussian random matrix + reduced QR + zero binarisation",
        "input_dimension": EMBEDDING_DIMENSION,
        "template_bits": TEMPLATE_BITS,
        "original_frozen_seeds": ORIGINAL_RO_SEEDS,
        "e05_extension_seeds": E05_EXTENSION_RO_SEEDS,
        "all_seeds": RO_SEEDS,
        "template_generation": template_generation_summary,
        "validation": ro_validation,
    },
    "key_trial_protocol": {
        "random_seed": KEY_TRIAL_RANDOM_SEED,
        "trials": json.loads(key_trials_df.to_json(orient="records")),
        "nested_exposures": True,
        "query_key_excluded_from_exposures": True,
        "each_key_used_once_as_query": True,
        "image_directions": ["A_exposed_to_B_query", "B_exposed_to_A_query"],
    },
    "same_key_validity": json.loads(same_key_sanity_df.to_json(orient="records")),
    "attack_audit": attack_audit,
    "cumulative_summary": json.loads(cumulative_summary_df.to_json(orient="records")),
    "growth_summary": json.loads(growth_summary_df.to_json(orient="records")),
    "interpretation": interpretation_summary,
    "scope_limitations": [
        "The result is empirical and restricted to the valid images in the frozen 7,701-image LFW cache.",
        "The generated 6,000-pair corpus is deterministic and identity-disjoint but is not the official LFW verification protocol.",
        "A numerical difference from E05 cannot be attributed only to identity overlap because E05-ID necessarily rebuilds the pair corpus.",
        "Bootstrap intervals describe variability across key-trial/direction settings, not demographic population uncertainty.",
        "The attacker uses protected-template agreement and Hamming evidence but not compromised RO matrices or secret keys.",
        "Only K=1,2,5,10 and 20 deterministic RO keys are tested.",
        "Reconstruction, inversion, impersonation, and unlimited adaptive exposures are not covered.",
        "The unchanged guardrails are exploratory research controls, not certification thresholds.",
    ],
}

with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)


def metric_row_for_brief(exposure_count):
    row = learned_summary_df.loc[exposure_count]
    return (
        f"| {exposure_count} | {row['mean_attacker_auc']:.6f} "
        f"[{row['auc_ci95_low']:.6f}, {row['auc_ci95_high']:.6f}] | "
        f"{row['worst_attacker_auc']:.6f} | {row['mean_d_sys']:.6f} | "
        f"{row['mean_balanced_accuracy']:.6f} |"
    )


brief_metric_rows = "\n".join(metric_row_for_brief(k) for k in EXPOSURE_COUNTS)
clear_k = [
    k for k, item in identity_generalisation_by_k.items()
    if item["clearly_above_chance"]
]

brief = f"""# LFW-E05-ID — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Researcher:** {RESEARCHER_NAME}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

Can the E05 protected-template attacker link identities that were completely absent
from its training data, and does exposure growth from K=1 to K=10 materially help it?

## Controlled design

- Same frozen 512-D MobileFaceNet cache, 128-bit RO protection, 20 keys, and E05 attacker.
- Ten identity-disjoint folds; maximum train/test identity overlap: **{protocol_checks['max_train_test_identity_overlap']}**.
- {EXPECTED_TOTAL_PAIRS:,} generated pairs: {EXPECTED_GENUINE_PAIRS:,} genuine and {EXPECTED_IMPOSTOR_PAIRS:,} impostor.
- Every fold has {PAIRS_PER_CLASS_PER_FOLD} genuine and {PAIRS_PER_CLASS_PER_FOLD} impostor pairs.
- Genuine contribution capped at {MAX_GENUINE_PAIRS_PER_IDENTITY} pairs per identity.
- Nested K=1,2,5,10 exposures, 20 query-key trials, and both image directions.

## Primary learned-attacker results

| Exposed templates | Mean AUC [95% key-setting CI] | Worst AUC | Mean D_sys | Mean balanced accuracy |
|---:|---:|---:|---:|---:|
{brief_metric_rows}

Mean learned-attacker AUC growth from K=1 to K=10: **{mean_k10_auc_growth:.6f}**.  
Worst same-key AUC across 20 keys: **{worst_same_key_auc:.6f}**.

## Interpretation

**Identity generalisation:** `{identity_generalisation_conclusion}`  
K values whose key-setting interval is clearly above chance: **{clear_k if clear_k else 'none'}**.

**Unchanged E05 security guardrails:** `{security_guardrail_conclusion}`

- If E05-ID remains clearly above 0.50, the attack signal generalises to unseen
  identities and cannot be explained only by train/test identity overlap.
- Do not attribute the exact numerical difference from E05 solely to identity leakage:
  E05-ID necessarily rebuilds the pair corpus to enforce zero identity overlap.
- K=10 minus K=1 growth answers the separate question of whether more exposures help.

These are empirical conclusions for this split, attacker, key bank, and LFW cache. They
do not cover compromised RO keys, reconstruction, impersonation, or unlimited exposures.
"""

(OUTPUT_DIR / "LFW-E05-ID_One-Page_Experiment_Brief.md").write_text(
    brief, encoding="utf-8"
)
(OUTPUT_DIR / "RESULTS_README.md").write_text(
    brief + "\n\n## Frozen run notes\n\n" + RUN_NOTES + "\n",
    encoding="utf-8",
)

with open(OUTPUT_DIR / "research_log.md", "a", encoding="utf-8") as file:
    file.write(
        f"\n## Completion\n\nRun completed (UTC): {RUN_COMPLETED_UTC}.  \n"
        f"Identity generalisation: **{identity_generalisation_conclusion}**.  \n"
        f"Security guardrails: **{security_guardrail_conclusion}**.\n"
    )

manifest_records = []
for result_path in sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file()):
    manifest_records.append(
        {
            "relative_path": result_path.relative_to(OUTPUT_DIR).as_posix(),
            "size_bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
        }
    )
pd.DataFrame(manifest_records).to_csv(
    OUTPUT_DIR / "result_file_manifest.csv", index=False
)

print(brief)
print("Result files before checksum manifest:", len(manifest_records))


# Cell 42
import shutil
from IPython.display import FileLink, display

RESULTS_ARCHIVE_BASE = Path(
    "/kaggle/working/LFW-E05-ID-identity-disjoint-multi-exposure-RO-results"
)
results_archive_path = Path(
    shutil.make_archive(str(RESULTS_ARCHIVE_BASE), "zip", root_dir=OUTPUT_DIR)
)

os.chdir("/kaggle/working")
print("Results archive:", results_archive_path)
print("Archive size (MiB):", round(results_archive_path.stat().st_size / 1024**2, 2))
print("Click below to download:")
display(FileLink(results_archive_path.name))

print(
    "If the link does not open, use Kaggle's Output/Files panel and download "
    "LFW-E05-ID-identity-disjoint-multi-exposure-RO-results.zip."
)
