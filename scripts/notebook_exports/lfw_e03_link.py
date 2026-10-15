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
from sklearn.metrics import roc_auc_score, roc_curve


EXPERIMENT_ID = "LFW-E03-cross-key-unlinkability"
EXPERIMENT_TITLE = "Cross-Key Unlinkability of Seeded RO Protected Templates"

DATASET_ROOT = Path("/kaggle/input/datasets/jessicali9530/lfw-dataset")
IMAGE_ROOT = DATASET_ROOT / "lfw-deepfunneled" / "lfw-deepfunneled"
PAIRS_PATH = DATASET_ROOT / "pairs.csv"

CACHE_DIR = Path("/kaggle/working/experiments/LFW-E01-raw-baseline")
OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional example: Path("/kaggle/input/my-lfw-e01-cache")
# The folder must contain both raw_embeddings.npy and embedding_metadata.csv.
PRECOMPUTED_CACHE_DIR = None

EMBEDDING_DIMENSION = 512
TEMPLATE_BITS = 128
RO_SEEDS = [
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
]

DETECTION_THRESHOLD = 0.50
DETECTION_SIZE = (640, 640)
CHECKPOINT_INTERVAL = 500
RUN_PILOT_IF_EXTRACTION_IS_NEEDED = True
PILOT_SIZE = 100
PILOT_RANDOM_SEED = 2026
STRICT_FROZEN_COUNTS = True

EXPECTED_TOTAL_PAIRS = 6000
EXPECTED_GENUINE_PAIRS = 3000
EXPECTED_IMPOSTOR_PAIRS = 3000
EXPECTED_UNIQUE_IMAGES = 7701
EXPECTED_EVALUABLE_PAIRS = 5965
EXPECTED_VALID_EMBEDDINGS = 7682

# Exploratory decision guardrails fixed before looking at E03 results.
MAX_WORST_KEY_PAIR_ATTACKER_AUC = 0.55
MAX_WORST_KEY_PAIR_DSYS = 0.10
MAX_OOF_BALANCED_LINKAGE_ACCURACY = 0.55
MAX_SAME_IMAGE_ATTACKER_AUC = 0.55

KEY_PAIRS = list(combinations(RO_SEEDS, 2))

print("Experiment:", EXPERIMENT_ID)
print("Output directory:", OUTPUT_DIR)
print("Dataset root exists:", DATASET_ROOT.is_dir())
print("Pairs file exists:", PAIRS_PATH.is_file())
print("Fixed keys:", len(RO_SEEDS))
print("Unordered key pairs:", len(KEY_PAIRS))


# Cell 7
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
RUN_PURPOSE = (
    "Measure whether same-identity templates protected under different seeded "
    "RO keys can be distinguished from different-identity cross-key templates."
)
HYPOTHESIS = (
    "Cross-key mated and non-mated Hamming similarities will overlap near 0.50, "
    "giving a direction-aware attacker AUC and linkage accuracy near chance."
)
RUN_NOTES = """
- Frozen input: LFW-E01 MobileFaceNet embeddings and LFW official pair protocol.
- Frozen protection: five seeded RO transforms, 128 bits, threshold at zero.
- Controlled E03 change: compare protected templates produced under different keys.
- All 10 unordered key pairs and both image/key orientations are evaluated.
- Detection failures are excluded because no protected template exists to link.
- Same-image cross-key comparison is reported separately as a stronger diagnostic.
- E03 does not claim revocation, repeated-exposure, inversion, or impersonation security.
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

## Pre-registered exploratory guardrails

- Worst key-pair attacker AUC <= {MAX_WORST_KEY_PAIR_ATTACKER_AUC:.2f}
- Worst key-pair standard D_sys <= {MAX_WORST_KEY_PAIR_DSYS:.2f}
- Leakage-free out-of-fold balanced linkage accuracy <= {MAX_OOF_BALANCED_LINKAGE_ACCURACY:.2f}
- Worst same-image attacker AUC <= {MAX_SAME_IMAGE_ATTACKER_AUC:.2f}

## Notes

{RUN_NOTES}
"""

(OUTPUT_DIR / "research_log.md").write_text(initial_log, encoding="utf-8")
print(initial_log)


# Cell 9
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


# Cell 11
if not PAIRS_PATH.is_file():
    raise FileNotFoundError(
        "pairs.csv was not found. Add jessicali9530/lfw-dataset to the notebook."
    )

pairs_df = pd.read_csv(PAIRS_PATH)


def make_image_path(person, image_number):
    image_number = int(float(image_number))
    return IMAGE_ROOT / str(person) / f"{person}_{image_number:04d}.jpg"


manifest_rows = []
for index, row in pairs_df.iterrows():
    is_genuine = pd.isna(row["Unnamed: 3"])
    if is_genuine:
        person_a = str(row["name"])
        person_b = person_a
        image_a = make_image_path(person_a, row["imagenum1"])
        image_b = make_image_path(person_b, row["imagenum2"])
        label = 1
        pair_type = "mated_same_identity"
    else:
        person_a = str(row["name"])
        person_b = str(row["imagenum2"])
        image_a = make_image_path(person_a, row["imagenum1"])
        image_b = make_image_path(person_b, row["Unnamed: 3"])
        label = 0
        pair_type = "non_mated_different_identity"

    manifest_rows.append(
        {
            "pair_id": index + 1,
            "fold": (index // 600) + 1,
            "pair_type": pair_type,
            "label": label,
            "person_a": person_a,
            "person_b": person_b,
            "image_a": str(image_a),
            "image_b": str(image_b),
        }
    )

pair_manifest_df = pd.DataFrame(manifest_rows)
unique_image_paths = sorted(
    set(pair_manifest_df["image_a"]) | set(pair_manifest_df["image_b"])
)
missing_images = [path for path in unique_image_paths if not Path(path).is_file()]

protocol_checks = {
    "total_pairs": len(pair_manifest_df),
    "genuine_pairs": int((pair_manifest_df["label"] == 1).sum()),
    "impostor_pairs": int((pair_manifest_df["label"] == 0).sum()),
    "folds": int(pair_manifest_df["fold"].nunique()),
    "unique_images": len(unique_image_paths),
    "missing_images": len(missing_images),
}
expected_protocol = {
    "total_pairs": EXPECTED_TOTAL_PAIRS,
    "genuine_pairs": EXPECTED_GENUINE_PAIRS,
    "impostor_pairs": EXPECTED_IMPOSTOR_PAIRS,
    "folds": 10,
    "unique_images": EXPECTED_UNIQUE_IMAGES,
    "missing_images": 0,
}

if protocol_checks != expected_protocol:
    raise RuntimeError(
        f"LFW protocol validation failed. Obtained {protocol_checks}; "
        f"expected {expected_protocol}."
    )

pair_manifest_df.to_csv(OUTPUT_DIR / "pair_manifest.csv", index=False)
print(json.dumps(protocol_checks, indent=2))
print("Protocol validation passed.")


# Cell 13
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


# Cell 15
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


# Cell 17
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


# Cell 19
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


# Cell 21
failed_images_df = metadata_df[~valid_embedding_mask].copy()
failed_paths = set(failed_images_df["image_path"].astype(str))
affected_mask = (
    pair_manifest_df["image_a"].isin(failed_paths)
    | pair_manifest_df["image_b"].isin(failed_paths)
)
affected_pairs_df = pair_manifest_df[affected_mask].copy()

failed_images_df.to_csv(OUTPUT_DIR / "failed_images.csv", index=False)
affected_pairs_df.to_csv(OUTPUT_DIR / "affected_pairs.csv", index=False)

failure_audit = {
    "failed_unique_images": len(failed_images_df),
    "affected_pairs": len(affected_pairs_df),
    "evaluable_pairs": len(pair_manifest_df) - len(affected_pairs_df),
    "evaluable_mated_pairs": int(((~affected_mask) & (pair_manifest_df["label"] == 1)).sum()),
    "evaluable_non_mated_pairs": int(((~affected_mask) & (pair_manifest_df["label"] == 0)).sum()),
}

print(json.dumps(failure_audit, indent=2))
if STRICT_FROZEN_COUNTS and failure_audit["evaluable_pairs"] != EXPECTED_EVALUABLE_PAIRS:
    raise RuntimeError("Matcher-evaluable count differs from frozen E01/E02.")


# Cell 23
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
        np.max(np.abs(projection.T @ projection - np.eye(TEMPLATE_BITS, dtype=np.float32)))
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

print(json.dumps(ro_validation, indent=2))
print(json.dumps(template_generation_summary, indent=2))

if not all(item["orthogonality_max_error"] < 1e-5 for item in ro_validation.values()):
    raise RuntimeError("At least one RO projection failed orthogonality validation.")


# Cell 25
path_to_id = dict(zip(metadata_df["image_path"], metadata_df["image_id"]))
image_a_ids = pair_manifest_df["image_a"].map(path_to_id).to_numpy(dtype=int)
image_b_ids = pair_manifest_df["image_b"].map(path_to_id).to_numpy(dtype=int)
matcher_evaluable = valid_embedding_mask[image_a_ids] & valid_embedding_mask[image_b_ids]

base_evaluable_df = pair_manifest_df.loc[
    matcher_evaluable, ["pair_id", "fold", "pair_type", "label"]
].reset_index(drop=True)
a_ids = image_a_ids[matcher_evaluable]
b_ids = image_b_ids[matcher_evaluable]

cross_key_frames = []
scoring_started = time.perf_counter()

for key_a, key_b in KEY_PAIRS:
    orientations = [
        ("forward", key_a, key_b),
        ("reverse", key_b, key_a),
    ]
    for orientation, key_for_image_a, key_for_image_b in orientations:
        scores = 1.0 - np.mean(
            np.logical_xor(
                biohashes_by_seed[key_for_image_a][a_ids],
                biohashes_by_seed[key_for_image_b][b_ids],
            ),
            axis=1,
        )
        frame = base_evaluable_df.copy()
        frame.insert(0, "key_pair_a", key_a)
        frame.insert(1, "key_pair_b", key_b)
        frame.insert(2, "orientation", orientation)
        frame.insert(3, "key_for_image_a", key_for_image_a)
        frame.insert(4, "key_for_image_b", key_for_image_b)
        frame["cross_key_hamming_similarity"] = scores
        cross_key_frames.append(frame)

cross_key_scores_df = pd.concat(cross_key_frames, ignore_index=True)
cross_key_scores_df.to_csv(OUTPUT_DIR / "cross_key_pair_scores.csv", index=False)

cross_key_score_audit = {
    "unordered_key_pairs": len(KEY_PAIRS),
    "orientations_per_key_pair": 2,
    "evaluable_lfw_pairs": int(matcher_evaluable.sum()),
    "score_rows": len(cross_key_scores_df),
    "expected_score_rows": len(KEY_PAIRS) * 2 * EXPECTED_EVALUABLE_PAIRS,
    "cpu_scoring_elapsed_ms": (time.perf_counter() - scoring_started) * 1000,
}

print(json.dumps(cross_key_score_audit, indent=2))
if cross_key_score_audit["score_rows"] != cross_key_score_audit["expected_score_rows"]:
    raise RuntimeError("Cross-key score row-count validation failed.")
if not cross_key_scores_df["cross_key_hamming_similarity"].between(0, 1).all():
    raise RuntimeError("A cross-key Hamming similarity lies outside [0,1].")


# Cell 27
within_key_rows = []
for seed in RO_SEEDS:
    scores = 1.0 - np.mean(
        np.logical_xor(
            biohashes_by_seed[seed][a_ids],
            biohashes_by_seed[seed][b_ids],
        ),
        axis=1,
    )
    labels = base_evaluable_df["label"].to_numpy(dtype=np.int8)
    within_key_rows.append(
        {
            "seed": seed,
            "mated_mean": float(scores[labels == 1].mean()),
            "mated_std": float(scores[labels == 1].std(ddof=1)),
            "non_mated_mean": float(scores[labels == 0].mean()),
            "non_mated_std": float(scores[labels == 0].std(ddof=1)),
            "recognition_auc": float(roc_auc_score(labels, scores)),
        }
    )

within_key_reference_df = pd.DataFrame(within_key_rows)
within_key_reference_df.to_csv(OUTPUT_DIR / "within_key_reference_summary.csv", index=False)
display(within_key_reference_df)


# Cell 29
def standard_unlinkability_metrics(mated_scores, non_mated_scores, omega=1.0, n_bins=100):
    # Gómez-Barrero et al. local D(s) and global D_sys with equal-width histograms.
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

    # NumPy 2.x uses trapezoid; trapz is retained as a compatibility fallback.
    integrate = getattr(np, "trapezoid", np.trapz)
    d_sys = float(integrate(local_d * mated_density, x=bin_centers))
    d_sys = float(np.clip(d_sys, 0.0, 1.0))

    curve = pd.DataFrame(
        {
            "score": bin_centers,
            "mated_density": mated_density,
            "non_mated_density": non_mated_density,
            "likelihood_ratio": likelihood_ratio,
            "local_d": local_d,
        }
    )
    return d_sys, curve


def direction_aware_linkability_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    raw_auc = float(roc_auc_score(labels, scores))
    attacker_auc = max(raw_auc, 1.0 - raw_auc)
    mated = scores[labels == 1]
    non_mated = scores[labels == 0]
    d_sys, _ = standard_unlinkability_metrics(mated, non_mated, omega=1.0, n_bins=100)
    signed_gap = float(mated.mean() - non_mated.mean())
    return {
        "samples": len(scores),
        "mated_samples": len(mated),
        "non_mated_samples": len(non_mated),
        "raw_auc": raw_auc,
        "attacker_auc": attacker_auc,
        "attacker_direction": "high_score_is_mated" if raw_auc >= 0.5 else "low_score_is_mated",
        "auc_linkability_index": 2.0 * (attacker_auc - 0.5),
        "d_sys_omega_1_nbins_100": d_sys,
        "mated_mean": float(mated.mean()),
        "mated_std": float(mated.std(ddof=1)),
        "non_mated_mean": float(non_mated.mean()),
        "non_mated_std": float(non_mated.std(ddof=1)),
        "signed_mean_gap": signed_gap,
        "absolute_mean_gap": abs(signed_gap),
    }


per_orientation_rows = []
for (key_a, key_b, orientation), group in cross_key_scores_df.groupby(
    ["key_pair_a", "key_pair_b", "orientation"], sort=False
):
    per_orientation_rows.append(
        {
            "key_a": int(key_a),
            "key_b": int(key_b),
            "orientation": orientation,
            **direction_aware_linkability_metrics(
                group["label"], group["cross_key_hamming_similarity"]
            ),
        }
    )

per_key_pair_rows = []
for (key_a, key_b), group in cross_key_scores_df.groupby(
    ["key_pair_a", "key_pair_b"], sort=False
):
    per_key_pair_rows.append(
        {
            "key_a": int(key_a),
            "key_b": int(key_b),
            **direction_aware_linkability_metrics(
                group["label"], group["cross_key_hamming_similarity"]
            ),
        }
    )

per_orientation_metrics_df = pd.DataFrame(per_orientation_rows)
per_key_pair_metrics_df = pd.DataFrame(per_key_pair_rows)
pooled_linkability = direction_aware_linkability_metrics(
    cross_key_scores_df["label"],
    cross_key_scores_df["cross_key_hamming_similarity"],
)
pooled_d_sys, pooled_local_unlinkability_curve_df = standard_unlinkability_metrics(
    cross_key_scores_df.loc[cross_key_scores_df["label"] == 1, "cross_key_hamming_similarity"],
    cross_key_scores_df.loc[cross_key_scores_df["label"] == 0, "cross_key_hamming_similarity"],
    omega=1.0,
    n_bins=100,
)
if not np.isclose(pooled_d_sys, pooled_linkability["d_sys_omega_1_nbins_100"]):
    raise RuntimeError("Pooled D_sys consistency check failed.")

per_orientation_metrics_df.to_csv(
    OUTPUT_DIR / "linkability_metrics_per_orientation.csv", index=False
)
per_key_pair_metrics_df.to_csv(
    OUTPUT_DIR / "linkability_metrics_per_key_pair.csv", index=False
)
pooled_local_unlinkability_curve_df.to_csv(
    OUTPUT_DIR / "pooled_local_unlinkability_curve.csv", index=False
)
with open(OUTPUT_DIR / "pooled_linkability_metrics.json", "w", encoding="utf-8") as file:
    json.dump(pooled_linkability, file, indent=2)

print("--- POOLED CROSS-KEY LINKABILITY ---")
print(json.dumps(pooled_linkability, indent=2))
print("\n--- PER KEY PAIR (BOTH ORIENTATIONS) ---")
display(per_key_pair_metrics_df)


# Cell 31
valid_image_ids = np.flatnonzero(valid_embedding_mask)
same_image_frames = []

for key_a, key_b in KEY_PAIRS:
    scores = 1.0 - np.mean(
        np.logical_xor(
            biohashes_by_seed[key_a][valid_image_ids],
            biohashes_by_seed[key_b][valid_image_ids],
        ),
        axis=1,
    )
    same_image_frames.append(
        pd.DataFrame(
            {
                "key_a": key_a,
                "key_b": key_b,
                "image_id": valid_image_ids,
                "same_image_cross_key_similarity": scores,
            }
        )
    )

same_image_scores_df = pd.concat(same_image_frames, ignore_index=True)
same_image_scores_df.to_csv(OUTPUT_DIR / "same_image_cross_key_scores.csv", index=False)

same_image_metric_rows = []
for key_a, key_b in KEY_PAIRS:
    mated_scores = same_image_scores_df.loc[
        (same_image_scores_df["key_a"] == key_a)
        & (same_image_scores_df["key_b"] == key_b),
        "same_image_cross_key_similarity",
    ].to_numpy()
    non_mated_scores = cross_key_scores_df.loc[
        (cross_key_scores_df["key_pair_a"] == key_a)
        & (cross_key_scores_df["key_pair_b"] == key_b)
        & (cross_key_scores_df["label"] == 0),
        "cross_key_hamming_similarity",
    ].to_numpy()
    labels = np.concatenate(
        [np.ones(len(mated_scores), dtype=np.int8), np.zeros(len(non_mated_scores), dtype=np.int8)]
    )
    scores = np.concatenate([mated_scores, non_mated_scores])
    same_image_metric_rows.append(
        {
            "key_a": key_a,
            "key_b": key_b,
            **direction_aware_linkability_metrics(labels, scores),
        }
    )

same_image_metrics_df = pd.DataFrame(same_image_metric_rows)
same_image_metrics_df.to_csv(
    OUTPUT_DIR / "same_image_linkability_metrics_per_key_pair.csv", index=False
)

print("Same-image score rows:", len(same_image_scores_df))
display(same_image_metrics_df)


# Cell 33
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


def evaluate_linkage_rule(labels, scores, rule):
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


fold_rows = []
oof_frames = []
score_column = "cross_key_hamming_similarity"

for test_fold in sorted(cross_key_scores_df["fold"].unique()):
    train_df = cross_key_scores_df[cross_key_scores_df["fold"] != test_fold]
    test_df = cross_key_scores_df[cross_key_scores_df["fold"] == test_fold].copy()

    rule = select_linkage_rule(train_df["label"], train_df[score_column])
    evaluation = evaluate_linkage_rule(test_df["label"], test_df[score_column], rule)
    predictions = evaluation.pop("predictions")

    fold_rows.append({"fold": int(test_fold), **rule, **evaluation})
    test_df["attacker_prediction"] = predictions
    test_df["attacker_direction"] = rule["direction"]
    test_df["attacker_threshold_transformed"] = rule["threshold_transformed"]
    oof_frames.append(test_df)

linkage_attack_fold_df = pd.DataFrame(fold_rows)
linkage_attack_oof_df = pd.concat(oof_frames, ignore_index=True)

pooled_tp = int(((linkage_attack_oof_df["attacker_prediction"] == 1) & (linkage_attack_oof_df["label"] == 1)).sum())
pooled_tn = int(((linkage_attack_oof_df["attacker_prediction"] == 0) & (linkage_attack_oof_df["label"] == 0)).sum())
pooled_fp = int(((linkage_attack_oof_df["attacker_prediction"] == 1) & (linkage_attack_oof_df["label"] == 0)).sum())
pooled_fn = int(((linkage_attack_oof_df["attacker_prediction"] == 0) & (linkage_attack_oof_df["label"] == 1)).sum())
pooled_tpr = pooled_tp / (pooled_tp + pooled_fn)
pooled_tnr = pooled_tn / (pooled_tn + pooled_fp)

oof_attack_summary = {
    "samples": len(linkage_attack_oof_df),
    "balanced_accuracy": 0.5 * (pooled_tpr + pooled_tnr),
    "link_detection_rate": pooled_tpr,
    "false_link_rate": 1.0 - pooled_tnr,
    "tp": pooled_tp,
    "tn": pooled_tn,
    "fp": pooled_fp,
    "fn": pooled_fn,
    "mean_fold_balanced_accuracy": float(linkage_attack_fold_df["balanced_accuracy"].mean()),
    "fold_balanced_accuracy_std": float(linkage_attack_fold_df["balanced_accuracy"].std(ddof=1)),
}

linkage_attack_fold_df.to_csv(OUTPUT_DIR / "linkage_attack_fold_results.csv", index=False)
linkage_attack_oof_df.to_csv(OUTPUT_DIR / "linkage_attack_out_of_fold_decisions.csv", index=False)
with open(OUTPUT_DIR / "linkage_attack_summary.json", "w", encoding="utf-8") as file:
    json.dump(oof_attack_summary, file, indent=2)

display(linkage_attack_fold_df)
print(json.dumps(oof_attack_summary, indent=2))


# Cell 35
worst_key_pair_auc = float(per_key_pair_metrics_df["attacker_auc"].max())
worst_key_pair_gap = float(per_key_pair_metrics_df["absolute_mean_gap"].max())
worst_key_pair_d_sys = float(per_key_pair_metrics_df["d_sys_omega_1_nbins_100"].max())
worst_same_image_auc = float(same_image_metrics_df["attacker_auc"].max())
oof_balanced_accuracy = float(oof_attack_summary["balanced_accuracy"])

guardrail_checks = {
    "worst_key_pair_attacker_auc_pass": worst_key_pair_auc <= MAX_WORST_KEY_PAIR_ATTACKER_AUC,
    "worst_key_pair_d_sys_pass": worst_key_pair_d_sys <= MAX_WORST_KEY_PAIR_DSYS,
    "oof_balanced_linkage_accuracy_pass": oof_balanced_accuracy <= MAX_OOF_BALANCED_LINKAGE_ACCURACY,
    "same_image_attacker_auc_pass": worst_same_image_auc <= MAX_SAME_IMAGE_ATTACKER_AUC,
}

e03_conclusion = (
    "SUPPORTED_WITHIN_E03_GUARDRAILS"
    if all(guardrail_checks.values())
    else "DETECTABLE_CROSS_KEY_LINKAGE_OBSERVED"
)

interpretation_summary = {
    "conclusion": e03_conclusion,
    "pooled_attacker_auc": pooled_linkability["attacker_auc"],
    "pooled_auc_linkability_index": pooled_linkability["auc_linkability_index"],
    "pooled_d_sys": pooled_linkability["d_sys_omega_1_nbins_100"],
    "worst_key_pair_attacker_auc": worst_key_pair_auc,
    "worst_key_pair_d_sys": worst_key_pair_d_sys,
    "worst_key_pair_absolute_mean_gap": worst_key_pair_gap,
    "oof_balanced_linkage_accuracy": oof_balanced_accuracy,
    "worst_same_image_attacker_auc": worst_same_image_auc,
    "guardrail_checks": guardrail_checks,
    "scope": "Empirical evidence for LFW, five fixed RO keys, and 128-bit templates only.",
}

with open(OUTPUT_DIR / "interpretation_summary.json", "w", encoding="utf-8") as file:
    json.dump(interpretation_summary, file, indent=2)

print(json.dumps(interpretation_summary, indent=2))


# Cell 37
score_bins = np.arange(-0.5 / TEMPLATE_BITS, 1.0 + 1.5 / TEMPLATE_BITS, 1 / TEMPLATE_BITS)

plt.figure(figsize=(8.5, 5.2))
for label, name, color in [
    (0, "Cross-key non-mated", "tab:blue"),
    (1, "Cross-key mated", "tab:orange"),
]:
    values = cross_key_scores_df.loc[
        cross_key_scores_df["label"] == label,
        "cross_key_hamming_similarity",
    ]
    plt.hist(values, bins=score_bins, density=True, alpha=0.55, label=name, color=color)

plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance similarity 0.50")
plt.xlabel("Normalized cross-key Hamming similarity")
plt.ylabel("Density")
plt.title("LFW-E03 cross-key mated vs non-mated distributions")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cross_key_score_distributions.png", dpi=180)
plt.show()

fig, density_axis = plt.subplots(figsize=(8.5, 5.2))
density_axis.plot(
    pooled_local_unlinkability_curve_df["score"],
    pooled_local_unlinkability_curve_df["mated_density"],
    label="Mated density",
    color="tab:orange",
)
density_axis.plot(
    pooled_local_unlinkability_curve_df["score"],
    pooled_local_unlinkability_curve_df["non_mated_density"],
    label="Non-mated density",
    color="tab:blue",
    linestyle="--",
)
density_axis.set_xlabel("Normalized cross-key Hamming similarity")
density_axis.set_ylabel("Probability density")

local_axis = density_axis.twinx()
local_axis.plot(
    pooled_local_unlinkability_curve_df["score"],
    pooled_local_unlinkability_curve_df["local_d"],
    label="Local D(s)",
    color="tab:green",
)
local_axis.set_ylabel("Local linkability D(s)")
local_axis.set_ylim(0, 1.05)

handles_a, labels_a = density_axis.get_legend_handles_labels()
handles_b, labels_b = local_axis.get_legend_handles_labels()
density_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
density_axis.set_title(
    f"Standard unlinkability analysis: D_sys={pooled_linkability['d_sys_omega_1_nbins_100']:.4f}"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "standard_unlinkability_dsys_curve.png", dpi=180)
plt.show()

plot_df = per_key_pair_metrics_df.copy()
plot_df["key_pair"] = plot_df["key_a"].astype(str).str[-4:] + "-" + plot_df["key_b"].astype(str).str[-4:]

plt.figure(figsize=(10, 5))
plt.bar(plot_df["key_pair"], plot_df["attacker_auc"], color="tab:purple", alpha=0.8)
plt.axhline(0.5, color="black", linestyle="--", label="Chance = 0.50")
plt.axhline(
    MAX_WORST_KEY_PAIR_ATTACKER_AUC,
    color="tab:red",
    linestyle=":",
    label=f"E03 guardrail = {MAX_WORST_KEY_PAIR_ATTACKER_AUC:.2f}",
)
plt.ylim(0.48, max(0.60, float(plot_df["attacker_auc"].max()) + 0.03))
plt.xlabel("RO key pair (seed suffixes)")
plt.ylabel("Direction-aware attacker AUC")
plt.title("Cross-key linkability by key pair")
plt.xticks(rotation=40, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "attacker_auc_by_key_pair.png", dpi=180)
plt.show()


# Cell 39
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
    "dataset_protocol": protocol_checks,
    "runtime": runtime_audit,
    "frozen_face_pipeline": model_audit,
    "embedding_extraction": extraction_summary,
    "failure_audit": failure_audit,
    "protected_template": {
        "method": "seeded Gaussian random matrix + reduced QR + zero binarisation",
        "input_dimension": EMBEDDING_DIMENSION,
        "template_bits": TEMPLATE_BITS,
        "seeds": RO_SEEDS,
        "key_pairs": [list(pair) for pair in KEY_PAIRS],
        "template_generation": template_generation_summary,
        "validation": ro_validation,
    },
    "cross_key_protocol": cross_key_score_audit,
    "pooled_linkability": pooled_linkability,
    "per_key_pair_linkability": json.loads(per_key_pair_metrics_df.to_json(orient="records")),
    "same_image_linkability": json.loads(same_image_metrics_df.to_json(orient="records")),
    "leakage_free_attack": oof_attack_summary,
    "interpretation": interpretation_summary,
    "scope_limitations": [
        "The conclusion is empirical and restricted to LFW, five fixed seeded RO keys, and 128-bit templates.",
        "Five frozen E02 keys yield 10 key pairs; some published protocols recommend 10 or more independently generated keys, so a larger-key robustness study may be required before publication.",
        "Official LFW pairs are used; the experiment is not an identity-disjoint population study.",
        "D_sys uses omega=1 and 100 equal-width bins following the published reference defaults.",
        "AUC-derived L_AUC is a transparent complementary index, not claimed as an ISO-standard metric.",
        "The four pass/fail limits are pre-registered E03 guardrails, not universal unlinkability standards.",
        "Revocation, repeated compromise, key exposure, reconstruction, and impersonation are not tested here.",
        "LFW has known demographic and capture-condition limitations.",
    ],
}

with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)


def pct(value):
    return f"{100 * float(value):.3f}%"


brief = f"""# LFW-E03 — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Researcher:** {RESEARCHER_NAME}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

Can an attacker decide whether two 128-bit protected face templates generated under different seeded RO keys belong to the same identity?

## Controlled design

- Frozen LFW deep-funnelled protocol: 6,000 pairs and 10 folds.
- Frozen 512-D MobileFaceNet embeddings: {extraction_summary['valid_embeddings']} valid of {EXPECTED_UNIQUE_IMAGES} images.
- Five fixed RO keys produce {len(KEY_PAIRS)} unordered key pairs.
- Both key/image orientations are scored: {cross_key_score_audit['score_rows']:,} matcher-only comparisons.
- Established global metric is D_sys (omega=1, 100 bins); ideal is 0.
- Complementary attacker metric is direction-aware AUC: `max(AUC, 1-AUC)`; ideal is 0.50.
- A 10-fold attacker learns direction and threshold from nine folds and is tested on the held-out fold.

## Main results

| Measure | Result | Ideal |
|---|---:|---:|
| Pooled standard D_sys | {pooled_linkability['d_sys_omega_1_nbins_100']:.6f} | 0.000000 |
| Pooled attacker AUC | {pooled_linkability['attacker_auc']:.6f} | 0.500000 |
| Pooled AUC linkability index | {pooled_linkability['auc_linkability_index']:.6f} | 0.000000 |
| Worst key-pair attacker AUC | {worst_key_pair_auc:.6f} | 0.500000 |
| Worst key-pair D_sys | {worst_key_pair_d_sys:.6f} | 0.000000 |
| Worst absolute mated/non-mated mean gap | {worst_key_pair_gap:.6f} | 0.000000 |
| Out-of-fold balanced linkage accuracy | {oof_balanced_accuracy:.6f} | 0.500000 |
| Worst same-image attacker AUC | {worst_same_image_auc:.6f} | 0.500000 |

Cross-key mated similarity: {pooled_linkability['mated_mean']:.6f} ± {pooled_linkability['mated_std']:.6f}.  
Cross-key non-mated similarity: {pooled_linkability['non_mated_mean']:.6f} ± {pooled_linkability['non_mated_std']:.6f}.

## E03 interpretation

**{e03_conclusion}**

This label applies only to the pre-registered E03 guardrails. It is not a proof of universal unlinkability. Security interpretation must include the worst key pair and same-image diagnostic, not only pooled AUC.

## Next experiment boundary

LFW-E04 should test revocation operationally: invalidate an old key version, issue a new version, confirm legitimate verification under the new version, and confirm that the revoked version is rejected. Repeated compromise remains reserved for LFW-E05.
"""

(OUTPUT_DIR / "LFW-E03_One-Page_Experiment_Brief.md").write_text(brief, encoding="utf-8")

results_readme = f"""# {EXPERIMENT_ID} results

{brief}

## Interpretation rules

- A raw AUC below 0.5 is reversible; always use direction-aware attacker AUC.
- Detection failures are omitted from unlinkability scoring because no protected template exists.
- Per-key-pair results are primary for weakness discovery; pooled values are supporting summaries.
- Threshold attack results are leakage-free across the official folds.
- Do not use E03 to claim revocation, repeated-compromise, inversion, or impersonation resistance.

## Editable run notes

{RUN_NOTES}
"""

(OUTPUT_DIR / "RESULTS_README.md").write_text(results_readme, encoding="utf-8")

with open(OUTPUT_DIR / "research_log.md", "a", encoding="utf-8") as file:
    file.write(
        f"\n## Completion\n\nRun completed (UTC): {RUN_COMPLETED_UTC}.  \n"
        f"Conclusion: **{e03_conclusion}**.\n"
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


# Cell 41
import shutil
from IPython.display import FileLink, display

RESULTS_ARCHIVE_BASE = Path("/kaggle/working/LFW-E03-cross-key-unlinkability-results")
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
    "LFW-E03-cross-key-unlinkability-results.zip from /kaggle/working."
)
