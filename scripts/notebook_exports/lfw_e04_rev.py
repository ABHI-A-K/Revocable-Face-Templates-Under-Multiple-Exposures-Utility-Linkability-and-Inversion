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
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display


EXPERIMENT_ID = "LFW-E04-revocation"
EXPERIMENT_TITLE = "Operational Revocation of Seeded RO Protected Templates"

DATASET_ROOT = Path("/kaggle/input/datasets/jessicali9530/lfw-dataset")
IMAGE_ROOT = DATASET_ROOT / "lfw-deepfunneled" / "lfw-deepfunneled"
PAIRS_PATH = DATASET_ROOT / "pairs.csv"

CACHE_DIR = Path("/kaggle/working/experiments/LFW-E01-raw-baseline")
OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional example: Path("/kaggle/input/my-lfw-e01-cache")
# The folder must contain raw_embeddings.npy and embedding_metadata.csv.
PRECOMPUTED_CACHE_DIR = Path("/kaggle/input/datasets/abhiak0709/cache-embeddings-from-lfw-e03")

EMBEDDING_DIMENSION = 512
TEMPLATE_BITS = 128
RO_SEEDS = [
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
]

# Every different old->new key choice is tested: 5 x 4 = 20 transitions.
KEY_TRANSITIONS = [
    (old_seed, new_seed)
    for old_seed in RO_SEEDS
    for new_seed in RO_SEEDS
    if old_seed != new_seed
]

DETECTION_THRESHOLD = 0.50
DETECTION_SIZE = (640, 640)
CHECKPOINT_INTERVAL = 500
RUN_PILOT_IF_EXTRACTION_IS_NEEDED = True
PILOT_SIZE = 100
PILOT_RANDOM_SEED = 2026
TARGET_TRAIN_FAR = 1e-3
STRICT_FROZEN_COUNTS = True

EXPECTED_TOTAL_PAIRS = 6000
EXPECTED_GENUINE_PAIRS = 3000
EXPECTED_IMPOSTOR_PAIRS = 3000
EXPECTED_UNIQUE_IMAGES = 7701
EXPECTED_EVALUABLE_PAIRS = 5965
EXPECTED_VALID_EMBEDDINGS = 7682

# Exploratory E04 success guardrails fixed before viewing E04 results.
MIN_WORST_CURRENT_VERSION_TAR = 0.95
MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE = 0.01
MAX_WORST_ABSOLUTE_RENEWAL_TAR_CHANGE = 0.02
REQUIRED_VERSION_GATE_REJECTION_RATE = 1.00

print("Experiment:", EXPERIMENT_ID)
print("Output directory:", OUTPUT_DIR)
print("Dataset root exists:", DATASET_ROOT.is_dir())
print("Pairs file exists:", PAIRS_PATH.is_file())
print("Fixed key versions:", len(RO_SEEDS))
print("Ordered old->new transitions:", len(KEY_TRANSITIONS))


# Cell 8
RESEARCHER_NAME = "Abhishek A K"
INSTITUTION = "IIIT Dharwad / MIT Manipal"
RUN_PURPOSE = (
    "Test whether an old protected-template version can be revoked, replaced by a "
    "new keyed version without major recognition loss, and rejected if replayed."
)
HYPOTHESIS = (
    "A current-version template will retain strong same-key recognition, while a "
    "revoked old-key template will be rejected by the version gate and will rarely "
    "pass the current version's matcher threshold if metadata is forged or removed."
)
RUN_NOTES = """
- Frozen input: official LFW protocol and E01 MobileFaceNet embeddings.
- Frozen protection: five seeded RO transforms, 128 bits, zero binarisation.
- All 20 ordered old-key -> new-key transitions are tested.
- Current-version thresholds are learned only from nine training folds.
- Revoked-template scores never influence threshold selection.
- The version gate and the matcher fallback are reported separately.
- Seeds are reproducible experimental stand-ins for keys, not production secrets.
- E04 tests one revocation event; repeated compromise remains LFW-E05.
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

## Threat model

An attacker has a revoked 128-bit protected template. The attacker may replay it with
its original version metadata or may remove/forge that metadata so the current matcher
receives the old bit string.

## Pre-registered exploratory guardrails

- Worst current-version matcher-only TAR >= {MIN_WORST_CURRENT_VERSION_TAR:.2f}
- Worst old-template acceptance rate <= {MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE:.2f}
- Worst absolute TAR change between old and renewed key versions <= {MAX_WORST_ABSOLUTE_RENEWAL_TAR_CHANGE:.2f}
- Version-gate rejection rate = {REQUIRED_VERSION_GATE_REJECTION_RATE:.2f}

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


# Cell 26
path_to_id = dict(zip(metadata_df["image_path"], metadata_df["image_id"]))
image_a_ids = pair_manifest_df["image_a"].map(path_to_id).to_numpy(dtype=int)
image_b_ids = pair_manifest_df["image_b"].map(path_to_id).to_numpy(dtype=int)
matcher_evaluable = valid_embedding_mask[image_a_ids] & valid_embedding_mask[image_b_ids]

base_pairs_df = pair_manifest_df[
    ["pair_id", "fold", "pair_type", "label", "person_a", "person_b"]
].copy()
base_pairs_df["matcher_evaluable"] = matcher_evaluable


def hamming_similarity(template_a, template_b):
    return 1.0 - np.mean(np.logical_xor(template_a, template_b), axis=1)


current_frames = []
for current_seed in RO_SEEDS:
    scores = np.full(len(pair_manifest_df), np.nan, dtype=np.float64)
    scores[matcher_evaluable] = hamming_similarity(
        biohashes_by_seed[current_seed][image_a_ids[matcher_evaluable]],
        biohashes_by_seed[current_seed][image_b_ids[matcher_evaluable]],
    )
    frame = base_pairs_df.copy()
    frame.insert(0, "current_seed", current_seed)
    frame["current_hamming_similarity"] = scores
    current_frames.append(frame)

current_scores_df = pd.concat(current_frames, ignore_index=True)

revoked_frames = []
evaluable_base = base_pairs_df.loc[matcher_evaluable].reset_index(drop=True)
a_ids = image_a_ids[matcher_evaluable]
b_ids = image_b_ids[matcher_evaluable]

for old_seed, new_seed in KEY_TRANSITIONS:
    scores = hamming_similarity(
        biohashes_by_seed[new_seed][a_ids],
        biohashes_by_seed[old_seed][b_ids],
    )
    frame = evaluable_base.copy()
    frame.insert(0, "old_seed", old_seed)
    frame.insert(1, "new_seed", new_seed)
    frame["revoked_to_current_hamming_similarity"] = scores
    revoked_frames.append(frame)

revoked_scores_df = pd.concat(revoked_frames, ignore_index=True)

current_scores_df.to_csv(OUTPUT_DIR / "current_version_pair_scores.csv", index=False)
revoked_scores_df.to_csv(OUTPUT_DIR / "revoked_to_current_pair_scores.csv", index=False)

score_table_audit = {
    "current_key_versions": len(RO_SEEDS),
    "ordered_revocation_transitions": len(KEY_TRANSITIONS),
    "current_score_rows": len(current_scores_df),
    "current_matcher_evaluable_rows": int(current_scores_df["matcher_evaluable"].sum()),
    "revoked_score_rows": len(revoked_scores_df),
    "expected_current_rows": len(RO_SEEDS) * EXPECTED_TOTAL_PAIRS,
    "expected_revoked_rows": len(KEY_TRANSITIONS) * EXPECTED_EVALUABLE_PAIRS,
}

print(json.dumps(score_table_audit, indent=2))
if score_table_audit["current_score_rows"] != score_table_audit["expected_current_rows"]:
    raise RuntimeError("Current-version score row-count validation failed.")
if score_table_audit["revoked_score_rows"] != score_table_audit["expected_revoked_rows"]:
    raise RuntimeError("Revoked-version score row-count validation failed.")


# Cell 28
def select_threshold_at_far(scores, labels, target_far):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    impostor_scores = scores[labels == 0]
    if not len(impostor_scores):
        raise ValueError("No impostor scores are available for FAR calibration.")

    allowed_false_accepts = int(np.floor(target_far * len(impostor_scores)))
    descending_scores = np.sort(impostor_scores)[::-1]
    threshold = (
        -np.inf
        if allowed_false_accepts >= len(descending_scores)
        else np.nextafter(descending_scores[allowed_false_accepts], np.inf)
    )
    achieved_false_accepts = int((impostor_scores >= threshold).sum())
    achieved_far = achieved_false_accepts / len(impostor_scores)
    if achieved_far > target_far:
        raise RuntimeError("Training FAR exceeded the target.")
    return {
        "threshold": float(threshold),
        "training_impostor_pairs": len(impostor_scores),
        "training_allowed_fp": allowed_false_accepts,
        "training_achieved_fp": achieved_false_accepts,
        "training_achieved_far": achieved_far,
    }


threshold_rows = []
current_oof_frames = []
revoked_oof_frames = []

for new_seed in RO_SEEDS:
    seed_current_df = current_scores_df[
        (current_scores_df["current_seed"] == new_seed)
        & current_scores_df["matcher_evaluable"]
    ].copy()

    for test_fold in sorted(seed_current_df["fold"].unique()):
        train_df = seed_current_df[seed_current_df["fold"] != test_fold]
        test_current_df = current_scores_df[
            (current_scores_df["current_seed"] == new_seed)
            & (current_scores_df["fold"] == test_fold)
        ].copy()

        calibration = select_threshold_at_far(
            train_df["current_hamming_similarity"],
            train_df["label"],
            TARGET_TRAIN_FAR,
        )
        threshold = calibration["threshold"]
        threshold_rows.append(
            {
                "new_seed": new_seed,
                "fold": int(test_fold),
                "target_training_far": TARGET_TRAIN_FAR,
                **calibration,
            }
        )

        # A missing face is an automatic rejection in the operational view.
        test_current_df["accepted"] = (
            test_current_df["current_hamming_similarity"].ge(threshold)
            & test_current_df["matcher_evaluable"]
        ).astype(np.int8)
        test_current_df["threshold"] = threshold
        current_oof_frames.append(test_current_df)

        # Apply the current threshold to every old-key transition without retraining.
        test_revoked_df = revoked_scores_df[
            (revoked_scores_df["new_seed"] == new_seed)
            & (revoked_scores_df["fold"] == test_fold)
        ].copy()
        test_revoked_df["accepted"] = (
            test_revoked_df["revoked_to_current_hamming_similarity"] >= threshold
        ).astype(np.int8)
        test_revoked_df["threshold"] = threshold
        revoked_oof_frames.append(test_revoked_df)

thresholds_df = pd.DataFrame(threshold_rows)
current_oof_df = pd.concat(current_oof_frames, ignore_index=True)
revoked_oof_df = pd.concat(revoked_oof_frames, ignore_index=True)

thresholds_df.to_csv(OUTPUT_DIR / "current_thresholds_by_fold.csv", index=False)
current_oof_df.to_csv(OUTPUT_DIR / "current_version_out_of_fold_decisions.csv", index=False)
revoked_oof_df.to_csv(OUTPUT_DIR / "revoked_template_out_of_fold_decisions.csv", index=False)

print("Threshold rows:", len(thresholds_df))
print("Current-version OOF rows:", len(current_oof_df))
print("Revoked-template OOF rows:", len(revoked_oof_df))
if not (thresholds_df["training_achieved_far"] <= TARGET_TRAIN_FAR).all():
    raise RuntimeError("At least one training fold exceeded the target FAR.")


# Cell 30
def binary_summary(frame):
    labels = frame["label"].to_numpy(dtype=np.int8)
    predictions = frame["accepted"].to_numpy(dtype=np.int8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    genuine = tp + fn
    impostor = tn + fp
    return {
        "evaluated_pairs": genuine + impostor,
        "genuine_pairs": genuine,
        "impostor_pairs": impostor,
        "tar": tp / genuine,
        "far": fp / impostor,
        "frr": fn / genuine,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


current_summary_rows = []
for new_seed, seed_df in current_oof_df.groupby("current_seed", sort=False):
    for evaluation, view_df in [
        ("end_to_end_operational", seed_df),
        ("matcher_only", seed_df[seed_df["matcher_evaluable"]]),
    ]:
        current_summary_rows.append(
            {
                "new_seed": int(new_seed),
                "evaluation": evaluation,
                "mean_threshold": float(view_df["threshold"].mean()),
                **binary_summary(view_df),
            }
        )

current_version_summary_df = pd.DataFrame(current_summary_rows)
current_version_summary_df.to_csv(
    OUTPUT_DIR / "current_version_recognition_summary.csv", index=False
)
display(current_version_summary_df)


# Cell 32
revocation_rows = []
for (old_seed, new_seed), transition_df in revoked_oof_df.groupby(
    ["old_seed", "new_seed"], sort=False
):
    genuine = transition_df[transition_df["label"] == 1]
    impostor = transition_df[transition_df["label"] == 0]
    revocation_rows.append(
        {
            "old_seed": int(old_seed),
            "new_seed": int(new_seed),
            "genuine_replay_attempts": len(genuine),
            "revoked_template_accepts": int(genuine["accepted"].sum()),
            "revoked_template_acceptance_rate": float(genuine["accepted"].mean()),
            "revoked_template_rejection_rate": float(1.0 - genuine["accepted"].mean()),
            "cross_identity_attempts": len(impostor),
            "cross_identity_acceptance_rate": float(impostor["accepted"].mean()),
            "genuine_replay_score_mean": float(
                genuine["revoked_to_current_hamming_similarity"].mean()
            ),
            "genuine_replay_score_std": float(
                genuine["revoked_to_current_hamming_similarity"].std(ddof=1)
            ),
            "mean_current_threshold": float(transition_df["threshold"].mean()),
        }
    )

revocation_summary_df = pd.DataFrame(revocation_rows)
revocation_summary_df.to_csv(
    OUTPUT_DIR / "revocation_matcher_fallback_summary.csv", index=False
)

pooled_genuine_replays = revoked_oof_df[revoked_oof_df["label"] == 1]
pooled_revocation = {
    "genuine_replay_attempts": len(pooled_genuine_replays),
    "revoked_template_accepts": int(pooled_genuine_replays["accepted"].sum()),
    "revoked_template_acceptance_rate": float(pooled_genuine_replays["accepted"].mean()),
    "revoked_template_rejection_rate": float(1.0 - pooled_genuine_replays["accepted"].mean()),
}

with open(OUTPUT_DIR / "pooled_revocation_metrics.json", "w", encoding="utf-8") as file:
    json.dump(pooled_revocation, file, indent=2)

display(revocation_summary_df)
print("Pooled matcher-fallback revocation:")
print(json.dumps(pooled_revocation, indent=2))


# Cell 34
version_gate_rows = []
genuine_pairs_per_transition = int(
    ((evaluable_base["label"] == 1)).sum()
)

for old_seed, new_seed in KEY_TRANSITIONS:
    submitted_version = f"seed-{old_seed}"
    active_version = f"seed-{new_seed}"
    version_matches = submitted_version == active_version
    rejected = not version_matches
    version_gate_rows.append(
        {
            "old_seed": old_seed,
            "new_seed": new_seed,
            "submitted_version": submitted_version,
            "active_version": active_version,
            "genuine_replay_attempts": genuine_pairs_per_transition,
            "rejected_attempts": genuine_pairs_per_transition if rejected else 0,
            "version_gate_rejection_rate": 1.0 if rejected else 0.0,
        }
    )

version_gate_summary_df = pd.DataFrame(version_gate_rows)
version_gate_summary_df.to_csv(OUTPUT_DIR / "version_gate_summary.csv", index=False)

pooled_version_gate_rejection_rate = (
    version_gate_summary_df["rejected_attempts"].sum()
    / version_gate_summary_df["genuine_replay_attempts"].sum()
)

print("Pooled version-gate rejection rate:", pooled_version_gate_rejection_rate)
display(version_gate_summary_df)


# Cell 36
matcher_current = current_version_summary_df[
    current_version_summary_df["evaluation"] == "matcher_only"
].set_index("new_seed")

renewal_rows = []
for old_seed, new_seed in KEY_TRANSITIONS:
    old_tar = float(matcher_current.loc[old_seed, "tar"])
    new_tar = float(matcher_current.loc[new_seed, "tar"])
    renewal_rows.append(
        {
            "old_seed": old_seed,
            "new_seed": new_seed,
            "old_version_tar": old_tar,
            "new_version_tar": new_tar,
            "signed_tar_change": new_tar - old_tar,
            "absolute_tar_change": abs(new_tar - old_tar),
        }
    )

renewal_stability_df = pd.DataFrame(renewal_rows)
renewal_stability_df.to_csv(OUTPUT_DIR / "renewal_tar_stability.csv", index=False)
display(renewal_stability_df)


# Cell 38
worst_current_tar = float(matcher_current["tar"].min())
worst_transition_rtar = float(
    revocation_summary_df["revoked_template_acceptance_rate"].max()
)
worst_absolute_tar_change = float(renewal_stability_df["absolute_tar_change"].max())

guardrail_checks = {
    "current_version_tar_pass": worst_current_tar >= MIN_WORST_CURRENT_VERSION_TAR,
    "matcher_fallback_revocation_pass": (
        worst_transition_rtar <= MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE
    ),
    "renewal_tar_stability_pass": (
        worst_absolute_tar_change <= MAX_WORST_ABSOLUTE_RENEWAL_TAR_CHANGE
    ),
    "version_gate_pass": np.isclose(
        pooled_version_gate_rejection_rate,
        REQUIRED_VERSION_GATE_REJECTION_RATE,
    ).item(),
}

e04_conclusion = (
    "SUPPORTED_WITHIN_E04_GUARDRAILS"
    if all(guardrail_checks.values())
    else "REVOCATION_WEAKNESS_OBSERVED"
)

interpretation_summary = {
    "conclusion": e04_conclusion,
    "worst_current_version_matcher_only_tar": worst_current_tar,
    "pooled_revoked_template_acceptance_rate": pooled_revocation[
        "revoked_template_acceptance_rate"
    ],
    "worst_transition_revoked_template_acceptance_rate": worst_transition_rtar,
    "pooled_version_gate_rejection_rate": float(pooled_version_gate_rejection_rate),
    "worst_absolute_renewal_tar_change": worst_absolute_tar_change,
    "guardrail_checks": guardrail_checks,
    "scope": "One revocation event on LFW, five fixed RO keys, 128-bit templates.",
}

with open(OUTPUT_DIR / "interpretation_summary.json", "w", encoding="utf-8") as file:
    json.dump(interpretation_summary, file, indent=2)

print(json.dumps(interpretation_summary, indent=2))


# Cell 40
key_labels = [str(seed)[-4:] for seed in matcher_current.index]

plt.figure(figsize=(8.5, 4.8))
plt.bar(key_labels, matcher_current["tar"], color="tab:green", alpha=0.85)
plt.axhline(
    MIN_WORST_CURRENT_VERSION_TAR,
    color="tab:red",
    linestyle=":",
    label=f"E04 guardrail = {MIN_WORST_CURRENT_VERSION_TAR:.2f}",
)
plt.ylim(max(0.0, min(0.90, float(matcher_current["tar"].min()) - 0.03)), 1.0)
plt.xlabel("Current key seed suffix")
plt.ylabel("Matcher-only TAR")
plt.title("Recognition after template renewal")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "current_version_tar_by_key.png", dpi=180)
plt.show()

transition_plot_df = revocation_summary_df.copy()
transition_plot_df["transition"] = (
    transition_plot_df["old_seed"].astype(str).str[-4:]
    + "→"
    + transition_plot_df["new_seed"].astype(str).str[-4:]
)

plt.figure(figsize=(12, 5.2))
plt.bar(
    transition_plot_df["transition"],
    transition_plot_df["revoked_template_acceptance_rate"],
    color="tab:purple",
    alpha=0.85,
)
plt.axhline(
    MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE,
    color="tab:red",
    linestyle=":",
    label=f"E04 guardrail = {MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE:.2%}",
)
plt.xlabel("Old key → new key (seed suffixes)")
plt.ylabel("Revoked Template Acceptance Rate")
plt.title("Old-template replay after metadata bypass")
plt.xticks(rotation=55, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "revoked_template_acceptance_by_transition.png", dpi=180)
plt.show()

current_matcher_scores = current_oof_df[current_oof_df["matcher_evaluable"]]
plot_bins = np.arange(-0.5 / TEMPLATE_BITS, 1.0 + 1.5 / TEMPLATE_BITS, 1 / TEMPLATE_BITS)

plt.figure(figsize=(9, 5.4))
plt.hist(
    current_matcher_scores.loc[current_matcher_scores["label"] == 1, "current_hamming_similarity"],
    bins=plot_bins,
    density=True,
    alpha=0.45,
    label="Current genuine",
    color="tab:green",
)
plt.hist(
    current_matcher_scores.loc[current_matcher_scores["label"] == 0, "current_hamming_similarity"],
    bins=plot_bins,
    density=True,
    alpha=0.45,
    label="Current impostor",
    color="tab:blue",
)
plt.hist(
    revoked_oof_df.loc[revoked_oof_df["label"] == 1, "revoked_to_current_hamming_similarity"],
    bins=plot_bins,
    density=True,
    alpha=0.45,
    label="Revoked genuine replay",
    color="tab:orange",
)
plt.xlabel("Normalized Hamming similarity")
plt.ylabel("Density")
plt.title("Current authentication versus revoked-template replay")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "current_vs_revoked_score_distributions.png", dpi=180)
plt.show()


# Cell 42
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
        "attacker_asset": "revoked 128-bit protected template",
        "case_1": "replay with old version metadata",
        "case_2": "remove or forge metadata and submit old bits to current matcher",
        "key_material_exposure": "not tested in E04",
    },
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
        "ordered_key_transitions": [list(item) for item in KEY_TRANSITIONS],
        "template_generation": template_generation_summary,
        "validation": ro_validation,
    },
    "evaluation": {
        "official_folds": 10,
        "threshold_training": "current-version scores from nine folds only",
        "target_training_far": TARGET_TRAIN_FAR,
        "revoked_scores_used_for_calibration": False,
        "score_table_audit": score_table_audit,
    },
    "current_version_recognition": json.loads(
        current_version_summary_df.to_json(orient="records")
    ),
    "matcher_fallback_revocation": json.loads(
        revocation_summary_df.to_json(orient="records")
    ),
    "pooled_matcher_fallback_revocation": pooled_revocation,
    "version_gate": json.loads(version_gate_summary_df.to_json(orient="records")),
    "renewal_stability": json.loads(renewal_stability_df.to_json(orient="records")),
    "interpretation": interpretation_summary,
    "scope_limitations": [
        "The conclusion is empirical and restricted to LFW, five fixed seeded RO keys, and 128-bit templates.",
        "Seeds provide reproducibility but are not production key-management evidence.",
        "The explicit version gate is an operational control and is separate from matcher-level security.",
        "Only one old-template to new-template transition is modelled at a time.",
        "Cumulative leakage from multiple exposed templates is deferred to LFW-E05.",
        "Key exposure, inversion, reconstruction, and impersonation attacks are not tested here.",
        "Official LFW pairs are not an identity-disjoint population study.",
        "The four pass/fail limits are pre-registered exploratory guardrails, not universal standards.",
    ],
}

with open(OUTPUT_DIR / "experiment_record.json", "w", encoding="utf-8") as file:
    json.dump(experiment_record, file, indent=2)


def pct(value):
    return f"{100 * float(value):.3f}%"


mean_current_tar = float(matcher_current["tar"].mean())

brief = f"""# LFW-E04 — One-Page Experiment Brief

**Experiment:** {EXPERIMENT_ID}  
**Researcher:** {RESEARCHER_NAME}  
**Completed (UTC):** {RUN_COMPLETED_UTC}

## Research question

After a protected template is compromised and revoked, does a newly keyed 128-bit
template preserve legitimate recognition while the old template is rejected?

## Controlled design

- Frozen LFW protocol: 6,000 official pairs and 10 folds.
- Frozen 512-D MobileFaceNet embeddings: {extraction_summary['valid_embeddings']} valid of {EXPECTED_UNIQUE_IMAGES} images.
- Five fixed seeded RO keys; all {len(KEY_TRANSITIONS)} ordered old->new transitions.
- Current thresholds learned from nine folds at target training FAR {TARGET_TRAIN_FAR:.0e}.
- Revoked scores never used for threshold calibration.
- Two layers: explicit version gate and matcher fallback after metadata bypass.

## Main results

| Measure | Result | E04 guardrail |
|---|---:|---:|
| Mean current-version matcher-only TAR | {pct(mean_current_tar)} | Descriptive |
| Worst current-version matcher-only TAR | {pct(worst_current_tar)} | >= {pct(MIN_WORST_CURRENT_VERSION_TAR)} |
| Pooled revoked-template acceptance rate | {pct(pooled_revocation['revoked_template_acceptance_rate'])} | Descriptive |
| Worst transition revoked-template acceptance rate | {pct(worst_transition_rtar)} | <= {pct(MAX_WORST_REVOKED_TEMPLATE_ACCEPTANCE_RATE)} |
| Pooled explicit version-gate rejection rate | {pct(pooled_version_gate_rejection_rate)} | = {pct(REQUIRED_VERSION_GATE_REJECTION_RATE)} |
| Worst absolute TAR change after renewal | {pct(worst_absolute_tar_change)} | <= {pct(MAX_WORST_ABSOLUTE_RENEWAL_TAR_CHANGE)} |

## E04 interpretation

**{e04_conclusion}**

This label applies only to the pre-registered E04 guardrails. The explicit version gate
proves the software invalidation rule worked; matcher-fallback RTAR measures old-bit-string
reuse after metadata bypass. Neither result proves unlimited revocability or resistance to
multiple-template compromise.

## Next experiment boundary

LFW-E05 should expose 1, 2, 5, and 10 protected templates for the same identity and test
whether cumulative exposure improves linkage, inference, or impersonation attacks.
"""

(OUTPUT_DIR / "LFW-E04_One-Page_Experiment_Brief.md").write_text(brief, encoding="utf-8")
(OUTPUT_DIR / "RESULTS_README.md").write_text(
    brief + "\n\n## Editable run notes\n\n" + RUN_NOTES + "\n",
    encoding="utf-8",
)

with open(OUTPUT_DIR / "research_log.md", "a", encoding="utf-8") as file:
    file.write(
        f"\n## Completion\n\nRun completed (UTC): {RUN_COMPLETED_UTC}.  \n"
        f"Conclusion: **{e04_conclusion}**.\n"
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
pd.DataFrame(manifest_records).to_csv(OUTPUT_DIR / "result_file_manifest.csv", index=False)

print(brief)
print("Result files before checksum manifest:", len(manifest_records))


# Cell 44
import shutil
from IPython.display import FileLink, display

RESULTS_ARCHIVE_BASE = Path("/kaggle/working/LFW-E04-revocation-results")
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
    "LFW-E04-revocation-results.zip from /kaggle/working."
)


# Cell 46

