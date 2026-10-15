# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 0
import sys
import platform
import importlib.util
from importlib.metadata import version, PackageNotFoundError

print("Python version:", sys.version.split()[0])
print("Operating system:", platform.platform())

packages = [
    "numpy",
    "pandas",
    "opencv-python",
    "Pillow",
    "scikit-learn",
    "matplotlib",
    "torch",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "insightface"
]

print("\n--- PACKAGE AUDIT ---")

for package in packages:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

print("\n--- PYTORCH DEVICE AUDIT ---")

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )
    else:
        print("Selected device: CPU")

except Exception as error:
    print("PyTorch device check failed:", error)

print("\n--- MODULE IMPORT CHECK ---")

modules = ["cv2", "onnxruntime", "insightface"]

for module in modules:
    available = importlib.util.find_spec(module) is not None
    print(f"{module}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")


# Cell 1
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------
# LFW-E02: SEEDED RANDOM ORTHOGONAL PROTECTED TEMPLATES
# ---------------------------------------------------------

EXPERIMENT_ID = "LFW-E02-seeded-RO"
OUTPUT_DIR = Path("/kaggle/working/experiments") / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_BITS = 128

# Five independent, fixed key/seed trials.
RO_SEEDS = [
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
]


def l2_normalize(x, eps=1e-12):
    """Row-wise L2 normalization."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def create_seeded_ro_matrix(
    embedding_dim,
    template_bits,
    seed,
):
    """
    Generate a seeded Gaussian matrix and orthogonalize it using QR.

    Returns a matrix of shape:
        [embedding_dim, template_bits]
    """
    if template_bits > embedding_dim:
        raise ValueError(
            "For this QR construction, template_bits cannot exceed "
            "embedding_dim."
        )

    rng = np.random.default_rng(seed)

    gaussian_matrix = rng.standard_normal(
        size=(embedding_dim, template_bits)
    )

    q, r = np.linalg.qr(gaussian_matrix, mode="reduced")

    # Canonicalize column signs for stronger reproducibility.
    diagonal_signs = np.sign(np.diag(r))
    diagonal_signs[diagonal_signs == 0] = 1.0
    q = q * diagonal_signs

    return q.astype(np.float32)


def generate_biohashes(embeddings, projection_matrix):
    """
    Project normalized embeddings and binarize at zero.

    Output:
        Boolean array [number_of_images, template_bits]
    """
    normalized_embeddings = l2_normalize(embeddings)
    projected_values = normalized_embeddings @ projection_matrix

    return projected_values >= 0.0


def normalized_hamming_similarity(template_a, template_b):
    """
    1.0 means all bits match.
    0.0 means no bits match.
    """
    template_a = np.asarray(template_a, dtype=bool)
    template_b = np.asarray(template_b, dtype=bool)

    return np.mean(template_a == template_b, axis=-1)


# Cell 2
import sys
import subprocess

print("Installing into:", sys.executable)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--upgrade",
    "insightface==1.0.1",
    "onnxruntime-gpu==1.26.0"
])

print("\nINSTALLATION COMPLETED")


# Cell 3
get_ipython().run_line_magic('pip', 'uninstall -y onnxruntime onnxruntime-gpu')

get_ipython().run_line_magic('pip', 'install --quiet --no-cache-dir      onnxruntime-gpu==1.26.0')


# Cell 4
import insightface
import onnxruntime as ort

print("InsightFace:", insightface.__version__)
print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())


# Cell 5
from pathlib import Path
from hashlib import sha256
from insightface.app import FaceAnalysis

model_root = Path("/kaggle/working/insightface_models")

providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

# Load only the two components required for LFW-E01
app = FaceAnalysis(
    name="buffalo_sc",
    root=str(model_root),
    allowed_modules=["detection", "recognition"],
    providers=providers
)

# Freeze detection settings for reproducibility
app.prepare(
    ctx_id=0,
    det_thresh=0.50,
    det_size=(640, 640)
)

model_dir = model_root / "models" / "buffalo_sc"

print("\n--- MODEL PACK ---")
print("Model pack:", "buffalo_sc")
print("Model directory:", model_dir)
print("Directory exists:", model_dir.is_dir())

print("\n--- MODEL FILES ---")

for model_file in sorted(model_dir.glob("*.onnx")):
    digest = sha256(model_file.read_bytes()).hexdigest()

    print("Filename:", model_file.name)
    print("Size:", round(model_file.stat().st_size / (1024 ** 2), 2), "MB")
    print("SHA-256:", digest)
    print()

print("--- LOADED COMPONENTS ---")

for task_name, model in app.models.items():
    print("\nTask:", task_name)
    print("Class:", type(model).__name__)
    print("Model file:", model.model_file)
    print("Providers:", model.session.get_providers())

    print("Inputs:")
    for model_input in model.session.get_inputs():
        print(
            " ",
            model_input.name,
            model_input.shape,
            model_input.type
        )

    print("Outputs:")
    for model_output in model.session.get_outputs():
        print(
            " ",
            model_output.name,
            model_output.shape,
            model_output.type
        )

print("\nMODEL INSPECTION COMPLETE")


# Cell 6
from pathlib import Path
from datetime import datetime
import sys
import insightface
import onnxruntime as ort

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)
output_dir.mkdir(parents=True, exist_ok=True)

config_text = f"""# LFW-E01 frozen experimental configuration

experiment:
  id: "LFW-E01"
  title: "Raw MobileFaceNet verification baseline on LFW"
  created_at: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

dataset:
  name: "Labeled Faces in the Wild"
  variant: "deep-funnelled"
  protocol: "pairs.csv"
  folds: 10
  pairs_per_fold: 600
  genuine_pairs_per_fold: 300
  impostor_pairs_per_fold: 300

face_detection:
  architecture: "SCRFD-500MF"
  model_file: "det_500m.onnx"
  sha256: "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a"
  detection_threshold: 0.50
  detection_size: [640, 640]
  multiple_face_rule: "select face with largest bounding-box area"
  no_face_rule: "record as detection failure"

face_recognition:
  architecture: "MobileFaceNet"
  training_set: "WebFace600K"
  model_file: "w600k_mbf.onnx"
  sha256: "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"
  aligned_input_size: [112, 112]
  embedding_dimension: 512
  embedding_normalization: "L2"
  similarity_measure: "cosine similarity"

evaluation:
  threshold_selection: "learn on nine folds and test on the held-out fold"
  metrics:
    - "AUC"
    - "EER"
    - "accuracy"
    - "FMR"
    - "FNMR"
    - "detection failure rate"
  biohashing_included: false

runtime:
  python: "{sys.version.split()[0]}"
  insightface: "{insightface.__version__}"
  onnxruntime: "{ort.__version__}"
  provider_priority:
    - "CUDAExecutionProvider"
    - "CPUExecutionProvider"
  gpu: "Tesla T4"

reproducibility:
  random_seed: 2026
"""

config_path = output_dir / "config.yaml"
config_path.write_text(config_text, encoding="utf-8")

print("Configuration saved:", config_path)
print("\n--- FROZEN CONFIGURATION ---")
print(config_text)


# Cell 7
import os
from pathlib import Path
from IPython.display import FileLink, display

# FileLink works reliably with paths relative to /kaggle/working
os.chdir("/kaggle/working")

relative_path = Path(
    "experiments/LFW-E01-raw-baseline/config.yaml"
)

print("File exists:", relative_path.is_file())
print("File size:", relative_path.stat().st_size, "bytes")

display(FileLink(relative_path.as_posix()))


# Cell 9
from pathlib import Path
import cv2
import numpy as np

image_root = Path(
    "/kaggle/input/datasets/jessicali9530/lfw-dataset/"
    "lfw-deepfunneled/lfw-deepfunneled"
)

image_paths = [
    image_root / "Abel_Pacheco" / "Abel_Pacheco_0001.jpg",
    image_root / "Abel_Pacheco" / "Abel_Pacheco_0004.jpg"
]

def extract_largest_face(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read: {image_path}")

    faces = app.get(image)

    if not faces:
        return None, 0, None

    # Apply the frozen multiple-face rule
    selected_face = max(
        faces,
        key=lambda face:
            (face.bbox[2] - face.bbox[0]) *
            (face.bbox[3] - face.bbox[1])
    )

    embedding = selected_face.normed_embedding.astype(np.float32)

    return embedding, len(faces), selected_face

results = []

for image_path in image_paths:
    embedding, face_count, face = extract_largest_face(image_path)

    print("\nImage:", image_path.name)
    print("Faces detected:", face_count)

    if embedding is None:
        print("Status: DETECTION FAILED")
    else:
        print("Selected bounding box:", np.round(face.bbox, 2))
        print("Detection score:", round(float(face.det_score), 6))
        print("Embedding shape:", embedding.shape)
        print("Embedding L2 norm:", round(float(np.linalg.norm(embedding)), 6))

    results.append(embedding)

if all(embedding is not None for embedding in results):
    cosine_similarity = float(np.dot(results[0], results[1]))

    print("\n--- GENUINE-PAIR TEST ---")
    print("Pair: Abel_Pacheco_0001 vs Abel_Pacheco_0004")
    print("Cosine similarity:", round(cosine_similarity, 6))
    print("PIPELINE TEST PASSED")
else:
    print("\nPIPELINE TEST FAILED: At least one face was not detected.")


# Cell 11
impostor_paths = [
    image_root
    / "Abdel_Madi_Shabneh"
    / "Abdel_Madi_Shabneh_0001.jpg",

    image_root
    / "Dean_Barker"
    / "Dean_Barker_0001.jpg"
]

impostor_embeddings = []

for image_path in impostor_paths:
    embedding, face_count, face = extract_largest_face(image_path)

    print("\nImage:", image_path.name)
    print("Faces detected:", face_count)

    if embedding is None:
        print("Status: DETECTION FAILED")
    else:
        print("Selected bounding box:", np.round(face.bbox, 2))
        print("Detection score:", round(float(face.det_score), 6))
        print("Embedding shape:", embedding.shape)
        print(
            "Embedding L2 norm:",
            round(float(np.linalg.norm(embedding)), 6)
        )

    impostor_embeddings.append(embedding)

if all(embedding is not None for embedding in impostor_embeddings):
    impostor_similarity = float(
        np.dot(impostor_embeddings[0], impostor_embeddings[1])
    )

    print("\n--- IMPOSTOR-PAIR TEST ---")
    print(
        "Pair: Abdel_Madi_Shabneh_0001 "
        "vs Dean_Barker_0001"
    )
    print(
        "Cosine similarity:",
        round(impostor_similarity, 6)
    )
    print("PIPELINE TEST PASSED")
else:
    print(
        "\nPIPELINE TEST FAILED: "
        "At least one face was not detected."
    )


# Cell 12
from pathlib import Path
import pandas as pd

dataset_root = Path(
    "/kaggle/input/datasets/jessicali9530/lfw-dataset"
)

image_root = (
    dataset_root
    / "lfw-deepfunneled"
    / "lfw-deepfunneled"
)

pairs_path = dataset_root / "pairs.csv"

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Load the previously validated official protocol
pairs_df = pd.read_csv(pairs_path)

def make_image_path(person, image_number):
    image_number = int(float(image_number))

    return (
        image_root
        / str(person)
        / f"{person}_{image_number:04d}.jpg"
    )

manifest_rows = []

for index, row in pairs_df.iterrows():

    is_genuine = pd.isna(row["Unnamed: 3"])

    if is_genuine:
        person_a = str(row["name"])
        person_b = person_a

        image_a = make_image_path(
            person_a,
            row["imagenum1"]
        )
        image_b = make_image_path(
            person_b,
            row["imagenum2"]
        )

        label = 1
        pair_type = "genuine"

    else:
        person_a = str(row["name"])
        person_b = str(row["imagenum2"])

        image_a = make_image_path(
            person_a,
            row["imagenum1"]
        )
        image_b = make_image_path(
            person_b,
            row["Unnamed: 3"]
        )

        label = 0
        pair_type = "impostor"

    manifest_rows.append({
        "pair_id": index + 1,
        "fold": (index // 600) + 1,
        "pair_type": pair_type,
        "label": label,
        "person_a": person_a,
        "person_b": person_b,
        "image_a": str(image_a),
        "image_b": str(image_b)
    })

pair_manifest_df = pd.DataFrame(manifest_rows)

# Collect unique images to avoid extracting the same embedding repeatedly
unique_image_paths = sorted(
    set(pair_manifest_df["image_a"])
    | set(pair_manifest_df["image_b"])
)

missing_images = [
    path for path in unique_image_paths
    if not Path(path).is_file()
]

# Save the E01-specific manifest
manifest_path = output_dir / "pair_manifest.csv"

pair_manifest_df.to_csv(
    manifest_path,
    index=False
)

print("Total protocol pairs:", len(pair_manifest_df))
print(
    "Genuine pairs:",
    (pair_manifest_df["label"] == 1).sum()
)
print(
    "Impostor pairs:",
    (pair_manifest_df["label"] == 0).sum()
)
print("Number of folds:", pair_manifest_df["fold"].nunique())
print("Unique images requiring embeddings:", len(unique_image_paths))
print("Missing unique images:", len(missing_images))
print("Manifest saved:", manifest_path)

if len(missing_images) == 0:
    print(
        "\nMANIFEST PREPARATION PASSED: "
        "All required images are available."
    )
else:
    print("\nMissing-image examples:")
    for path in missing_images[:10]:
        print(path)


# Cell 13
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PILOT_SIZE = 100
RANDOM_SEED = 2026

rng = np.random.default_rng(RANDOM_SEED)

pilot_paths = rng.choice(
    np.asarray(unique_image_paths, dtype=object),
    size=PILOT_SIZE,
    replace=False
).tolist()

# Warm up the GPU; exclude this run from timing
warmup_image = cv2.imread(pilot_paths[0])

if warmup_image is None:
    raise RuntimeError("Could not read the warm-up image.")

_ = app.get(warmup_image)

pilot_rows = []

for index, image_path in enumerate(pilot_paths, start=1):
    start_time = time.perf_counter()

    image = cv2.imread(str(image_path))

    if image is None:
        pilot_rows.append({
            "image_path": str(image_path),
            "status": "read_failure",
            "faces_detected": 0,
            "detection_score": np.nan,
            "embedding_dimension": np.nan,
            "embedding_norm": np.nan,
            "latency_ms": (time.perf_counter() - start_time) * 1000
        })
        continue

    faces = app.get(image)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not faces:
        pilot_rows.append({
            "image_path": str(image_path),
            "status": "detection_failure",
            "faces_detected": 0,
            "detection_score": np.nan,
            "embedding_dimension": np.nan,
            "embedding_norm": np.nan,
            "latency_ms": elapsed_ms
        })
        continue

    # Frozen rule: select the largest detected face
    selected_face = max(
        faces,
        key=lambda face:
            (face.bbox[2] - face.bbox[0]) *
            (face.bbox[3] - face.bbox[1])
    )

    embedding = selected_face.normed_embedding.astype(np.float32)

    embedding_valid = (
        embedding.shape == (512,)
        and np.isfinite(embedding).all()
        and np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)
    )

    pilot_rows.append({
        "image_path": str(image_path),
        "status": "valid" if embedding_valid else "invalid_embedding",
        "faces_detected": len(faces),
        "detection_score": float(selected_face.det_score),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_norm": float(np.linalg.norm(embedding)),
        "latency_ms": elapsed_ms
    })

    if index % 20 == 0:
        print(f"Processed {index}/{PILOT_SIZE} images")

pilot_df = pd.DataFrame(pilot_rows)

valid_count = int((pilot_df["status"] == "valid").sum())
read_failures = int((pilot_df["status"] == "read_failure").sum())
detection_failures = int(
    (pilot_df["status"] == "detection_failure").sum()
)
invalid_embeddings = int(
    (pilot_df["status"] == "invalid_embedding").sum()
)
multiple_face_images = int((pilot_df["faces_detected"] > 1).sum())

mean_latency_ms = float(pilot_df["latency_ms"].mean())
p50_latency_ms = float(pilot_df["latency_ms"].median())
p95_latency_ms = float(
    pilot_df["latency_ms"].quantile(0.95)
)

estimated_minutes = (
    mean_latency_ms * len(unique_image_paths)
    / 1000
    / 60
)

pilot_path = output_dir / "pilot_runtime_100.csv"
pilot_df.to_csv(pilot_path, index=False)

print("\n--- TIMED PILOT SUMMARY ---")
print("Pilot images:", len(pilot_df))
print("Valid embeddings:", valid_count)
print("Image-read failures:", read_failures)
print("Detection failures:", detection_failures)
print("Invalid embeddings:", invalid_embeddings)
print("Images with multiple faces:", multiple_face_images)
print("Mean latency per image:", round(mean_latency_ms, 2), "ms")
print("p50 latency:", round(p50_latency_ms, 2), "ms")
print("p95 latency:", round(p95_latency_ms, 2), "ms")
print(
    "Estimated runtime for all 7,701 images:",
    round(estimated_minutes, 2),
    "minutes"
)
print("Pilot diagnostics saved:", pilot_path)

if valid_count == PILOT_SIZE:
    print("\nPILOT PASSED: All sampled images produced valid embeddings.")
else:
    print(
        "\nPILOT REQUIRES REVIEW:",
        PILOT_SIZE - valid_count,
        "sampled images did not produce valid embeddings."
    )


# Cell 14
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CHECKPOINT_INTERVAL = 500
EMBEDDING_DIMENSION = 512

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

manifest_path = output_dir / "pair_manifest.csv"
embeddings_path = output_dir / "raw_embeddings.npy"
metadata_path = output_dir / "embedding_metadata.csv"

# Reconstruct the deterministic unique-image list
pair_manifest_df = pd.read_csv(manifest_path)

unique_image_paths = sorted(
    set(pair_manifest_df["image_a"])
    | set(pair_manifest_df["image_b"])
)

total_images = len(unique_image_paths)

if total_images != 7701:
    raise RuntimeError(
        f"Expected 7,701 unique images, found {total_images}."
    )

# ---------------------------------------------------------
# Create or restore extraction state
# ---------------------------------------------------------
if embeddings_path.exists() and metadata_path.exists():
    print("Existing checkpoint found. Resuming extraction.")

    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)

    if embeddings.shape != (
        total_images,
        EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if metadata_df["image_path"].tolist() != unique_image_paths:
        raise RuntimeError(
            "Checkpoint image order does not match the manifest."
        )

else:
    print("No checkpoint found. Starting new extraction.")

    embeddings = np.full(
        (total_images, EMBEDDING_DIMENSION),
        np.nan,
        dtype=np.float32
    )

    metadata_df = pd.DataFrame({
        "image_id": np.arange(total_images),
        "image_path": unique_image_paths,
        "identity": [
            Path(path).parent.name
            for path in unique_image_paths
        ],
        "status": "pending",
        "faces_detected": 0,
        "detection_score": np.nan,
        "bbox_x1": np.nan,
        "bbox_y1": np.nan,
        "bbox_x2": np.nan,
        "bbox_y2": np.nan,
        "embedding_norm": np.nan,
        "latency_ms": np.nan
    })

def save_checkpoint():
    """Atomically save embeddings and metadata."""

    temporary_embeddings = (
        output_dir / "raw_embeddings.tmp"
    )
    temporary_metadata = (
        output_dir / "embedding_metadata.tmp"
    )

    with open(temporary_embeddings, "wb") as file:
        np.save(file, embeddings)

    os.replace(
        temporary_embeddings,
        embeddings_path
    )

    metadata_df.to_csv(
        temporary_metadata,
        index=False
    )

    os.replace(
        temporary_metadata,
        metadata_path
    )

# Only pending rows are processed during a resumed run
pending_indices = metadata_df.index[
    metadata_df["status"] == "pending"
].tolist()

print("Total images:", total_images)
print("Already completed:", total_images - len(pending_indices))
print("Remaining:", len(pending_indices))

batch_start = time.perf_counter()

for progress, row_index in enumerate(
    pending_indices,
    start=1
):
    image_path = metadata_df.at[
        row_index,
        "image_path"
    ]

    image_start = time.perf_counter()
    image = cv2.imread(image_path)

    if image is None:
        metadata_df.at[
            row_index,
            "status"
        ] = "read_failure"

        metadata_df.at[
            row_index,
            "latency_ms"
        ] = (
            time.perf_counter() - image_start
        ) * 1000

    else:
        faces = app.get(image)

        metadata_df.at[
            row_index,
            "faces_detected"
        ] = len(faces)

        if not faces:
            metadata_df.at[
                row_index,
                "status"
            ] = "detection_failure"

        else:
            # Frozen selection rule
            selected_face = max(
                faces,
                key=lambda face:
                    (face.bbox[2] - face.bbox[0])
                    * (face.bbox[3] - face.bbox[1])
            )

            embedding = (
                selected_face.normed_embedding
                .astype(np.float32)
            )

            embedding_norm = float(
                np.linalg.norm(embedding)
            )

            embedding_valid = (
                embedding.shape
                == (EMBEDDING_DIMENSION,)
                and np.isfinite(embedding).all()
                and np.isclose(
                    embedding_norm,
                    1.0,
                    atol=1e-4
                )
            )

            if embedding_valid:
                embeddings[row_index] = embedding
                status = "valid"
            else:
                status = "invalid_embedding"

            bbox = selected_face.bbox.astype(float)

            metadata_df.at[
                row_index,
                "status"
            ] = status

            metadata_df.at[
                row_index,
                "detection_score"
            ] = float(selected_face.det_score)

            metadata_df.loc[
                row_index,
                [
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2"
                ]
            ] = bbox

            metadata_df.at[
                row_index,
                "embedding_norm"
            ] = embedding_norm

        metadata_df.at[
            row_index,
            "latency_ms"
        ] = (
            time.perf_counter() - image_start
        ) * 1000

    if (
        progress % CHECKPOINT_INTERVAL == 0
        or progress == len(pending_indices)
    ):
        save_checkpoint()

        print(
            f"Checkpoint: {progress}/"
            f"{len(pending_indices)} remaining images processed"
        )

# Ensure files are also created when nothing was pending
save_checkpoint()

elapsed_minutes = (
    time.perf_counter() - batch_start
) / 60

# ---------------------------------------------------------
# Final validation
# ---------------------------------------------------------
status_counts = (
    metadata_df["status"]
    .value_counts()
    .to_dict()
)

valid_count = int(
    (metadata_df["status"] == "valid").sum()
)

multiple_face_count = int(
    (metadata_df["faces_detected"] > 1).sum()
)

finite_embedding_rows = int(
    np.isfinite(embeddings).all(axis=1).sum()
)

print("\n--- COMPLETE EXTRACTION SUMMARY ---")
print("Total unique images:", total_images)
print("Valid embeddings:", valid_count)
print(
    "Read failures:",
    status_counts.get("read_failure", 0)
)
print(
    "Detection failures:",
    status_counts.get("detection_failure", 0)
)
print(
    "Invalid embeddings:",
    status_counts.get("invalid_embedding", 0)
)
print(
    "Images with multiple faces:",
    multiple_face_count
)
print(
    "Finite embedding rows:",
    finite_embedding_rows
)
print(
    "Embedding matrix shape:",
    embeddings.shape
)
print(
    "Extraction time:",
    round(elapsed_minutes, 2),
    "minutes"
)
print("Embeddings saved:", embeddings_path)
print("Metadata saved:", metadata_path)

if (
    valid_count == total_images
    and finite_embedding_rows == total_images
):
    print(
        "\nFULL EXTRACTION PASSED: "
        "All images produced valid embeddings."
    )
else:
    print(
        "\nFULL EXTRACTION REQUIRES REVIEW: "
        "At least one image did not produce a valid embedding."
    )


# Cell 15
import numpy as np

print("--- POSSIBLE EMBEDDING VARIABLES ---")

keywords = ("embed", "cache", "feature", "record")

for name, value in globals().copy().items():
    if any(keyword in name.lower() for keyword in keywords):
        try:
            shape = np.asarray(value).shape
        except Exception:
            shape = "not array-like"

        print(
            f"{name:35s} "
            f"type={type(value).__name__:15s} "
            f"shape={shape}"
        )


# Cell 16
# ---------------------------------------------------------
# E02: GENERATE AND VALIDATE 128-BIT RO BIOHASHES
# ---------------------------------------------------------

X = np.asarray(embeddings, dtype=np.float32)

if X.ndim != 2:
    raise ValueError(
        f"Expected a 2-D embedding array, obtained shape {X.shape}"
    )

number_of_embeddings, embedding_dim = X.shape

if embedding_dim != 512:
    raise ValueError(
        f"Expected the frozen E01 512-D embeddings, "
        f"but obtained {embedding_dim}-D embeddings."
    )

if not np.isfinite(X).all():
    raise ValueError("The embedding array contains NaN or infinite values.")

print("Frozen E01 embedding array:", X.shape)
print("E02 protected-template length:", TEMPLATE_BITS, "bits")

validation_results = {}
biohashes_by_seed = {}

for seed in RO_SEEDS:
    projection = create_seeded_ro_matrix(
        embedding_dim=embedding_dim,       # 512
        template_bits=TEMPLATE_BITS,       # 128
        seed=seed,
    )

    biohashes = generate_biohashes(
        embeddings=X,
        projection_matrix=projection,
    )

    biohashes_by_seed[seed] = biohashes

    identity_matrix = np.eye(
        TEMPLATE_BITS,
        dtype=np.float32,
    )

    orthogonality_error = np.max(
        np.abs(
            projection.T @ projection
            - identity_matrix
        )
    )

    bit_one_rates = biohashes.mean(axis=0)

    validation_results[str(seed)] = {
        "embedding_shape": list(X.shape),
        "projection_shape": list(projection.shape),
        "biohash_shape": list(biohashes.shape),
        "orthogonality_max_error": float(orthogonality_error),
        "overall_one_rate": float(biohashes.mean()),
        "minimum_bit_one_rate": float(bit_one_rates.min()),
        "maximum_bit_one_rate": float(bit_one_rates.max()),
    }

    np.save(
        OUTPUT_DIR / f"ro_projection_seed_{seed}.npy",
        projection,
    )

    np.save(
        OUTPUT_DIR / f"biohashes_seed_{seed}.npy",
        biohashes,
    )

print(json.dumps(validation_results, indent=2))

with open(OUTPUT_DIR / "ro_validation.json", "w") as file:
    json.dump(validation_results, file, indent=2)

print("\nSaved E02 validation files to:")
print(OUTPUT_DIR)


# Cell 17
from pathlib import Path
import pandas as pd

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

metadata_df = pd.read_csv(
    output_dir / "embedding_metadata.csv"
)

pair_manifest_df = pd.read_csv(
    output_dir / "pair_manifest.csv"
)

# Identify every unsuccessful image
failed_df = metadata_df[
    metadata_df["status"] != "valid"
].copy()

failed_paths = set(failed_df["image_path"].astype(str))

# Count how many official pairs use each failed image
image_usage_counts = pd.concat([
    pair_manifest_df["image_a"],
    pair_manifest_df["image_b"]
]).value_counts()

failed_df["pair_usage_count"] = (
    failed_df["image_path"]
    .map(image_usage_counts)
    .fillna(0)
    .astype(int)
)

failed_df["filename"] = (
    failed_df["image_path"]
    .map(lambda path: Path(path).name)
)

# Find every official pair affected by at least one failure
affected_mask = (
    pair_manifest_df["image_a"].isin(failed_paths)
    | pair_manifest_df["image_b"].isin(failed_paths)
)

affected_pairs_df = pair_manifest_df[
    affected_mask
].copy()

affected_pairs_df["failed_image_a"] = (
    affected_pairs_df["image_a"].isin(failed_paths)
)

affected_pairs_df["failed_image_b"] = (
    affected_pairs_df["image_b"].isin(failed_paths)
)

affected_pairs_df["failed_image_count"] = (
    affected_pairs_df[
        ["failed_image_a", "failed_image_b"]
    ].sum(axis=1)
)

# Save permanent audit records
failed_images_path = output_dir / "detection_failures.csv"
affected_pairs_path = output_dir / "affected_pairs.csv"

failed_df.to_csv(failed_images_path, index=False)
affected_pairs_df.to_csv(affected_pairs_path, index=False)

total_pairs = len(pair_manifest_df)
affected_count = len(affected_pairs_df)

print("--- DETECTION-FAILURE AUDIT ---")
print("Failed unique images:", len(failed_df))
print(
    "Detection failures:",
    (failed_df["status"] == "detection_failure").sum()
)
print("Affected official pairs:", affected_count)
print("Unaffected evaluable pairs:", total_pairs - affected_count)
print(
    "Affected genuine pairs:",
    (affected_pairs_df["label"] == 1).sum()
)
print(
    "Affected impostor pairs:",
    (affected_pairs_df["label"] == 0).sum()
)
print(
    "Folds containing affected pairs:",
    affected_pairs_df["fold"].nunique()
)
print(
    "Affected-pair percentage:",
    round(100 * affected_count / total_pairs, 4),
    "%"
)
print("Failure audit saved:", failed_images_path)
print("Affected pairs saved:", affected_pairs_path)

print("\n--- FAILED IMAGE LIST ---")
print(
    failed_df[
        [
            "filename",
            "identity",
            "status",
            "pair_usage_count"
        ]
    ].to_string(index=False)
)


# Cell 18
from pathlib import Path
import numpy as np
import pandas as pd

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

manifest_df = pd.read_csv(
    output_dir / "pair_manifest.csv"
)

metadata_df = pd.read_csv(
    output_dir / "embedding_metadata.csv"
)

embeddings = np.load(
    output_dir / "raw_embeddings.npy"
)

# A score slightly below the theoretical minimum cosine similarity.
# It guarantees rejection under every valid cosine threshold.
FAILURE_REJECT_SCORE = -1.000001

# Confirm that metadata rows still correspond to embedding rows.
if len(metadata_df) != embeddings.shape[0]:
    raise RuntimeError(
        "Metadata and embedding row counts do not match."
    )

if embeddings.shape != (7701, 512):
    raise RuntimeError(
        f"Unexpected embedding shape: {embeddings.shape}"
    )

path_to_id = dict(
    zip(
        metadata_df["image_path"],
        metadata_df["image_id"]
    )
)

path_to_status = dict(
    zip(
        metadata_df["image_path"],
        metadata_df["status"]
    )
)

pair_rows = []

for _, pair in manifest_df.iterrows():
    image_a = pair["image_a"]
    image_b = pair["image_b"]

    image_a_id = int(path_to_id[image_a])
    image_b_id = int(path_to_id[image_b])

    status_a = path_to_status[image_a]
    status_b = path_to_status[image_b]

    matcher_evaluable = (
        status_a == "valid"
        and status_b == "valid"
    )

    if matcher_evaluable:
        matcher_similarity = float(
            np.dot(
                embeddings[image_a_id],
                embeddings[image_b_id]
            )
        )

        operational_score = matcher_similarity
        pair_status = "evaluable"

    else:
        # No comparison is possible, so the operational
        # authentication system rejects the attempt.
        matcher_similarity = np.nan
        operational_score = FAILURE_REJECT_SCORE
        pair_status = "detection_failure_reject"

    pair_rows.append({
        "pair_id": int(pair["pair_id"]),
        "fold": int(pair["fold"]),
        "pair_type": pair["pair_type"],
        "label": int(pair["label"]),
        "image_a": image_a,
        "image_b": image_b,
        "status_a": status_a,
        "status_b": status_b,
        "pair_status": pair_status,
        "matcher_evaluable": matcher_evaluable,
        "matcher_similarity": matcher_similarity,
        "operational_score": operational_score
    })

pair_scores_df = pd.DataFrame(pair_rows)

pair_scores_path = output_dir / "pair_scores.csv"

pair_scores_df.to_csv(
    pair_scores_path,
    index=False
)

# Validation
evaluable_count = int(
    pair_scores_df["matcher_evaluable"].sum()
)

failed_pair_count = int(
    (~pair_scores_df["matcher_evaluable"]).sum()
)

failed_genuine_count = int(
    (
        (~pair_scores_df["matcher_evaluable"])
        & (pair_scores_df["label"] == 1)
    ).sum()
)

failed_impostor_count = int(
    (
        (~pair_scores_df["matcher_evaluable"])
        & (pair_scores_df["label"] == 0)
    ).sum()
)

finite_matcher_scores = int(
    np.isfinite(
        pair_scores_df["matcher_similarity"]
    ).sum()
)

finite_operational_scores = int(
    np.isfinite(
        pair_scores_df["operational_score"]
    ).sum()
)

print("--- PAIR-SCORE PREPARATION SUMMARY ---")
print("Total official pairs:", len(pair_scores_df))
print("Matcher-evaluable pairs:", evaluable_count)
print("Detection-failure pairs:", failed_pair_count)
print("Failed genuine pairs:", failed_genuine_count)
print("Failed impostor pairs:", failed_impostor_count)
print("Finite matcher scores:", finite_matcher_scores)
print("Finite operational scores:", finite_operational_scores)
print("Number of folds:", pair_scores_df["fold"].nunique())
print("Pair scores saved:", pair_scores_path)

if (
    len(pair_scores_df) == 6000
    and evaluable_count == 5965
    and failed_pair_count == 35
    and failed_genuine_count == 19
    and failed_impostor_count == 16
    and finite_matcher_scores == 5965
    and finite_operational_scores == 6000
):
    print(
        "\nPAIR-SCORE PREPARATION PASSED: "
        "Failure handling is consistent with the audit."
    )
else:
    print(
        "\nPAIR-SCORE PREPARATION REQUIRES REVIEW."
    )


# Cell 20
from pathlib import Path

import numpy as np
import pandas as pd

output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

pair_scores_df = pd.read_csv(
    output_dir / "pair_scores.csv"
)

# Convert safely in case CSV loading treats booleans as strings.
if pair_scores_df["matcher_evaluable"].dtype != bool:
    pair_scores_df["matcher_evaluable"] = (
        pair_scores_df["matcher_evaluable"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

if pair_scores_df["matcher_evaluable"].isna().any():
    raise RuntimeError(
        "Could not interpret matcher_evaluable values."
    )

# Frozen cosine-threshold search grid.
# Resolution = 0.001.
THRESHOLD_GRID = np.arange(
    -1.0,
    1.0001,
    0.001,
    dtype=np.float64
)


def select_accuracy_threshold(scores, labels):
    """
    Select the threshold giving the highest training accuracy.

    Decision rule:
        score >= threshold -> genuine
        score < threshold  -> impostor

    If several thresholds tie, use their mean.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    if len(scores) == 0:
        raise ValueError("Threshold-selection set is empty.")

    if not np.isfinite(scores).all():
        raise ValueError(
            "Threshold-selection scores contain NaN or infinity."
        )

    predictions = (
        scores[:, None]
        >= THRESHOLD_GRID[None, :]
    )

    accuracies = (
        predictions
        == labels[:, None]
    ).mean(axis=0)

    maximum_accuracy = float(accuracies.max())

    best_indices = np.flatnonzero(
        np.isclose(
            accuracies,
            maximum_accuracy,
            rtol=0,
            atol=1e-12
        )
    )

    selected_threshold = float(
        THRESHOLD_GRID[best_indices].mean()
    )

    return selected_threshold, maximum_accuracy


def evaluate_at_threshold(scores, labels, threshold):
    """Calculate threshold-dependent verification results."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    true_positive = int(
        ((predictions == 1) & (labels == 1)).sum()
    )
    true_negative = int(
        ((predictions == 0) & (labels == 0)).sum()
    )
    false_positive = int(
        ((predictions == 1) & (labels == 0)).sum()
    )
    false_negative = int(
        ((predictions == 0) & (labels == 1)).sum()
    )

    genuine_count = true_positive + false_negative
    impostor_count = true_negative + false_positive

    accuracy = (
        true_positive + true_negative
    ) / len(labels)

    tar = (
        true_positive / genuine_count
        if genuine_count else np.nan
    )

    far = (
        false_positive / impostor_count
        if impostor_count else np.nan
    )

    frr = (
        false_negative / genuine_count
        if genuine_count else np.nan
    )

    return {
        "pairs": len(labels),
        "genuine_pairs": genuine_count,
        "impostor_pairs": impostor_count,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "accuracy": accuracy,
        "tar": tar,
        "far": far,
        "frr": frr,
        "predictions": predictions
    }


# Store the decision made for every official pair.
oof_decisions_df = pair_scores_df[
    [
        "pair_id",
        "fold",
        "pair_type",
        "label",
        "pair_status",
        "matcher_evaluable",
        "matcher_similarity",
        "operational_score"
    ]
].copy()

oof_decisions_df["operational_threshold"] = np.nan
oof_decisions_df["operational_prediction"] = np.nan
oof_decisions_df["matcher_threshold"] = np.nan
oof_decisions_df["matcher_prediction"] = np.nan

fold_rows = []

for test_fold in sorted(pair_scores_df["fold"].unique()):

    train_mask = (
        pair_scores_df["fold"] != test_fold
    )
    test_mask = (
        pair_scores_df["fold"] == test_fold
    )

    # =========================================================
    # 1. PRIMARY OPERATIONAL EVALUATION
    # Includes detection-failure rejection scores.
    # =========================================================
    operational_train = pair_scores_df[train_mask]
    operational_test = pair_scores_df[test_mask]

    operational_threshold, operational_train_accuracy = (
        select_accuracy_threshold(
            operational_train["operational_score"],
            operational_train["label"]
        )
    )

    operational_metrics = evaluate_at_threshold(
        operational_test["operational_score"],
        operational_test["label"],
        operational_threshold
    )

    oof_decisions_df.loc[
        test_mask,
        "operational_threshold"
    ] = operational_threshold

    oof_decisions_df.loc[
        test_mask,
        "operational_prediction"
    ] = operational_metrics["predictions"]

    # =========================================================
    # 2. SECONDARY MATCHER-ONLY EVALUATION
    # Uses only pairs with two valid embeddings.
    # =========================================================
    matcher_train_mask = (
        train_mask
        & pair_scores_df["matcher_evaluable"]
    )

    matcher_test_mask = (
        test_mask
        & pair_scores_df["matcher_evaluable"]
    )

    matcher_train = pair_scores_df[
        matcher_train_mask
    ]

    matcher_test = pair_scores_df[
        matcher_test_mask
    ]

    matcher_threshold, matcher_train_accuracy = (
        select_accuracy_threshold(
            matcher_train["matcher_similarity"],
            matcher_train["label"]
        )
    )

    matcher_metrics = evaluate_at_threshold(
        matcher_test["matcher_similarity"],
        matcher_test["label"],
        matcher_threshold
    )

    oof_decisions_df.loc[
        matcher_test_mask,
        "matcher_threshold"
    ] = matcher_threshold

    oof_decisions_df.loc[
        matcher_test_mask,
        "matcher_prediction"
    ] = matcher_metrics["predictions"]

    fold_rows.append({
        "fold": int(test_fold),

        "operational_threshold":
            operational_threshold,
        "operational_train_accuracy":
            operational_train_accuracy,
        "operational_test_pairs":
            operational_metrics["pairs"],
        "operational_tp":
            operational_metrics["tp"],
        "operational_tn":
            operational_metrics["tn"],
        "operational_fp":
            operational_metrics["fp"],
        "operational_fn":
            operational_metrics["fn"],
        "operational_accuracy":
            operational_metrics["accuracy"],
        "operational_tar":
            operational_metrics["tar"],
        "operational_far":
            operational_metrics["far"],
        "operational_frr":
            operational_metrics["frr"],

        "matcher_threshold":
            matcher_threshold,
        "matcher_train_accuracy":
            matcher_train_accuracy,
        "matcher_test_pairs":
            matcher_metrics["pairs"],
        "matcher_tp":
            matcher_metrics["tp"],
        "matcher_tn":
            matcher_metrics["tn"],
        "matcher_fp":
            matcher_metrics["fp"],
        "matcher_fn":
            matcher_metrics["fn"],
        "matcher_accuracy":
            matcher_metrics["accuracy"],
        "matcher_tar":
            matcher_metrics["tar"],
        "matcher_far":
            matcher_metrics["far"],
        "matcher_frr":
            matcher_metrics["frr"]
    })


fold_results_df = pd.DataFrame(fold_rows)


def create_summary(prefix, evaluation_name):
    """Create fold-level and pooled summaries."""
    tp = int(fold_results_df[f"{prefix}_tp"].sum())
    tn = int(fold_results_df[f"{prefix}_tn"].sum())
    fp = int(fold_results_df[f"{prefix}_fp"].sum())
    fn = int(fold_results_df[f"{prefix}_fn"].sum())

    total_pairs = tp + tn + fp + fn
    genuine_pairs = tp + fn
    impostor_pairs = tn + fp

    return {
        "evaluation": evaluation_name,
        "evaluated_pairs": total_pairs,
        "mean_threshold": fold_results_df[
            f"{prefix}_threshold"
        ].mean(),
        "threshold_std": fold_results_df[
            f"{prefix}_threshold"
        ].std(ddof=1),
        "mean_fold_accuracy": fold_results_df[
            f"{prefix}_accuracy"
        ].mean(),
        "fold_accuracy_std": fold_results_df[
            f"{prefix}_accuracy"
        ].std(ddof=1),
        "pooled_accuracy": (tp + tn) / total_pairs,
        "pooled_tar": tp / genuine_pairs,
        "pooled_far": fp / impostor_pairs,
        "pooled_frr": fn / genuine_pairs,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


summary_df = pd.DataFrame([
    create_summary(
        "operational",
        "end_to_end_operational"
    ),
    create_summary(
        "matcher",
        "matcher_only"
    )
])

# Save reproducible records.
fold_results_path = (
    output_dir / "cross_validated_fold_results.csv"
)

summary_path = (
    output_dir / "cross_validated_summary.csv"
)

oof_decisions_path = (
    output_dir / "out_of_fold_decisions.csv"
)

fold_results_df.to_csv(
    fold_results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

oof_decisions_df.to_csv(
    oof_decisions_path,
    index=False
)

operational_summary = summary_df.iloc[0]
matcher_summary = summary_df.iloc[1]

print("--- LEAKAGE-FREE 10-FOLD CV SUMMARY ---")

print("\nPRIMARY: END-TO-END OPERATIONAL")
print(
    "Evaluated pairs:",
    int(operational_summary["evaluated_pairs"])
)
print(
    "Mean threshold:",
    round(operational_summary["mean_threshold"], 6)
)
print(
    "Threshold standard deviation:",
    round(operational_summary["threshold_std"], 6)
)
print(
    "Mean fold accuracy:",
    round(
        100 * operational_summary["mean_fold_accuracy"],
        4
    ),
    "%"
)
print(
    "Fold accuracy standard deviation:",
    round(
        100 * operational_summary["fold_accuracy_std"],
        4
    ),
    "%"
)
print(
    "Pooled accuracy:",
    round(
        100 * operational_summary["pooled_accuracy"],
        4
    ),
    "%"
)
print(
    "Pooled TAR:",
    round(100 * operational_summary["pooled_tar"], 4),
    "%"
)
print(
    "Pooled FAR:",
    round(100 * operational_summary["pooled_far"], 4),
    "%"
)
print(
    "Pooled FRR:",
    round(100 * operational_summary["pooled_frr"], 4),
    "%"
)

print("\nSECONDARY: MATCHER ONLY")
print(
    "Evaluated pairs:",
    int(matcher_summary["evaluated_pairs"])
)
print(
    "Mean threshold:",
    round(matcher_summary["mean_threshold"], 6)
)
print(
    "Threshold standard deviation:",
    round(matcher_summary["threshold_std"], 6)
)
print(
    "Mean fold accuracy:",
    round(
        100 * matcher_summary["mean_fold_accuracy"],
        4
    ),
    "%"
)
print(
    "Fold accuracy standard deviation:",
    round(
        100 * matcher_summary["fold_accuracy_std"],
        4
    ),
    "%"
)
print(
    "Pooled accuracy:",
    round(
        100 * matcher_summary["pooled_accuracy"],
        4
    ),
    "%"
)
print(
    "Pooled TAR:",
    round(100 * matcher_summary["pooled_tar"], 4),
    "%"
)
print(
    "Pooled FAR:",
    round(100 * matcher_summary["pooled_far"], 4),
    "%"
)
print(
    "Pooled FRR:",
    round(100 * matcher_summary["pooled_frr"], 4),
    "%"
)

print("\nFold results saved:", fold_results_path)
print("Summary saved:", summary_path)
print("Out-of-fold decisions saved:", oof_decisions_path)

validation_passed = (
    len(fold_results_df) == 10
    and int(
        fold_results_df["operational_test_pairs"].sum()
    ) == 6000
    and int(
        fold_results_df["matcher_test_pairs"].sum()
    ) == 5965
    and oof_decisions_df[
        "operational_prediction"
    ].notna().sum() == 6000
    and oof_decisions_df[
        "matcher_prediction"
    ].notna().sum() == 5965
    and np.isfinite(
        fold_results_df[
            [
                "operational_threshold",
                "matcher_threshold",
                "operational_accuracy",
                "matcher_accuracy"
            ]
        ].to_numpy()
    ).all()
)

if validation_passed:
    print(
        "\n10-FOLD EVALUATION PASSED: "
        "All held-out decisions were generated without "
        "test-fold threshold leakage."
    )
else:
    print(
        "\n10-FOLD EVALUATION REQUIRES REVIEW."
    )


# Cell 21
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

pair_scores_df = pd.read_csv(
    output_dir / "pair_scores.csv"
)

# Safely restore the Boolean column after CSV loading.
if pair_scores_df["matcher_evaluable"].dtype != bool:
    pair_scores_df["matcher_evaluable"] = (
        pair_scores_df["matcher_evaluable"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

if pair_scores_df["matcher_evaluable"].isna().any():
    raise RuntimeError(
        "Could not interpret matcher_evaluable values."
    )


def calculate_auc_eer(labels, scores):
    """
    Calculate ROC AUC and linearly interpolated EER.

    EER is a descriptive ROC metric. Its corresponding threshold
    must not replace the leakage-free operational thresholds.
    """
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) == 0:
        raise ValueError("Evaluation data is empty.")

    if not np.isfinite(scores).all():
        raise ValueError(
            "Scores contain NaN or infinity."
        )

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "Both genuine and impostor labels are required."
        )

    auc_value = float(
        roc_auc_score(labels, scores)
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
        drop_intermediate=False
    )

    fnr = 1.0 - tpr
    difference = fpr - fnr

    # Find adjacent ROC points surrounding the EER crossing.
    crossing_indices = np.where(
        difference[:-1] * difference[1:] <= 0
    )[0]

    if len(crossing_indices) > 0:
        # Use the crossing closest to equality.
        crossing_index = min(
            crossing_indices,
            key=lambda index: (
                abs(difference[index])
                + abs(difference[index + 1])
            )
        )

        difference_start = difference[crossing_index]
        difference_end = difference[crossing_index + 1]

        denominator = (
            difference_end - difference_start
        )

        if denominator == 0:
            interpolation_weight = 0.5
        else:
            interpolation_weight = (
                -difference_start / denominator
            )

        eer_value = (
            fpr[crossing_index]
            + interpolation_weight
            * (
                fpr[crossing_index + 1]
                - fpr[crossing_index]
            )
        )

        threshold_start = thresholds[crossing_index]
        threshold_end = thresholds[crossing_index + 1]

        if (
            np.isfinite(threshold_start)
            and np.isfinite(threshold_end)
        ):
            eer_threshold = (
                threshold_start
                + interpolation_weight
                * (threshold_end - threshold_start)
            )
        else:
            closest_index = int(
                np.argmin(np.abs(difference))
            )
            eer_threshold = thresholds[closest_index]

    else:
        # Discrete fallback: closest empirical ROC point.
        closest_index = int(
            np.argmin(np.abs(difference))
        )

        eer_value = (
            fpr[closest_index]
            + fnr[closest_index]
        ) / 2.0

        eer_threshold = thresholds[closest_index]

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    return {
        "pairs": len(labels),
        "genuine_pairs": len(genuine_scores),
        "impostor_pairs": len(impostor_scores),
        "auc": auc_value,
        "eer": float(eer_value),
        "eer_threshold": float(eer_threshold),
        "genuine_mean": float(genuine_scores.mean()),
        "genuine_std": float(
            genuine_scores.std(ddof=1)
        ),
        "impostor_mean": float(impostor_scores.mean()),
        "impostor_std": float(
            impostor_scores.std(ddof=1)
        )
    }


evaluation_views = {
    "end_to_end_operational": {
        "data": pair_scores_df.copy(),
        "score_column": "operational_score"
    },
    "matcher_only": {
        "data": pair_scores_df[
            pair_scores_df["matcher_evaluable"]
        ].copy(),
        "score_column": "matcher_similarity"
    }
}

fold_rows = []
summary_rows = []

for evaluation_name, specification in evaluation_views.items():

    evaluation_df = specification["data"]
    score_column = specification["score_column"]

    # Pooled ROC-derived metrics.
    pooled_metrics = calculate_auc_eer(
        evaluation_df["label"],
        evaluation_df[score_column]
    )

    view_fold_rows = []

    # Independently describe each official fold.
    for fold in sorted(evaluation_df["fold"].unique()):

        fold_df = evaluation_df[
            evaluation_df["fold"] == fold
        ]

        fold_metrics = calculate_auc_eer(
            fold_df["label"],
            fold_df[score_column]
        )

        fold_row = {
            "evaluation": evaluation_name,
            "fold": int(fold),
            **fold_metrics
        }

        fold_rows.append(fold_row)
        view_fold_rows.append(fold_row)

    view_fold_df = pd.DataFrame(view_fold_rows)

    summary_rows.append({
        "evaluation": evaluation_name,
        **pooled_metrics,
        "mean_fold_auc":
            view_fold_df["auc"].mean(),
        "fold_auc_std":
            view_fold_df["auc"].std(ddof=1),
        "mean_fold_eer":
            view_fold_df["eer"].mean(),
        "fold_eer_std":
            view_fold_df["eer"].std(ddof=1)
    })


fold_metrics_df = pd.DataFrame(fold_rows)
summary_df = pd.DataFrame(summary_rows)

fold_metrics_path = (
    output_dir / "roc_fold_metrics.csv"
)

summary_path = (
    output_dir / "roc_summary.csv"
)

fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("--- ROC, AUC AND EER SUMMARY ---")

for _, result in summary_df.iterrows():

    print(f"\n{result['evaluation'].upper()}")
    print("Evaluated pairs:", int(result["pairs"]))
    print(
        "Pooled AUC:",
        round(100 * result["auc"], 4),
        "%"
    )
    print(
        "Mean fold AUC:",
        round(100 * result["mean_fold_auc"], 4),
        "%"
    )
    print(
        "Fold AUC standard deviation:",
        round(100 * result["fold_auc_std"], 4),
        "%"
    )
    print(
        "Pooled EER:",
        round(100 * result["eer"], 4),
        "%"
    )
    print(
        "EER threshold:",
        round(result["eer_threshold"], 6)
    )
    print(
        "Mean fold EER:",
        round(100 * result["mean_fold_eer"], 4),
        "%"
    )
    print(
        "Fold EER standard deviation:",
        round(100 * result["fold_eer_std"], 4),
        "%"
    )
    print(
        "Genuine score:",
        round(result["genuine_mean"], 6),
        "±",
        round(result["genuine_std"], 6)
    )
    print(
        "Impostor score:",
        round(result["impostor_mean"], 6),
        "±",
        round(result["impostor_std"], 6)
    )

print("\nFold ROC metrics saved:", fold_metrics_path)
print("ROC summary saved:", summary_path)

validation_passed = (
    len(fold_metrics_df) == 20
    and len(summary_df) == 2
    and set(summary_df["pairs"].astype(int))
    == {6000, 5965}
    and np.isfinite(
        summary_df[
            [
                "auc",
                "eer",
                "eer_threshold",
                "mean_fold_auc",
                "mean_fold_eer"
            ]
        ].to_numpy()
    ).all()
)

if validation_passed:
    print(
        "\nROC EVALUATION PASSED: "
        "AUC and EER were calculated for both "
        "evaluation views."
    )
else:
    print(
        "\nROC EVALUATION REQUIRES REVIEW."
    )


# Cell 22
from pathlib import Path

import numpy as np
import pandas as pd


output_dir = Path(
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

pair_scores_df = pd.read_csv(
    output_dir / "pair_scores.csv"
)

if pair_scores_df["matcher_evaluable"].dtype != bool:
    pair_scores_df["matcher_evaluable"] = (
        pair_scores_df["matcher_evaluable"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

if pair_scores_df["matcher_evaluable"].isna().any():
    raise RuntimeError(
        "Could not interpret matcher_evaluable values."
    )


TARGET_FAR = 1e-3


def select_threshold_at_far(scores, labels, target_far):
    """
    Select the lowest threshold that keeps empirical training
    FAR at or below the requested target.

    Decision:
        score >= threshold -> accept
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    impostor_scores = scores[labels == 0]

    if len(impostor_scores) == 0:
        raise ValueError("No impostor scores available.")

    allowed_false_accepts = int(
        np.floor(target_far * len(impostor_scores))
    )

    descending_scores = np.sort(impostor_scores)[::-1]

    if allowed_false_accepts >= len(descending_scores):
        threshold = -np.inf
    else:
        boundary_score = descending_scores[
            allowed_false_accepts
        ]

        threshold = np.nextafter(
            boundary_score,
            np.inf
        )

    achieved_false_accepts = int(
        (impostor_scores >= threshold).sum()
    )

    achieved_far = (
        achieved_false_accepts
        / len(impostor_scores)
    )

    if achieved_far > target_far:
        raise RuntimeError(
            "Selected training threshold exceeded target FAR."
        )

    return {
        "threshold": float(threshold),
        "impostor_pairs": len(impostor_scores),
        "allowed_false_accepts": allowed_false_accepts,
        "achieved_false_accepts": achieved_false_accepts,
        "achieved_far": achieved_far
    }


def evaluate(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    predictions = (
        scores >= threshold
    ).astype(np.int8)

    tp = int(
        ((predictions == 1) & (labels == 1)).sum()
    )
    tn = int(
        ((predictions == 0) & (labels == 0)).sum()
    )
    fp = int(
        ((predictions == 1) & (labels == 0)).sum()
    )
    fn = int(
        ((predictions == 0) & (labels == 1)).sum()
    )

    genuine_pairs = tp + fn
    impostor_pairs = tn + fp

    return {
        "pairs": len(labels),
        "genuine_pairs": genuine_pairs,
        "impostor_pairs": impostor_pairs,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tar": tp / genuine_pairs,
        "far": fp / impostor_pairs,
        "frr": fn / genuine_pairs
    }


evaluation_views = {
    "end_to_end_operational": {
        "data": pair_scores_df.copy(),
        "score_column": "operational_score"
    },
    "matcher_only": {
        "data": pair_scores_df[
            pair_scores_df["matcher_evaluable"]
        ].copy(),
        "score_column": "matcher_similarity"
    }
}

fold_rows = []

for evaluation_name, specification in evaluation_views.items():

    evaluation_df = specification["data"]
    score_column = specification["score_column"]

    for test_fold in sorted(
        evaluation_df["fold"].unique()
    ):
        train_df = evaluation_df[
            evaluation_df["fold"] != test_fold
        ]

        test_df = evaluation_df[
            evaluation_df["fold"] == test_fold
        ]

        threshold_result = select_threshold_at_far(
            train_df[score_column],
            train_df["label"],
            TARGET_FAR
        )

        test_result = evaluate(
            test_df[score_column],
            test_df["label"],
            threshold_result["threshold"]
        )

        fold_rows.append({
            "evaluation": evaluation_name,
            "fold": int(test_fold),
            "target_far": TARGET_FAR,
            "threshold":
                threshold_result["threshold"],
            "training_impostor_pairs":
                threshold_result["impostor_pairs"],
            "training_allowed_fp":
                threshold_result[
                    "allowed_false_accepts"
                ],
            "training_achieved_fp":
                threshold_result[
                    "achieved_false_accepts"
                ],
            "training_achieved_far":
                threshold_result["achieved_far"],
            **test_result
        })


fold_results_df = pd.DataFrame(fold_rows)

summary_rows = []

for evaluation_name in evaluation_views:

    view_df = fold_results_df[
        fold_results_df["evaluation"]
        == evaluation_name
    ]

    tp = int(view_df["tp"].sum())
    tn = int(view_df["tn"].sum())
    fp = int(view_df["fp"].sum())
    fn = int(view_df["fn"].sum())

    genuine_pairs = tp + fn
    impostor_pairs = tn + fp

    summary_rows.append({
        "evaluation": evaluation_name,
        "target_training_far": TARGET_FAR,
        "evaluated_pairs":
            genuine_pairs + impostor_pairs,
        "genuine_pairs": genuine_pairs,
        "impostor_pairs": impostor_pairs,
        "mean_threshold":
            view_df["threshold"].mean(),
        "threshold_std":
            view_df["threshold"].std(ddof=1),
        "pooled_tar": tp / genuine_pairs,
        "pooled_observed_far": fp / impostor_pairs,
        "pooled_frr": fn / genuine_pairs,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    })


summary_df = pd.DataFrame(summary_rows)

fold_path = (
    output_dir / "tar_at_far_1e-3_fold_results.csv"
)

summary_path = (
    output_dir / "tar_at_far_1e-3_summary.csv"
)

fold_results_df.to_csv(
    fold_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("--- TAR AT TRAINING FAR 1e-3 ---")

for _, result in summary_df.iterrows():

    print(f"\n{result['evaluation'].upper()}")
    print(
        "Evaluated pairs:",
        int(result["evaluated_pairs"])
    )
    print(
        "Mean threshold:",
        round(result["mean_threshold"], 6)
    )
    print(
        "Threshold standard deviation:",
        round(result["threshold_std"], 6)
    )
    print(
        "Pooled TAR:",
        round(100 * result["pooled_tar"], 4),
        "%"
    )
    print(
        "Pooled observed FAR:",
        round(
            100 * result["pooled_observed_far"],
            4
        ),
        "%"
    )
    print(
        "Pooled FRR:",
        round(100 * result["pooled_frr"], 4),
        "%"
    )
    print(
        "False acceptances:",
        int(result["fp"])
    )
    print(
        "False rejections:",
        int(result["fn"])
    )

print("\nFold results saved:", fold_path)
print("Summary saved:", summary_path)

validation_passed = (
    len(fold_results_df) == 20
    and len(summary_df) == 2
    and set(summary_df["evaluated_pairs"].astype(int))
    == {6000, 5965}
    and (
        fold_results_df["training_achieved_far"]
        <= TARGET_FAR
    ).all()
    and np.isfinite(
        fold_results_df[
            ["threshold", "tar", "far", "frr"]
        ].to_numpy()
    ).all()
)

if validation_passed:
    print(
        "\nLOW-FAR EVALUATION PASSED: "
        "Every test-fold threshold was calibrated using "
        "only the other nine folds."
    )
else:
    print(
        "\nLOW-FAR EVALUATION REQUIRES REVIEW."
    )


# Cell 23
import shutil

archive_path = shutil.make_archive(
    "/kaggle/working/LFW-E01-raw-baseline",
    "zip",
    "/kaggle/working/experiments/LFW-E01-raw-baseline"
)

print("Experiment archive created:", archive_path)


# Cell 24

