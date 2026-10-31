"""
Final evaluation for EXP-TROCR-VIN-001.

Evaluates the best validation checkpoint on the held-out test set.

Outputs:
- training_loss_curve.png
- predictions.csv
- failures.csv
- metrics.json
- evaluation_report.txt

Metrics:
- Exact-string accuracy
- Aligned character accuracy
- Character error rate (CER)
- String error rate
- Character-level micro-F1 and macro-F1
- Mean, median and p95 inference latency
- Images per second
- Test-time peak GPU memory

Timing:
- Full pipeline: image loading + preprocessing + generation
- Model only: generation using preprocessed tensors
"""

import csv
import json
import time
from pathlib import Path
from itertools import zip_longest

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import f1_score
from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
)


# ============================================================
# 1. PATHS AND EXPERIMENT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "EXP-TROCR-VIN-001.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EXPERIMENT = config["project"]["experiment_name"]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / EXPERIMENT
)

MODEL_DIR = RESULTS_DIR / "best_model"

DATASET_ROOT = (
    PROJECT_ROOT
    / config["data"]["dataset_root"]
)

TEST_MANIFEST = (
    DATASET_ROOT
    / config["data"]["test_manifest"]
)

METRICS_CSV = RESULTS_DIR / "epoch_metrics.csv"

OUTPUT_DIR = RESULTS_DIR / "final_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = config["model"]["image_width"]
IMAGE_HEIGHT = config["model"]["image_height"]

MAX_LENGTH = config["evaluation"]["generation_max_length"]
NUM_BEAMS = config["evaluation"]["generation_num_beams"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP_RUNS = 10

# Match the training configuration.
BATCH_SIZE = 1
PRECISION = "FP16" if (
    DEVICE == "cuda" and config["training"]["mixed_precision"] == "fp16"
) else "FP32"

GAP = "<gap>"


# ============================================================
# 2. VALIDATE INPUT FILES
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Best model not found: {MODEL_DIR}")

if not TEST_MANIFEST.exists():
    raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST}")

if not METRICS_CSV.exists():
    raise FileNotFoundError(f"Training metrics CSV not found: {METRICS_CSV}")


# ============================================================
# 3. NORMALISATION AND CHARACTER ALIGNMENT
# ============================================================

def normalise(text):
    """
    VIN normalisation:
    - Convert to uppercase.
    - Remove spaces.
    - Remove leading/trailing whitespace.

    No character substitutions are performed.
    """

    return str(text).upper().replace(" ", "").strip()


def align_chars(gt, pred):
    """
    Align ground truth and prediction using Levenshtein opcodes.

    Insertions and deletions are represented using GAP.
    This prevents one missing character from shifting all
    subsequent character comparisons.
    """

    y_true = []
    y_pred = []

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gt, pred):

        gt_part = gt[i1:i2]
        pred_part = pred[j1:j2]

        for true_char, pred_char in zip_longest(
            gt_part,
            pred_part,
            fillvalue=GAP
        ):
            y_true.append(true_char)
            y_pred.append(pred_char)

    return y_true, y_pred


# ============================================================
# 4. READ TEST MANIFEST
# ============================================================

def load_test_manifest(manifest_path):

    samples = []

    with open(manifest_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid manifest line {line_number}: {line}"
                )

            image_path = Path(parts[0])
            ground_truth = normalise(parts[1])

            if not image_path.is_absolute():
                image_path = DATASET_ROOT / image_path

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Test image does not exist: {image_path}"
                )

            if len(ground_truth) != 17:
                raise ValueError(
                    f"Invalid VIN length on line {line_number}: "
                    f"{ground_truth}"
                )

            samples.append({
                "image_path": image_path,
                "ground_truth": ground_truth,
            })

    if not samples:
        raise ValueError("The test manifest contains no images.")

    return samples


samples = load_test_manifest(TEST_MANIFEST)

print("\nTEST DATASET")
print("============")
print(f"Test images: {len(samples)}")
print(f"Test manifest: {TEST_MANIFEST}")


# ============================================================
# 5. LOAD MODEL AND PROCESSOR
# ============================================================

print("\nLoading best TrOCR checkpoint...")

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

# Ensure the processor does not resize independently.
# Images are padded to the same dimensions used in training.
processor.image_processor.do_resize = False

print(f"Model: {MODEL_DIR}")
print(f"Device: {DEVICE}")
print(f"Precision: {PRECISION}")


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = ImageOps.pad(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    return pixel_values


# ============================================================
# 7. MODEL GENERATION
# ============================================================

def generate_prediction(pixel_values):

    pixel_values = pixel_values.to(DEVICE)

    with torch.inference_mode():

        # Explicitly enable positional-encoding interpolation
        # for the 768 x 128 training input dimensions.
        encoder_outputs = model.encoder(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
            return_dict=True,
        )

        generated_ids = model.generate(
            encoder_outputs=encoder_outputs,
            num_beams=NUM_BEAMS,
            max_length=MAX_LENGTH,
        )

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return normalise(prediction)


# ============================================================
# 8. WARM-UP AND FULL PIPELINE EVALUATION
# ============================================================

print("\nRunning warm-up...")

warmup_samples = samples[:min(WARMUP_RUNS, len(samples))]

with torch.inference_mode():

    for sample in warmup_samples:

        pixels = preprocess_image(sample["image_path"])
        generate_prediction(pixels)

if DEVICE == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


print("\nRunning final test evaluation...")

predictions = []
full_pipeline_times = []

for index, sample in enumerate(samples, start=1):

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    # Full pipeline timing starts before image loading.
    pixels = preprocess_image(sample["image_path"])
    prediction = generate_prediction(pixels)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    predictions.append(prediction)
    full_pipeline_times.append(elapsed_ms)

    print(
        f"[{index}/{len(samples)}] "
        f"GT: {sample['ground_truth']} | "
        f"PRED: {prediction} | "
        f"{elapsed_ms:.2f} ms"
    )


# ============================================================
# 9. MODEL-ONLY INFERENCE TIMING
# ============================================================

print("\nMeasuring model-only inference time...")

# Preprocess outside the timer.
preprocessed_images = [
    preprocess_image(sample["image_path"])
    for sample in samples
]

# Separate model-only warm-up.
with torch.inference_mode():

    for pixels in preprocessed_images[:min(WARMUP_RUNS, len(samples))]:
        generate_prediction(pixels)

if DEVICE == "cuda":
    torch.cuda.synchronize()

model_only_times = []

with torch.inference_mode():

    for pixels in preprocessed_images:

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        generate_prediction(pixels)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000

        model_only_times.append(elapsed_ms)


# ============================================================
# 10. IMAGE-LEVEL AND CHARACTER-LEVEL METRICS
# ============================================================

gts = [
    normalise(sample["ground_truth"])
    for sample in samples
]

preds = [
    normalise(prediction)
    for prediction in predictions
]

assert len(gts) == len(preds), (
    "Every test image must have exactly one prediction."
)

number_of_images = len(gts)

correct = sum(
    gt == pred
    for gt, pred in zip(gts, preds)
)

fail = number_of_images - correct

exact_string_accuracy = correct / number_of_images

string_error_rate = fail / number_of_images

total_gt_characters = sum(len(gt) for gt in gts)

total_edit_distance = sum(
    Levenshtein.distance(gt, pred)
    for gt, pred in zip(gts, preds)
)

cer = total_edit_distance / total_gt_characters


# Character alignment.
y_true = []
y_pred = []

for gt, pred in zip(gts, preds):

    aligned_true, aligned_pred = align_chars(gt, pred)

    y_true.extend(aligned_true)
    y_pred.extend(aligned_pred)


# Aligned character accuracy.
# This counts matching real characters divided by total GT characters.
character_matches = sum(
    true_char == pred_char
    for true_char, pred_char in zip(y_true, y_pred)
    if true_char != GAP
)

character_accuracy = (
    character_matches / total_gt_characters
)


# F1 uses the VIN character alphabet as its class labels.
# GAP is deliberately not a character class:
# deletions count as false negatives and insertions as false positives.
VIN_ALPHABET = sorted(set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ"))

f1_micro = f1_score(
    y_true,
    y_pred,
    labels=VIN_ALPHABET,
    average="micro",
    zero_division=0,
)

# Macro F1 over the characters present in the ground truth.
ground_truth_labels = sorted(set(y_true) - {GAP})

f1_macro = f1_score(
    y_true,
    y_pred,
    labels=ground_truth_labels,
    average="macro",
    zero_division=0,
)


# ============================================================
# 11. INFERENCE SPEED
# ============================================================

def latency_statistics(times):

    values = np.asarray(times, dtype=float)

    mean_ms = float(np.mean(values))
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))

    images_per_second = (
        1000.0 / mean_ms if mean_ms > 0 else 0.0
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "images_per_second": images_per_second,
    }


full_pipeline_speed = latency_statistics(
    full_pipeline_times
)

model_only_speed = latency_statistics(
    model_only_times
)


# ============================================================
# 12. TEST-TIME PEAK GPU MEMORY
# ============================================================

if DEVICE == "cuda":

    torch.cuda.synchronize()

    peak_gpu_memory_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
    )

    peak_gpu_memory_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024 ** 2)
    )

    total_gpu_memory_mb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 2)
    )

    gpu_name = torch.cuda.get_device_name(0)

else:

    peak_gpu_memory_allocated_mb = None
    peak_gpu_memory_reserved_mb = None
    total_gpu_memory_mb = None
    gpu_name = "CPU"


# ============================================================
# 13. FAILURE RECORDS
# ============================================================

failures = []

for index, (sample, gt, pred) in enumerate(
    zip(samples, gts, preds),
    start=1
):

    if gt != pred:

        failures.append({
            "image_number": index,
            "image_path": str(sample["image_path"]),
            "ground_truth": gt,
            "prediction": pred,
            "edit_distance": Levenshtein.distance(gt, pred),
        })


# ============================================================
# 14. SAVE PREDICTIONS CSV
# ============================================================

predictions_path = OUTPUT_DIR / "predictions.csv"

with open(
    predictions_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_number",
            "image_path",
            "ground_truth",
            "prediction",
            "correct",
            "edit_distance",
            "full_pipeline_time_ms",
            "model_only_time_ms",
        ]
    )

    writer.writeheader()

    for index, (sample, gt, pred) in enumerate(
        zip(samples, gts, preds),
        start=1
    ):

        writer.writerow({
            "image_number": index,
            "image_path": str(sample["image_path"]),
            "ground_truth": gt,
            "prediction": pred,
            "correct": gt == pred,
            "edit_distance": Levenshtein.distance(gt, pred),
            "full_pipeline_time_ms": full_pipeline_times[index - 1],
            "model_only_time_ms": model_only_times[index - 1],
        })


# ============================================================
# 15. SAVE FAILURE CSV
# ============================================================

failures_path = OUTPUT_DIR / "failures.csv"

with open(
    failures_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "image_number",
        "image_path",
        "ground_truth",
        "prediction",
        "edit_distance",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(failures)


# ============================================================
# 16. GENERATE TRAINING CURVES
# ============================================================

epochs = []
train_losses = []
validation_losses = []

with open(METRICS_CSV, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        validation_losses.append(float(row["validation_loss"]))


plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training loss"
)

plt.plot(
    epochs,
    validation_losses,
    marker="o",
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{EXPERIMENT} - Training and Validation Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

curve_path = OUTPUT_DIR / "training_loss_curve.png"

plt.savefig(curve_path, dpi=300)
plt.close()


# ============================================================
# 17. SAVE METRICS JSON
# ============================================================

metrics = {

    "experiment": EXPERIMENT,

    "checkpoint": str(MODEL_DIR),

    "test_dataset": str(TEST_MANIFEST),

    "number_of_test_images": number_of_images,

    "normalisation": (
        "Uppercase, remove spaces, strip leading/trailing whitespace. "
        "No character substitutions."
    ),

    "image_level": {
        "correct": correct,
        "fail": fail,
        "exact_string_accuracy": exact_string_accuracy,
        "string_error_rate": string_error_rate,
    },

    "character_level": {
        "total_ground_truth_characters": total_gt_characters,
        "character_matches": character_matches,
        "aligned_character_accuracy": character_accuracy,
        "total_edit_distance": total_edit_distance,
        "CER": cer,
        "F1_micro": float(f1_micro),
        "F1_macro": float(f1_macro),
        "F1_macro_classes": ground_truth_labels,
    },

    "inference": {
        "warmup_runs": min(WARMUP_RUNS, number_of_images),
        "batch_size": BATCH_SIZE,
        "precision": PRECISION,
        "full_pipeline": full_pipeline_speed,
        "model_only": model_only_speed,
    },

    "hardware": {
        "device": DEVICE,
        "gpu": gpu_name,
        "total_gpu_memory_mb": total_gpu_memory_mb,
        "test_peak_gpu_memory_allocated_mb": (
            peak_gpu_memory_allocated_mb
        ),
        "test_peak_gpu_memory_reserved_mb": (
            peak_gpu_memory_reserved_mb
        ),
    },

    "training_curve": str(curve_path),

    "predictions_csv": str(predictions_path),

    "failures_csv": str(failures_path),

    "important_note": (
        "GPU peak memory in this report is measured during final "
        "test evaluation. It is not the historical peak memory "
        "used during model training."
    ),
}

metrics_path = OUTPUT_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 18. GENERATE HUMAN-READABLE REPORT
# ============================================================

report_path = OUTPUT_DIR / "evaluation_report.txt"

report = []

report.append("=" * 65)
report.append(f"FINAL OCR EVALUATION: {EXPERIMENT}")
report.append("=" * 65)

report.append("\nIMAGE-LEVEL RESULTS")
report.append("-" * 65)
report.append(f"Test images: {number_of_images}")
report.append(f"Correct VINs: {correct}")
report.append(f"Failed VINs: {fail}")
report.append(
    f"Exact-string accuracy: {exact_string_accuracy * 100:.2f}%"
)
report.append(
    f"String error rate: {string_error_rate * 100:.2f}%"
)

report.append("\nCHARACTER-LEVEL RESULTS")
report.append("-" * 65)
report.append(f"Total GT characters: {total_gt_characters}")
report.append(f"Character matches: {character_matches}")
report.append(
    f"Aligned character accuracy: {character_accuracy * 100:.2f}%"
)
report.append(f"Total edit distance: {total_edit_distance}")
report.append(f"CER: {cer * 100:.2f}%")
report.append(f"Character micro-F1: {f1_micro:.4f}")
report.append(f"Character macro-F1: {f1_macro:.4f}")

report.append("\nFULL PIPELINE INFERENCE")
report.append("-" * 65)
report.append(
    f"Mean: {full_pipeline_speed['mean_ms']:.2f} ms/image"
)
report.append(
    f"Median: {full_pipeline_speed['median_ms']:.2f} ms/image"
)
report.append(
    f"P95: {full_pipeline_speed['p95_ms']:.2f} ms/image"
)
report.append(
    f"Throughput: "
    f"{full_pipeline_speed['images_per_second']:.2f} images/second"
)

report.append("\nMODEL-ONLY INFERENCE")
report.append("-" * 65)
report.append(
    f"Mean: {model_only_speed['mean_ms']:.2f} ms/image"
)
report.append(
    f"Median: {model_only_speed['median_ms']:.2f} ms/image"
)
report.append(
    f"P95: {model_only_speed['p95_ms']:.2f} ms/image"
)
report.append(
    f"Throughput: "
    f"{model_only_speed['images_per_second']:.2f} images/second"
)

report.append("\nHARDWARE")
report.append("-" * 65)
report.append(f"Device: {DEVICE}")
report.append(f"GPU: {gpu_name}")
report.append(f"Precision: {PRECISION}")
report.append(f"Batch size: {BATCH_SIZE}")

if DEVICE == "cuda":

    report.append(
        f"Test peak allocated GPU memory: "
        f"{peak_gpu_memory_allocated_mb:.2f} MB"
    )

    report.append(
        f"Test peak reserved GPU memory: "
        f"{peak_gpu_memory_reserved_mb:.2f} MB"
    )

    report.append(
        f"Total GPU memory: {total_gpu_memory_mb:.2f} MB"
    )

else:

    report.append("GPU memory: Not applicable (CPU evaluation)")


report.append("\nFAILURE ANALYSIS")
report.append("-" * 65)

if failures:

    for failure in failures:

        report.append(
            f"FAIL #{failure['image_number']}: "
            f"GT={failure['ground_truth']} | "
            f"PRED={failure['prediction']} | "
            f"Edits={failure['edit_distance']}"
        )

else:

    report.append("No incorrect VIN predictions.")


report.append("\nOUTPUT FILES")
report.append("-" * 65)
report.append(str(predictions_path))
report.append(str(failures_path))
report.append(str(metrics_path))
report.append(str(curve_path))

report.append("\nIMPORTANT")
report.append("-" * 65)
report.append(
    "Peak GPU memory is from final test evaluation, "
    "not historical training."
)

report.append(
    "Full pipeline timing includes image loading, preprocessing "
    "and TrOCR generation."
)

report.append(
    "Model-only timing includes TrOCR generation using "
    "preprocessed tensors."
)

report.append(
    "Character accuracy is aligned matching characters divided "
    "by total ground-truth characters. CER uses total Levenshtein "
    "edit distance divided by total ground-truth characters. "
    "They are not necessarily complements when insertions occur."
)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("\n".join(report))


# ============================================================
# 19. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print(f"FINAL TEST RESULTS: {EXPERIMENT}")
print("=" * 65)

print(
    f"Exact-string accuracy: "
    f"{exact_string_accuracy * 100:.2f}%"
)

print(
    f"Aligned character accuracy: "
    f"{character_accuracy * 100:.2f}%"
)

print(f"CER: {cer * 100:.2f}%")
print(f"String error rate: {string_error_rate * 100:.2f}%")

print(f"Character micro-F1: {f1_micro:.4f}")
print(f"Character macro-F1: {f1_macro:.4f}")

print("\nFULL PIPELINE SPEED")
print(
    f"Mean: {full_pipeline_speed['mean_ms']:.2f} ms"
)
print(
    f"Median: {full_pipeline_speed['median_ms']:.2f} ms"
)
print(
    f"P95: {full_pipeline_speed['p95_ms']:.2f} ms"
)
print(
    f"Throughput: "
    f"{full_pipeline_speed['images_per_second']:.2f} images/s"
)

print("\nMODEL-ONLY SPEED")
print(
    f"Mean: {model_only_speed['mean_ms']:.2f} ms"
)
print(
    f"Median: {model_only_speed['median_ms']:.2f} ms"
)
print(
    f"P95: {model_only_speed['p95_ms']:.2f} ms"
)
print(
    f"Throughput: "
    f"{model_only_speed['images_per_second']:.2f} images/s"
)

print("\nTEST-TIME GPU MEMORY")
print(
    f"Peak allocated: "
    f"{peak_gpu_memory_allocated_mb} MB"
)
print(
    f"Peak reserved: "
    f"{peak_gpu_memory_reserved_mb} MB"
)

print(f"\nIncorrect VINs: {fail}")

print("\nEvaluation completed.")
print(f"Results saved in: {OUTPUT_DIR}")