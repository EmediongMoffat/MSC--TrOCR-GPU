
"""
EXP-TROCR-VIN-002
Independent VIN OCR test evaluation.

- Loads the saved best_model checkpoint.
- Evaluates rec_gt_test.txt only.
- Reproduces the training image preprocessing.
- Does not train or update model weights.
- Saves predictions, metrics and annotated recognition examples.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# 1. PROJECT PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "EXP-TROCR-VIN-002"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{EXPERIMENT_NAME}.yaml"

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "TROCR_YOLO_PREDICTED_LINE_RECOGNITION"
)

TEST_MANIFEST = DATA_ROOT / "rec_gt_test.txt"

RESULTS_ROOT = PROJECT_ROOT / "results" / EXPERIMENT_NAME

BEST_MODEL_DIR = RESULTS_ROOT / "best_model"

EVALUATION_ROOT = RESULTS_ROOT / "test_evaluation"

# These must match the training script.
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 128
MAX_TARGET_LENGTH = 32

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_BEAMS = 1
GENERATION_MAX_LENGTH = 32

# Number of annotated examples to save.
MAX_RECOGNITION_EXAMPLES = 8


# ============================================================
# 2. DEVICE AND PRECISION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_FP16 = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("=" * 70)
print("EXP-TROCR-VIN-002 - INDEPENDENT TEST EVALUATION")
print("=" * 70)

print("Project root:", PROJECT_ROOT)
print("Checkpoint:", BEST_MODEL_DIR)
print("Test manifest:", TEST_MANIFEST)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. LOAD EXPERIMENT CONFIGURATION
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("\nConfiguration loaded:", CONFIG_PATH.name)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

if not BEST_MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Best model checkpoint not found:\n{BEST_MODEL_DIR}"
    )

if not TEST_MANIFEST.exists():
    raise FileNotFoundError(
        f"Test manifest not found:\n{TEST_MANIFEST}"
    )

if not (BEST_MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        "The best_model folder does not appear to contain a saved model."
    )

# Each run receives its own folder so previous results are not overwritten.
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR = EVALUATION_ROOT / f"run_{run_timestamp}"

PREDICTIONS_DIR = RUN_DIR / "predictions"
EXAMPLES_DIR = RUN_DIR / "recognition_examples"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=False)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=False)


# ============================================================
# 5. EDIT DISTANCE AND CHARACTER METRICS
# ============================================================

def edit_distance(reference, prediction):
    """Levenshtein edit distance."""

    rows = len(reference) + 1
    cols = len(prediction) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def aligned_character_counts(reference, prediction):
    """
    Character-level alignment counts.

    TP: correctly matched characters.
    FP: substitutions and insertions.
    FN: substitutions and deletions.
    """

    rows = len(reference) + 1
    cols = len(prediction) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    # Backtrack through the alignment.
    i = len(reference)
    j = len(prediction)

    tp = 0
    fp = 0
    fn = 0

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and reference[i - 1] == prediction[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            tp += 1
            i -= 1
            j -= 1

        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            fp += 1
            fn += 1
            i -= 1
            j -= 1

        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            fp += 1
            j -= 1

        else:
            fn += 1
            i -= 1

    return tp, fp, fn


def calculate_auc_roc(labels, scores):
    """
    Calculate binary ROC-AUC using the rank method.

    Positive class: exact VIN prediction.
    Score: mean generated-token confidence.

    Returns None if the test set contains only one class.
    """

    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)

    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.zeros(len(scores), dtype=np.float64)

    start = 0

    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank

        start = end

    positive_rank_sum = np.sum(ranks[labels == 1])

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)

    return float(auc)


def calculate_metrics(references, predictions, confidences):

    total = len(references)

    if total == 0:
        raise ValueError("The test dataset contains no records.")

    exact_matches = [
        reference == prediction
        for reference, prediction in zip(references, predictions)
    ]

    exact_correct = sum(exact_matches)

    total_edit_distance = 0
    total_reference_characters = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for reference, prediction in zip(references, predictions):

        total_edit_distance += edit_distance(reference, prediction)

        total_reference_characters += len(reference)

        tp, fp, fn = aligned_character_counts(
            reference,
            prediction
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    exact_accuracy = exact_correct / total

    character_accuracy = (
        total_tp / total_reference_characters
        if total_reference_characters > 0
        else 0.0
    )

    cer = (
        total_edit_distance / total_reference_characters
        if total_reference_characters > 0
        else 0.0
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    string_error_rate = 1.0 - exact_accuracy

    auc = calculate_auc_roc(
        [int(value) for value in exact_matches],
        confidences
    )

    return {
        "test_images": total,
        "correct_vins": exact_correct,
        "incorrect_vins": total - exact_correct,

        "accuracy_percent": exact_accuracy * 100,
        "exact_string_accuracy": exact_accuracy,

        "character_accuracy": character_accuracy,
        "character_accuracy_percent": character_accuracy * 100,

        "character_error_rate_CER": cer,
        "word_string_error_rate": string_error_rate,

        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,

        "AUC_ROC": auc,

        "total_edit_distance": total_edit_distance,
        "total_reference_characters": total_reference_characters
    }


# ============================================================
# 6. DATASET
# ============================================================

class VINDataset(Dataset):

    def __init__(self, manifest_path, data_root, processor):

        self.data_root = Path(data_root)
        self.processor = processor
        self.samples = []

        with open(manifest_path, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split("\t", 1)

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest row {line_number}: {line}"
                    )

                image_path_text = parts[0].strip()
                vin = parts[1].strip()

                image_path = Path(image_path_text)

                if not image_path.is_absolute():
                    image_path = self.data_root / image_path

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image missing at manifest row {line_number}:\n"
                        f"{image_path}"
                    )

                if len(vin) != 17:
                    raise ValueError(
                        f"VIN at manifest row {line_number} "
                        f"has {len(vin)} characters, not 17: {vin}"
                    )

                self.samples.append(
                    (image_path, image_path_text, vin)
                )

        print("\nTest samples loaded:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, relative_path, vin = self.samples[index]

        # Reproduce the training preprocessing exactly.
        image = Image.open(image_path).convert("RGB")

        image = ImageOps.pad(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5)
        )

        pixel_values = self.processor.image_processor(
            images=image,
            return_tensors="pt",
            do_resize=False
        ).pixel_values.squeeze(0)

        tokenized = self.processor.tokenizer(
            vin,
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

        labels = tokenized.input_ids.squeeze(0)

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "vin": vin,
            "image_path": str(image_path),
            "relative_path": relative_path
        }


def collate_fn(batch):

    return {
        "pixel_values": torch.stack(
            [item["pixel_values"] for item in batch]
        ),

        "labels": torch.stack(
            [item["labels"] for item in batch]
        ),

        "vins": [item["vin"] for item in batch],

        "image_paths": [item["image_path"] for item in batch],

        "relative_paths": [
            item["relative_path"] for item in batch
        ]
    }


# ============================================================
# 7. LOAD PROCESSOR AND BEST CHECKPOINT
# ============================================================

print("\nLoading saved processor...")

processor = TrOCRProcessor.from_pretrained(
    BEST_MODEL_DIR,
    local_files_only=True
)

print("Loading best model checkpoint...")

model = VisionEncoderDecoderModel.from_pretrained(
    BEST_MODEL_DIR,
    local_files_only=True
)

tokenizer = processor.tokenizer

model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.use_cache = True

model.to(DEVICE)
model.eval()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model loaded successfully.")
print("Total parameters:", f"{total_parameters:,}")


# ============================================================
# 8. LOAD TEST DATA
# ============================================================

test_dataset = VINDataset(
    manifest_path=TEST_MANIFEST,
    data_root=DATA_ROOT,
    processor=processor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

if len(test_dataset) == 0:
    raise ValueError("No test images were loaded.")


# ============================================================
# 9. GENERATION FUNCTION
# ============================================================

def generate_batch(pixel_values, return_scores=False):

    generation_kwargs = {
        "max_length": GENERATION_MAX_LENGTH,
        "num_beams": NUM_BEAMS,
        "interpolate_pos_encoding": True,
        "use_cache": True
    }

    if return_scores:
        generation_kwargs["return_dict_in_generate"] = True
        generation_kwargs["output_scores"] = True

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=USE_FP16
    ):

        output = model.generate(
            pixel_values,
            **generation_kwargs
        )

    if return_scores:
        return output

    return output


# ============================================================
# 10. WARM-UP
# ============================================================

print("\nRunning warm-up...")

first_batch = next(iter(test_loader))

warmup_pixels = first_batch["pixel_values"].to(DEVICE)

with torch.inference_mode():

    for _ in range(2):

        _ = generate_batch(
            warmup_pixels,
            return_scores=False
        )

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print("Warm-up complete.")


# Reset peak memory statistics after warm-up.
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)


# ============================================================
# 11. TIMED TEST INFERENCE
# ============================================================

print("\nRunning timed test inference...")

references = []
predictions = []
image_paths = []
relative_paths = []

batch_times_seconds = []
batch_sizes = []

with torch.inference_mode():

    for batch_number, batch in enumerate(test_loader, start=1):

        pixel_values = batch["pixel_values"].to(DEVICE)

        actual_batch_size = pixel_values.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        generated_output = generate_batch(
            pixel_values,
            return_scores=False
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        decoded_predictions = processor.batch_decode(
            generated_output,
            skip_special_tokens=True
        )

        batch_times_seconds.append(elapsed)
        batch_sizes.append(actual_batch_size)

        references.extend(batch["vins"])
        predictions.extend(
            [text.strip() for text in decoded_predictions]
        )

        image_paths.extend(batch["image_paths"])
        relative_paths.extend(batch["relative_paths"])

        print(
            f"Batch {batch_number}/{len(test_loader)} "
            f"| Images: {actual_batch_size} "
            f"| Time: {elapsed:.4f}s"
        )


# ============================================================
# 12. CALCULATE TEST LOSS
# ============================================================

print("\nCalculating teacher-forced test loss...")

total_loss = 0.0
total_label_tokens = 0

with torch.inference_mode():

    for batch in test_loader:

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        valid_token_count = (labels != -100).sum().item()

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_FP16
        ):

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
                interpolate_pos_encoding=True,
                use_cache=False
            )

        total_loss += outputs.loss.item() * valid_token_count
        total_label_tokens += valid_token_count

test_loss = (
    total_loss / total_label_tokens
    if total_label_tokens > 0
    else None
)

print("Teacher-forced test loss:", test_loss)


# ============================================================
# 13. GENERATE TOKEN CONFIDENCE SCORES
# ============================================================

print("\nCalculating generated-token confidence scores...")

confidences = []

special_token_ids = set(tokenizer.all_special_ids)

with torch.inference_mode():

    for batch in test_loader:

        pixel_values = batch["pixel_values"].to(DEVICE)

        generation_output = generate_batch(
            pixel_values,
            return_scores=True
        )

        sequences = generation_output.sequences
        step_scores = generation_output.scores

        batch_confidences = [[] for _ in range(sequences.size(0))]

        for step, logits in enumerate(step_scores):

            sequence_position = step + 1

            if sequence_position >= sequences.size(1):
                break

            token_ids = sequences[:, sequence_position]

            probabilities = torch.softmax(
                logits.float(),
                dim=-1
            )

            selected_probabilities = probabilities.gather(
                1,
                token_ids.unsqueeze(1)
            ).squeeze(1)

            for item_index, token_id in enumerate(token_ids.tolist()):

                if token_id not in special_token_ids:

                    batch_confidences[item_index].append(
                        selected_probabilities[item_index].item()
                    )

        for token_confidences in batch_confidences:

            if token_confidences:
                confidences.append(
                    float(np.mean(token_confidences))
                )
            else:
                confidences.append(0.0)


if len(confidences) != len(predictions):
    raise RuntimeError(
        "Confidence count does not match prediction count."
    )


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

print("\nCalculating test metrics...")

metrics = calculate_metrics(
    references,
    predictions,
    confidences
)

metrics["test_loss_teacher_forced"] = test_loss

metrics["experiment"] = EXPERIMENT_NAME

metrics["checkpoint"] = str(BEST_MODEL_DIR)

metrics["test_manifest"] = str(TEST_MANIFEST)

metrics["test_set_used_for_training"] = False

metrics["test_set_used_for_model_selection"] = False

metrics["device"] = str(DEVICE)

metrics["gpu_name"] = (
    torch.cuda.get_device_name(0)
    if DEVICE.type == "cuda"
    else None
)

metrics["model_parameters"] = total_parameters

metrics["image_width"] = IMAGE_WIDTH
metrics["image_height"] = IMAGE_HEIGHT
metrics["max_target_length"] = MAX_TARGET_LENGTH
metrics["batch_size"] = BATCH_SIZE
metrics["num_beams"] = NUM_BEAMS


# ============================================================
# 15. INFERENCE TIME AND THROUGHPUT
# ============================================================

total_inference_seconds = sum(batch_times_seconds)

total_images = len(predictions)

mean_latency_ms_per_image = (
    total_inference_seconds * 1000 / total_images
)

# Amortised batch time per image.
# This is not the same as measuring each image individually.
amortised_per_image_ms = []

for batch_time, batch_size in zip(
    batch_times_seconds,
    batch_sizes
):

    value = (batch_time * 1000) / batch_size

    amortised_per_image_ms.extend([value] * batch_size)

batch_latency_ms = [
    value * 1000
    for value in batch_times_seconds
]

metrics["inference_total_seconds"] = total_inference_seconds

metrics["inference_latency_ms_per_image_mean"] = (
    mean_latency_ms_per_image
)

metrics["inference_latency_ms_per_image_median_amortised"] = float(
    np.median(amortised_per_image_ms)
)

metrics["inference_latency_ms_per_image_p95_amortised"] = float(
    np.percentile(amortised_per_image_ms, 95)
)

metrics["batch_latency_ms_mean"] = float(
    np.mean(batch_latency_ms)
)

metrics["batch_latency_ms_median"] = float(
    np.median(batch_latency_ms)
)

metrics["batch_latency_ms_p95"] = float(
    np.percentile(batch_latency_ms, 95)
)

metrics["inference_throughput_images_per_second"] = (
    total_images / total_inference_seconds
    if total_inference_seconds > 0
    else None
)


# ============================================================
# 16. GPU MEMORY
# ============================================================

if DEVICE.type == "cuda":

    gpu_properties = torch.cuda.get_device_properties(DEVICE)

    metrics["gpu_total_memory_gib"] = (
        gpu_properties.total_memory / (1024 ** 3)
    )

    metrics["gpu_peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    )

    metrics["gpu_peak_reserved_gib"] = (
        torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 3)
    )

else:

    metrics["gpu_total_memory_gib"] = None
    metrics["gpu_peak_allocated_gib"] = None
    metrics["gpu_peak_reserved_gib"] = None


# ============================================================
# 17. SAVE TEST PREDICTIONS CSV
# ============================================================

prediction_csv = PREDICTIONS_DIR / "test_predictions.csv"

with open(
    prediction_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image_path",
        "ground_truth_vin",
        "predicted_vin",
        "exact_match",
        "edit_distance",
        "character_error_rate",
        "mean_token_confidence"
    ])

    for index in range(total_images):

        reference = references[index]
        prediction = predictions[index]

        distance = edit_distance(reference, prediction)

        cer = (
            distance / len(reference)
            if len(reference) > 0
            else 0.0
        )

        writer.writerow([
            relative_paths[index],
            reference,
            prediction,
            reference == prediction,
            distance,
            cer,
            confidences[index]
        ])

print("\nPredictions saved:", prediction_csv)


# ============================================================
# 18. CREATE ANNOTATED RECOGNITION IMAGES
# ============================================================

def load_font(size=20):

    possible_fonts = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf"
    ]

    for font_path in possible_fonts:

        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass

    return ImageFont.load_default()


def create_annotated_image(
    image_path,
    reference,
    prediction,
    output_path
):

    image = Image.open(image_path).convert("RGB")

    # Display the same padded image shape used by training.
    image = ImageOps.pad(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5)
    )

    # Enlarge the displayed crop for easier visual inspection.
    display_width = 1200
    display_height = int(
        image.height * display_width / image.width
    )

    image = image.resize(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )

    caption_height = 105

    canvas = Image.new(
        "RGB",
        (display_width, display_height + caption_height),
        "white"
    )

    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    font = load_font(20)

    exact_match = reference == prediction

    status = (
        "EXACT MATCH: YES"
        if exact_match
        else "EXACT MATCH: NO"
    )

    draw.text(
        (15, display_height + 8),
        f"GROUND TRUTH: {reference}",
        fill="black",
        font=font
    )

    draw.text(
        (15, display_height + 37),
        f"PREDICTION:   {prediction}",
        fill="black",
        font=font
    )

    draw.text(
        (15, display_height + 68),
        status,
        fill="green" if exact_match else "red",
        font=font
    )

    canvas.save(output_path)


# Choose a mixture of correct and incorrect examples.
correct_indices = [
    i for i in range(total_images)
    if references[i] == predictions[i]
]

incorrect_indices = [
    i for i in range(total_images)
    if references[i] != predictions[i]
]

# Put the most visibly incorrect examples first, based on edit distance.
incorrect_indices.sort(
    key=lambda i: edit_distance(
        references[i],
        predictions[i]
    ),
    reverse=True
)

selected_indices = []

# Include up to four correct examples.
selected_indices.extend(
    correct_indices[:4]
)

# Include up to four incorrect examples.
selected_indices.extend(
    incorrect_indices[:4]
)

# If fewer than eight have been selected, fill from remaining records.
for index in range(total_images):

    if len(selected_indices) >= MAX_RECOGNITION_EXAMPLES:
        break

    if index not in selected_indices:
        selected_indices.append(index)

selected_indices = selected_indices[
    :MAX_RECOGNITION_EXAMPLES
]

example_records = []

for example_number, index in enumerate(
    selected_indices,
    start=1
):

    is_correct = references[index] == predictions[index]

    category = "correct" if is_correct else "incorrect"

    filename = (
        f"example_{example_number:02d}_{category}.png"
    )

    output_path = EXAMPLES_DIR / filename

    create_annotated_image(
        image_path=image_paths[index],
        reference=references[index],
        prediction=predictions[index],
        output_path=output_path
    )

    example_records.append({
        "filename": filename,
        "image_path": relative_paths[index],
        "ground_truth": references[index],
        "prediction": predictions[index],
        "exact_match": is_correct
    })

print("\nRecognition examples saved:", EXAMPLES_DIR)


# ============================================================
# 19. CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(example_records, examples_dir, output_path):

    if not example_records:
        return

    columns = 2
    tile_width = 900
    tile_height = 260

    rows = (
        len(example_records) + columns - 1
    ) // columns

    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        (235, 235, 235)
    )

    for position, record in enumerate(example_records):

        image_path = examples_dir / record["filename"]

        image = Image.open(image_path).convert("RGB")

        image.thumbnail(
            (tile_width - 20, tile_height - 20),
            Image.Resampling.LANCZOS
        )

        x = (position % columns) * tile_width
        y = (position // columns) * tile_height

        sheet.paste(image, (x + 10, y + 10))

    sheet.save(output_path)


contact_sheet_path = (
    RUN_DIR / "recognition_examples_contact_sheet.jpg"
)

create_contact_sheet(
    example_records,
    EXAMPLES_DIR,
    contact_sheet_path
)

print("Recognition contact sheet saved:", contact_sheet_path)


# ============================================================
# 20. SAVE METRICS JSON
# ============================================================

metrics["recognition_examples"] = example_records

metrics["evaluation_timestamp"] = datetime.now().isoformat()

metrics_json_path = RUN_DIR / "test_metrics.json"

with open(
    metrics_json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 21. SAVE HUMAN-READABLE SUMMARY
# ============================================================

summary_path = RUN_DIR / "test_evaluation_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write("EXP-TROCR-VIN-002 - TEST EVALUATION SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    file.write("EXPERIMENT\n")
    file.write(f"Name: {EXPERIMENT_NAME}\n")
    file.write(f"Checkpoint: {BEST_MODEL_DIR}\n")
    file.write(f"Test manifest: {TEST_MANIFEST}\n")
    file.write(f"Device: {DEVICE}\n")
    file.write(f"GPU: {metrics['gpu_name']}\n\n")

    file.write("TEST RECOGNITION METRICS\n")
    file.write("-" * 40 + "\n")

    for key in [
        "test_images",
        "correct_vins",
        "incorrect_vins",
        "accuracy_percent",
        "exact_string_accuracy",
        "character_accuracy_percent",
        "character_error_rate_CER",
        "word_string_error_rate",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "AUC_ROC",
        "test_loss_teacher_forced"
    ]:

        file.write(f"{key}: {metrics.get(key)}\n")

    file.write("\nINFERENCE PERFORMANCE\n")
    file.write("-" * 40 + "\n")

    for key in [
        "inference_total_seconds",
        "inference_latency_ms_per_image_mean",
        "inference_latency_ms_per_image_median_amortised",
        "inference_latency_ms_per_image_p95_amortised",
        "batch_latency_ms_mean",
        "batch_latency_ms_median",
        "batch_latency_ms_p95",
        "inference_throughput_images_per_second"
    ]:

        file.write(f"{key}: {metrics.get(key)}\n")

    file.write("\nGPU MEMORY\n")
    file.write("-" * 40 + "\n")

    for key in [
        "gpu_total_memory_gib",
        "gpu_peak_allocated_gib",
        "gpu_peak_reserved_gib"
    ]:

        file.write(f"{key}: {metrics.get(key)}\n")

    file.write("\nEVALUATION INTEGRITY\n")
    file.write("-" * 40 + "\n")
    file.write("Test set used for training: NO\n")
    file.write("Test set used for model selection: NO\n")
    file.write("Model weights updated during evaluation: NO\n")

    file.write("\nOUTPUT FILES\n")
    file.write("-" * 40 + "\n")
    file.write(f"Predictions CSV: {prediction_csv}\n")
    file.write(f"Recognition examples: {EXAMPLES_DIR}\n")
    file.write(f"Contact sheet: {contact_sheet_path}\n")
    file.write(f"Metrics JSON: {metrics_json_path}\n")


# ============================================================
# 22. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)

print(f"Test images: {metrics['test_images']}")
print(f"Correct VINs: {metrics['correct_vins']}")
print(f"Incorrect VINs: {metrics['incorrect_vins']}")

print(
    "Exact-string accuracy: "
    f"{metrics['accuracy_percent']:.2f}%"
)

print(
    "Character accuracy: "
    f"{metrics['character_accuracy_percent']:.2f}%"
)

print(
    "CER: "
    f"{metrics['character_error_rate_CER']:.6f}"
)

print(
    "String error rate: "
    f"{metrics['word_string_error_rate']:.6f}"
)

print("Precision:", metrics["precision_micro"])
print("Recall:", metrics["recall_micro"])
print("F1:", metrics["f1_micro"])
print("AUC-ROC:", metrics["AUC_ROC"])

print("\nInference time:", metrics["inference_total_seconds"], "seconds")

print(
    "Mean latency per image:",
    metrics["inference_latency_ms_per_image_mean"],
    "ms"
)

print(
    "Throughput:",
    metrics["inference_throughput_images_per_second"],
    "images/second"
)

print("\nResults folder:")
print(RUN_DIR)

print("\nRecognition contact sheet:")
print(contact_sheet_path)

print("\nEvaluation completed successfully.")
