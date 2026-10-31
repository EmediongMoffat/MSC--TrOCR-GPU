
import os
import csv
import json
import math
import time
import random
import shutil
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import torch

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "EXP-TROCR-VIN-002.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["project"]["experiment_name"]
SEED = int(cfg["project"]["seed"])

MODEL_ID = cfg["model"]["pretrained_checkpoint"]
IMAGE_WIDTH = int(cfg["model"]["image_width"])
IMAGE_HEIGHT = int(cfg["model"]["image_height"])
MAX_TARGET_LENGTH = int(cfg["model"]["max_target_length"])

DATA_ROOT = PROJECT_ROOT / cfg["data"]["dataset_root"]

TRAIN_MANIFEST = DATA_ROOT / cfg["data"]["train_manifest"]
VAL_MANIFEST = DATA_ROOT / cfg["data"]["validation_manifest"]

BATCH_SIZE = int(cfg["data"]["batch_size"])
NUM_WORKERS = int(cfg["data"]["num_workers"])

EPOCHS = int(cfg["training"]["epochs"])
LEARNING_RATE = float(cfg["training"]["learning_rate"])
WEIGHT_DECAY = float(cfg["training"]["weight_decay"])
GRAD_ACCUM = int(cfg["training"]["gradient_accumulation_steps"])
GRAD_CLIP = float(cfg["training"]["gradient_clipping"])
MIXED_PRECISION = cfg["training"]["mixed_precision"]
WARMUP_RATIO = float(cfg["training"]["warmup_ratio"])

EARLY_STOPPING = bool(cfg["training"]["early_stopping"])
PATIENCE = int(cfg["training"]["early_stopping_patience"])

# Development-guide early stopping threshold.
MIN_DELTA = 0.0001

GENERATION_BEAMS = int(cfg["evaluation"]["generation_num_beams"])
GENERATION_LENGTH = int(cfg["evaluation"]["generation_max_length"])

OUTPUT_DIR = PROJECT_ROOT / cfg["checkpoints"]["output_directory"]
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

KEEP_LAST = int(cfg["checkpoints"]["keep_last_n_checkpoints"])
RESUME = bool(cfg["checkpoints"]["resume_from_latest_if_available"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda" and MIXED_PRECISION == "fp16"

if BATCH_SIZE < 1 or GRAD_ACCUM < 1:
    raise ValueError("Batch size and gradient accumulation must be >= 1.")

if EXPERIMENT != "EXP-TROCR-VIN-002":
    raise ValueError(
        f"Expected EXP-TROCR-VIN-002, but configuration says {EXPERIMENT}"
    )

if not RESUME:
    existing_checkpoints = list(
        CHECKPOINT_DIR.glob("checkpoint-epoch-*.pt")
    )

    if existing_checkpoints:
        raise FileExistsError(
            "Checkpoints already exist but resume is disabled. "
            "Enable resume or use a new output directory."
        )


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if cfg["reproducibility"]["deterministic"]:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DATASET
# ============================================================

class VINDataset(Dataset):

    def __init__(self, manifest_path, dataset_root, processor):

        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.samples = []

        if not Path(manifest_path).exists():
            raise FileNotFoundError(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.rstrip("\r\n")

                if not line:
                    continue

                parts = line.split("\t", 1)

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest line {line_number}: {line}"
                    )

                image_relative_path = parts[0].strip()
                vin = parts[1].strip()

                image_path = self.dataset_root / image_relative_path

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image missing: {image_path}"
                    )

                if len(vin) != 17:
                    raise ValueError(
                        f"VIN must contain 17 characters. "
                        f"Found {len(vin)}: {vin}"
                    )

                self.samples.append((image_path, vin))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {manifest_path}"
            )

        print(
            f"Loaded {len(self.samples)} samples from "
            f"{Path(manifest_path).name}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, vin = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

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
            "vin": vin
        }


def collate_batch(batch):

    return {
        "pixel_values": torch.stack(
            [item["pixel_values"] for item in batch]
        ),
        "labels": torch.stack(
            [item["labels"] for item in batch]
        ),
        "vins": [item["vin"] for item in batch]
    }


# ============================================================
# 4. OCR METRICS
# ============================================================

def edit_distance(reference, prediction):

    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):

        current = [i]

        for j, pred_char in enumerate(prediction, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                ref_char != pred_char
            )

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def aligned_character_counts(reference, prediction):

    m = len(reference)
    n = len(prediction)

    dp = [
        [0] * (n + 1)
        for _ in range(m + 1)
    ]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            cost = (
                0
                if reference[i - 1] == prediction[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    tp = Counter()
    fp = Counter()
    fn = Counter()

    correct_characters = 0

    i = m
    j = n

    while i > 0 or j > 0:

        if i > 0 and j > 0:

            cost = (
                0
                if reference[i - 1] == prediction[j - 1]
                else 1
            )

            if dp[i][j] == dp[i - 1][j - 1] + cost:

                ref_char = reference[i - 1]
                pred_char = prediction[j - 1]

                if ref_char == pred_char:

                    tp[ref_char] += 1
                    correct_characters += 1

                else:

                    fp[pred_char] += 1
                    fn[ref_char] += 1

                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:

            fn[reference[i - 1]] += 1
            i -= 1
            continue

        if j > 0:

            fp[prediction[j - 1]] += 1
            j -= 1

    return tp, fp, fn, correct_characters


def calculate_auc_roc(binary_labels, confidence_scores):

    if confidence_scores is None:
        return None

    if len(binary_labels) != len(confidence_scores):
        return None

    positives = sum(bool(value) for value in binary_labels)
    negatives = len(binary_labels) - positives

    if positives == 0 or negatives == 0:
        return None

    order = sorted(
        range(len(confidence_scores)),
        key=lambda index: confidence_scores[index]
    )

    positive_rank_sum = 0.0
    i = 0

    while i < len(order):

        j = i + 1

        while (
            j < len(order)
            and confidence_scores[order[j]]
            == confidence_scores[order[i]]
        ):
            j += 1

        average_rank = ((i + 1) + j) / 2.0

        for position in range(i, j):

            sample_index = order[position]

            if binary_labels[sample_index]:
                positive_rank_sum += average_rank

        i = j

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    return float(auc)


def calculate_metrics(references, predictions, confidence_scores=None):

    if len(references) != len(predictions):
        raise ValueError(
            "References and predictions have different lengths."
        )

    if not references:
        raise ValueError(
            "Cannot calculate metrics on an empty dataset."
        )

    exact_matches = 0
    total_edit_errors = 0
    total_reference_characters = 0
    total_correct_characters = 0

    total_tp = Counter()
    total_fp = Counter()
    total_fn = Counter()

    for reference, prediction in zip(references, predictions):

        if reference == prediction:
            exact_matches += 1

        total_edit_errors += edit_distance(
            reference,
            prediction
        )

        total_reference_characters += len(reference)

        tp, fp, fn, correct = aligned_character_counts(
            reference,
            prediction
        )

        total_tp.update(tp)
        total_fp.update(fp)
        total_fn.update(fn)

        total_correct_characters += correct

    number_of_samples = len(references)

    exact_accuracy = exact_matches / number_of_samples

    string_error_rate = 1.0 - exact_accuracy

    cer = (
        total_edit_errors / total_reference_characters
        if total_reference_characters else 0.0
    )

    character_accuracy = (
        total_correct_characters / total_reference_characters
        if total_reference_characters else 0.0
    )

    micro_tp = sum(total_tp.values())
    micro_fp = sum(total_fp.values())
    micro_fn = sum(total_fn.values())

    precision = (
        micro_tp / (micro_tp + micro_fp)
        if micro_tp + micro_fp else 0.0
    )

    recall = (
        micro_tp / (micro_tp + micro_fn)
        if micro_tp + micro_fn else 0.0
    )

    f1_micro = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    character_classes = (
        set(total_tp)
        | set(total_fp)
        | set(total_fn)
    )

    class_f1_scores = []

    for character in character_classes:

        tp = total_tp[character]
        fp = total_fp[character]
        fn = total_fn[character]

        class_precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        class_recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        class_f1 = (
            2 * class_precision * class_recall
            / (class_precision + class_recall)
            if class_precision + class_recall else 0.0
        )

        class_f1_scores.append(class_f1)

    f1_macro = (
        sum(class_f1_scores) / len(class_f1_scores)
        if class_f1_scores else 0.0
    )

    binary_labels = [
        reference == prediction
        for reference, prediction in zip(references, predictions)
    ]

    auc_roc = calculate_auc_roc(
        binary_labels,
        confidence_scores
    )

    return {
        "accuracy_percent": exact_accuracy * 100.0,
        "exact_string_accuracy": exact_accuracy,
        "string_error_rate": string_error_rate,
        "character_accuracy": character_accuracy,
        "CER": cer,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "AUC_ROC": auc_roc
    }


def calculate_sequence_confidences(generation_output, tokenizer):

    sequences = generation_output.sequences
    scores = generation_output.scores

    batch_size = sequences.size(0)

    token_confidences = [
        [] for _ in range(batch_size)
    ]

    special_token_ids = set(tokenizer.all_special_ids)

    for step, step_scores in enumerate(scores):

        sequence_position = step + 1

        if sequence_position >= sequences.size(1):
            break

        generated_token_ids = sequences[:, sequence_position]

        probabilities = torch.softmax(
            step_scores.float(),
            dim=-1
        )

        selected_probabilities = probabilities.gather(
            1,
            generated_token_ids.unsqueeze(1)
        ).squeeze(1)

        for sample_index in range(batch_size):

            token_id = int(
                generated_token_ids[sample_index].item()
            )

            if token_id not in special_token_ids:

                token_confidences[sample_index].append(
                    float(
                        selected_probabilities[sample_index].item()
                    )
                )

    return [
        float(np.mean(values)) if values else 0.0
        for values in token_confidences
    ]


# ============================================================
# 5. RESOURCE MEASUREMENT FUNCTIONS
# ============================================================

def synchronise_device():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def reset_gpu_peak_memory():

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory():

    if DEVICE.type != "cuda":

        return {
            "gpu_peak_allocated_gb": None,
            "gpu_peak_reserved_gb": None,
            "gpu_total_memory_gb": None
        }

    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()

    total = torch.cuda.get_device_properties(
        0
    ).total_memory

    return {
        "gpu_peak_allocated_gb": allocated / (1024 ** 3),
        "gpu_peak_reserved_gb": reserved / (1024 ** 3),
        "gpu_total_memory_gb": total / (1024 ** 3)
    }


def get_directory_size_bytes(directory):

    directory = Path(directory)

    if not directory.exists():
        return 0

    return sum(
        file.stat().st_size
        for file in directory.rglob("*")
        if file.is_file()
    )


def get_model_weight_size_bytes(directory):

    directory = Path(directory)

    if not directory.exists():
        return 0

    weight_extensions = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth"
    }

    return sum(
        file.stat().st_size
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in weight_extensions
    )


def calculate_inference_summary(batch_times, total_images):

    if not batch_times or total_images == 0:

        return {
            "inference_total_seconds": None,
            "inference_latency_ms_per_image_mean": None,
            "inference_latency_ms_per_image_median": None,
            "inference_latency_ms_per_image_p95": None,
            "inference_throughput_images_per_second": None
        }

    total_seconds = sum(batch_times)

    per_image_times = [
        batch_time * 1000.0
        for batch_time in batch_times
    ]

    mean_latency = total_seconds * 1000.0 / total_images

    return {
        "inference_total_seconds": total_seconds,
        "inference_latency_ms_per_image_mean": mean_latency,
        "inference_latency_ms_per_image_median":
            float(np.median(per_image_times)),
        "inference_latency_ms_per_image_p95":
            float(np.percentile(per_image_times, 95)),
        "inference_throughput_images_per_second":
            total_images / total_seconds if total_seconds else None
    }


# ============================================================
# 6. LOAD PROCESSOR AND PRETRAINED MODEL
# ============================================================

print("\n" + "=" * 65)
print("TROCR VIN TRAINING")
print("=" * 65)

print("Project:", PROJECT_ROOT)
print("Experiment:", EXPERIMENT)
print("Configuration:", CONFIG_PATH)
print("Device:", DEVICE)
print("Pretrained model:", MODEL_ID)

if DEVICE.type == "cuda":

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

print("\nLoading pretrained processor...")

processor = TrOCRProcessor.from_pretrained(MODEL_ID)

print("Loading pretrained TrOCR weights...")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

tokenizer = processor.tokenizer

model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.generation_config.max_length = GENERATION_LENGTH
model.generation_config.num_beams = GENERATION_BEAMS

model.config.use_cache = False

model.to(DEVICE)

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

print("Model loaded successfully.")

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


# ============================================================
# 7. LOAD TRAINING AND VALIDATION DATA
# ============================================================

train_dataset = VINDataset(
    TRAIN_MANIFEST,
    DATA_ROOT,
    processor
)

val_dataset = VINDataset(
    VAL_MANIFEST,
    DATA_ROOT,
    processor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=(DEVICE.type == "cuda")
)

effective_batch_size = BATCH_SIZE * GRAD_ACCUM

print("\nDATASET SUMMARY")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation:", GRAD_ACCUM)
print("Nominal effective batch size:", effective_batch_size)


# ============================================================
# 8. OPTIMISER AND LEARNING RATE SCHEDULER
# ============================================================

OPTIMIZER_NAME = cfg["training"]["optimizer"]

if OPTIMIZER_NAME.lower() != "adamw":
    raise ValueError("This script currently supports AdamW only.")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=False
)

updates_per_epoch = math.ceil(
    len(train_loader) / GRAD_ACCUM
)

total_training_steps = updates_per_epoch * EPOCHS
warmup_steps = int(total_training_steps * WARMUP_RATIO)


def lr_lambda(current_step):

    if warmup_steps > 0 and current_step < warmup_steps:

        return (
            float(current_step + 1)
            / float(warmup_steps)
        )

    remaining = total_training_steps - current_step

    decay_steps = max(
        1,
        total_training_steps - warmup_steps
    )

    return max(
        0.0,
        float(remaining) / float(decay_steps)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
)

scaler = torch.cuda.amp.GradScaler(
    enabled=USE_FP16
)


# ============================================================
# 9. CHECKPOINT FUNCTIONS
# ============================================================

def checkpoint_files():

    files = list(
        CHECKPOINT_DIR.glob("checkpoint-epoch-*.pt")
    )

    return sorted(
        files,
        key=lambda p: int(p.stem.split("-")[-1])
    )


def save_checkpoint(
    epoch,
    best_val_loss,
    best_epoch,
    epochs_without_improvement,
    elapsed_training_seconds
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"checkpoint-epoch-{epoch:03d}.pt"
    )

    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "elapsed_training_seconds": elapsed_training_seconds,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
    }

    torch.save(state, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)

    saved_files = checkpoint_files()

    while len(saved_files) > KEEP_LAST:

        old_file = saved_files.pop(0)
        old_file.unlink()

        print("Removed older checkpoint:", old_file.name)


def load_latest_checkpoint():

    files = checkpoint_files()

    if not files:

        return (
            0,
            float("inf"),
            0,
            0,
            0.0
        )

    checkpoint_path = files[-1]

    print("\nResuming from:", checkpoint_path)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        state["model_state_dict"]
    )

    optimizer.load_state_dict(
        state["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        state["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        state["scaler_state_dict"]
    )

    random.setstate(
        state["python_rng_state"]
    )

    np.random.set_state(
        state["numpy_rng_state"]
    )

    torch.set_rng_state(
        state["torch_rng_state"]
    )

    if (
        torch.cuda.is_available()
        and state.get("cuda_rng_state") is not None
    ):

        torch.cuda.set_rng_state_all(
            state["cuda_rng_state"]
        )

    completed_epoch = int(state["epoch"])

    best_val_loss = float(
        state["best_val_loss"]
    )

    best_epoch = int(
        state["best_epoch"]
    )

    epochs_without_improvement = int(
        state.get("epochs_without_improvement", 0)
    )

    elapsed_training_seconds = float(
        state.get("elapsed_training_seconds", 0.0)
    )

    print("Completed epochs:", completed_epoch)
    print("Best validation loss:", best_val_loss)

    return (
        completed_epoch,
        best_val_loss,
        best_epoch,
        epochs_without_improvement,
        elapsed_training_seconds
    )


# ============================================================
# 10. VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    references = []
    predictions = []
    confidence_scores = []
    batch_inference_times = []

    model.config.use_cache = True

    with torch.no_grad():

        for batch in val_loader:

            pixel_values = batch["pixel_values"].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True
            )

            batch_size = pixel_values.size(0)

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

            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size

            synchronise_device()
            inference_start = time.perf_counter()

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_FP16
            ):

                generation_output = model.generate(
                    pixel_values=pixel_values,
                    interpolate_pos_encoding=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=GENERATION_LENGTH,
                    num_beams=GENERATION_BEAMS
                )

            synchronise_device()

            inference_seconds = (
                time.perf_counter() - inference_start
            )

            batch_inference_times.append(
                inference_seconds
            )

            decoded = tokenizer.batch_decode(
                generation_output.sequences,
                skip_special_tokens=True
            )

            predictions.extend(
                [text.strip() for text in decoded]
            )

            references.extend(batch["vins"])

            confidence_scores.extend(
                calculate_sequence_confidences(
                    generation_output,
                    tokenizer
                )
            )

    model.config.use_cache = False

    val_loss = (
        total_loss / max(1, total_samples)
    )

    metrics = calculate_metrics(
        references,
        predictions,
        confidence_scores
    )

    inference_summary = calculate_inference_summary(
        batch_inference_times,
        total_samples
    )

    return (
        val_loss,
        metrics,
        inference_summary
    )


# ============================================================
# 11. SAVE BEST MODEL AND MODEL SIZE
# ============================================================

def save_best_model():

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        BEST_MODEL_DIR,
        safe_serialization=True
    )

    processor.save_pretrained(
        BEST_MODEL_DIR
    )

    print("Best model saved to:", BEST_MODEL_DIR)


# ============================================================
# 12. METRICS CSV
# ============================================================

METRICS_FILE = OUTPUT_DIR / "epoch_metrics.csv"

METRIC_FIELDS = [
    "epoch",
    "learning_rate",
    "batch_size",
    "effective_batch_size",
    "optimizer",
    "loss_function",
    "train_loss",
    "validation_loss",
    "accuracy_percent",
    "exact_string_accuracy",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "f1_macro",
    "character_accuracy",
    "CER",
    "string_error_rate",
    "AUC_ROC",
    "epoch_training_time_seconds",
    "total_training_time_seconds",
    "inference_total_seconds",
    "inference_latency_ms_per_image_mean",
    "inference_latency_ms_per_image_median",
    "inference_latency_ms_per_image_p95",
    "inference_throughput_images_per_second",
    "gpu_peak_allocated_gb",
    "gpu_peak_reserved_gb"
]

if METRICS_FILE.exists() and not RESUME:

    raise FileExistsError(
        f"{METRICS_FILE} already exists. "
        "Enable resume or use a clean experiment directory."
    )

if not METRICS_FILE.exists():

    with open(
        METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=METRIC_FIELDS
        )

        writer.writeheader()


# ============================================================
# 13. RESUME OR START TRAINING
# ============================================================

start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
previous_training_seconds = 0.0

if RESUME:

    (
        start_epoch,
        best_val_loss,
        best_epoch,
        epochs_without_improvement,
        previous_training_seconds
    ) = load_latest_checkpoint()

if cfg["reproducibility"]["save_config_copy_with_results"]:

    shutil.copy2(
        CONFIG_PATH,
        OUTPUT_DIR / "trocr_vin_config_used.yaml"
    )

if DEVICE.type == "cuda":
    reset_gpu_peak_memory()

session_start_time = time.perf_counter()

completed_epochs = start_epoch

global_peak_allocated_gb = 0.0
global_peak_reserved_gb = 0.0

if DEVICE.type == "cuda":

    global_peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    global_peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )


# ============================================================
# 14. MAIN TRAINING LOOP
# ============================================================

for epoch in range(start_epoch, EPOCHS):

    epoch_number = epoch + 1
    epoch_start = time.perf_counter()

    if DEVICE.type == "cuda":
        reset_gpu_peak_memory()

    model.train()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    total_train_samples = 0

    # Record the learning rate at the beginning of the epoch.
    epoch_learning_rate = optimizer.param_groups[0]["lr"]

    print("\n" + "=" * 65)
    print(f"EPOCH {epoch_number}/{EPOCHS}")
    print("=" * 65)

    try:

        for step, batch in enumerate(train_loader):

            pixel_values = batch["pixel_values"].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True
            )

            batch_size = pixel_values.size(0)

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

                loss = outputs.loss

            running_loss += loss.item() * batch_size
            total_train_samples += batch_size

            group_start = (
                (step // GRAD_ACCUM) * GRAD_ACCUM
            )

            group_size = min(
                GRAD_ACCUM,
                len(train_loader) - group_start
            )

            scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            is_accumulation_step = (
                (step + 1) % GRAD_ACCUM == 0
            )

            is_last_batch = (
                step + 1 == len(train_loader)
            )

            if is_accumulation_step or is_last_batch:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                scheduler.step()

            if (step + 1) % 10 == 0:

                print(
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        train_loss = (
            running_loss / max(1, total_train_samples)
        )

        val_loss, metrics, inference_summary = validate()

        synchronise_device()

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        elapsed_time = (
            previous_training_seconds
            + time.perf_counter()
            - session_start_time
        )

        gpu_memory = get_gpu_memory()

        if gpu_memory["gpu_peak_allocated_gb"] is not None:

            global_peak_allocated_gb = max(
                global_peak_allocated_gb,
                gpu_memory["gpu_peak_allocated_gb"]
            )

            global_peak_reserved_gb = max(
                global_peak_reserved_gb,
                gpu_memory["gpu_peak_reserved_gb"]
            )

        print("\nEPOCH RESULTS")
        print("-" * 55)

        print(f"Training loss:          {train_loss:.6f}")
        print(f"Validation loss:        {val_loss:.6f}")

        print(
            f"Accuracy:               "
            f"{metrics['accuracy_percent']:.2f}%"
        )

        print(
            f"Exact-string accuracy:  "
            f"{metrics['exact_string_accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision (micro):      "
            f"{metrics['precision_micro']:.6f}"
        )

        print(
            f"Recall (micro):         "
            f"{metrics['recall_micro']:.6f}"
        )

        print(
            f"F1 (micro):             "
            f"{metrics['f1_micro']:.6f}"
        )

        print(
            f"F1 (macro):             "
            f"{metrics['f1_macro']:.6f}"
        )

        print(
            f"Character accuracy:     "
            f"{metrics['character_accuracy'] * 100:.2f}%"
        )

        print(f"CER:                    {metrics['CER']:.6f}")

        print(
            f"String error rate:      "
            f"{metrics['string_error_rate'] * 100:.2f}%"
        )

        print(
            "AUC-ROC:                "
            f"{metrics['AUC_ROC'] if metrics['AUC_ROC'] is not None else 'N/A'}"
        )

        print(f"Learning rate:          {epoch_learning_rate:.8f}")
        print(f"Epoch time:             {epoch_time:.2f} seconds")

        print(
            "Mean inference latency: "
            f"{inference_summary['inference_latency_ms_per_image_mean']:.2f} ms/image"
        )

        print(
            "Inference throughput:   "
            f"{inference_summary['inference_throughput_images_per_second']:.2f} images/sec"
        )

        if DEVICE.type == "cuda":

            print(
                "Peak GPU allocated:     "
                f"{gpu_memory['gpu_peak_allocated_gb']:.3f} GB"
            )

            print(
                "Peak GPU reserved:      "
                f"{gpu_memory['gpu_peak_reserved_gb']:.3f} GB"
            )

        # ----------------------------------------------------
        # SAVE EPOCH METRICS
        # ----------------------------------------------------

        if cfg["reproducibility"]["save_epoch_metrics_csv"]:

            row = {
                "epoch": epoch_number,
                "learning_rate": epoch_learning_rate,
                "batch_size": BATCH_SIZE,
                "effective_batch_size": effective_batch_size,
                "optimizer": OPTIMIZER_NAME,
                "loss_function": "CrossEntropyLoss (ignore_index=-100)",
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "accuracy_percent": metrics["accuracy_percent"],
                "exact_string_accuracy":
                    metrics["exact_string_accuracy"],
                "precision_micro": metrics["precision_micro"],
                "recall_micro": metrics["recall_micro"],
                "f1_micro": metrics["f1_micro"],
                "f1_macro": metrics["f1_macro"],
                "character_accuracy":
                    metrics["character_accuracy"],
                "CER": metrics["CER"],
                "string_error_rate":
                    metrics["string_error_rate"],
                "AUC_ROC": metrics["AUC_ROC"],
                "epoch_training_time_seconds": epoch_time,
                "total_training_time_seconds": elapsed_time,
                **inference_summary,
                "gpu_peak_allocated_gb":
                    gpu_memory["gpu_peak_allocated_gb"],
                "gpu_peak_reserved_gb":
                    gpu_memory["gpu_peak_reserved_gb"]
            }

            with open(
                METRICS_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=METRIC_FIELDS
                )

                writer.writerow(row)

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss - MIN_DELTA:

            best_val_loss = val_loss
            best_epoch = epoch_number
            epochs_without_improvement = 0

            if cfg["checkpoints"]["keep_best_model_separately"]:
                save_best_model()

            print("New best validation loss.")

        else:

            epochs_without_improvement += 1

        completed_epochs = epoch_number

        # ----------------------------------------------------
        # SAVE RESUMABLE CHECKPOINT
        # ----------------------------------------------------

        cumulative_training_seconds = (
            previous_training_seconds
            + time.perf_counter()
            - session_start_time
        )

        if cfg["checkpoints"]["save_after_every_epoch"]:

            save_checkpoint(
                epoch_number,
                best_val_loss,
                best_epoch,
                epochs_without_improvement,
                cumulative_training_seconds
            )

        print("\nBest epoch:", best_epoch)
        print("Best validation loss:", best_val_loss)

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            EARLY_STOPPING
            and epochs_without_improvement >= PATIENCE
        ):

            print(
                "\nEarly stopping: validation loss did not "
                f"improve by at least {MIN_DELTA} for "
                f"{PATIENCE} consecutive epochs."
            )

            break

    except torch.cuda.OutOfMemoryError:

        print("\nCUDA OUT OF MEMORY ERROR.")
        print("Training stopped.")
        print("Try reducing batch_size in the YAML file to 1.")
        print(
            "Then enable resume_from_latest_if_available: true."
        )

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        raise


# ============================================================
# 15. FINAL TRAINING SUMMARY
# ============================================================

total_training_time = (
    previous_training_seconds
    + time.perf_counter()
    - session_start_time
)

final_gpu_memory = get_gpu_memory()

if BEST_MODEL_DIR.exists():

    model_weight_bytes = get_model_weight_size_bytes(
        BEST_MODEL_DIR
    )

    model_bundle_bytes = get_directory_size_bytes(
        BEST_MODEL_DIR
    )

else:

    model_weight_bytes = 0
    model_bundle_bytes = 0

summary = {
    "experiment": EXPERIMENT,
    "pretrained_model": MODEL_ID,
    "completed_epochs": completed_epochs,
    "best_epoch": best_epoch,
    "best_validation_loss": (
        best_val_loss if math.isfinite(best_val_loss) else None
    ),
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "effective_batch_size": effective_batch_size,
    "learning_rate": LEARNING_RATE,
    "optimizer": OPTIMIZER_NAME,
    "loss_function": "CrossEntropyLoss (ignore_index=-100)",
    "training_time_seconds": total_training_time,
    "training_time_minutes": total_training_time / 60.0,
    "device": str(DEVICE),
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else None
    ),
    "gpu_peak_allocated_gb": (
        global_peak_allocated_gb
        if DEVICE.type == "cuda"
        else None
    ),
    "gpu_peak_reserved_gb": (
        global_peak_reserved_gb
        if DEVICE.type == "cuda"
        else None
    ),
    "gpu_total_memory_gb": (
        final_gpu_memory["gpu_total_memory_gb"]
    ),
    "model_parameter_count": parameter_count,
    "model_weights_size_mb": (
        model_weight_bytes / (1024 ** 2)
    ),
    "model_bundle_size_mb": (
        model_bundle_bytes / (1024 ** 2)
    ),
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "test_set_used_during_training": False,
    "early_stopping_min_delta": MIN_DELTA,
    "early_stopping_patience": PATIENCE
}

with open(
    OUTPUT_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)


# Save a readable text summary as well.
with open(
    OUTPUT_DIR / "training_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("TROCR VIN TRAINING SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    for key, value in summary.items():
        f.write(f"{key}: {value}\n")


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINISHED")
print("=" * 65)

print(json.dumps(summary, indent=4))

print("\nMetrics file:", METRICS_FILE)
print("Checkpoint folder:", CHECKPOINT_DIR)
print("Best model folder:", BEST_MODEL_DIR)
print("Training summary:", OUTPUT_DIR / "training_summary.json")
print("Text summary:", OUTPUT_DIR / "training_summary.txt")
