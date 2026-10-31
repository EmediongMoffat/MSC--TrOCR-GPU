
import os
import csv
import json
import math
import time
import random
import shutil
from pathlib import Path

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "EXP-TROCR-VIN-001.yaml"

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

        # Preserve aspect ratio and pad to the configured size.
        image = ImageOps.pad(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5)
        )

        # Do not allow the pretrained processor to resize to 384x384.
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

        # Ignore padding tokens when calculating the training loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

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


def calculate_metrics(references, predictions):

    exact_matches = 0
    total_edit_errors = 0
    total_reference_characters = 0

    for reference, prediction in zip(references, predictions):

        if reference == prediction:
            exact_matches += 1

        total_edit_errors += edit_distance(
            reference, prediction
        )

        total_reference_characters += len(reference)

    number_of_samples = len(references)

    exact_accuracy = (
        exact_matches / number_of_samples
        if number_of_samples else 0
    )

    string_error_rate = 1.0 - exact_accuracy

    cer = (
        total_edit_errors / total_reference_characters
        if total_reference_characters else 0
    )

    character_accuracy = 1.0 - cer

    return {
        "exact_string_accuracy": exact_accuracy,
        "string_error_rate": string_error_rate,
        "character_accuracy": character_accuracy,
        "CER": cer
    }


# ============================================================
# 5. LOAD PROCESSOR AND PRETRAINED MODEL
# ============================================================

print("\n" + "=" * 65)
print("TROCR VIN TRAINING")
print("=" * 65)

print("Project:", PROJECT_ROOT)
print("Experiment:", EXPERIMENT)
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

# TrOCR configuration required for training.
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Transformers 5.x: generation settings belong in generation_config.
model.generation_config.max_length = GENERATION_LENGTH
model.generation_config.num_beams = GENERATION_BEAMS

model.config.use_cache = False

# Allow the ViT encoder to interpolate positional embeddings
# for our 768 x 128 input rather than its original 384 x 384 input.
model.to(DEVICE)

# Reduce activation memory during training where supported.
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

print("Model loaded successfully.")


# ============================================================
# 6. LOAD TRAINING AND VALIDATION DATA
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


# ============================================================
# 7. OPTIMISER AND LEARNING RATE SCHEDULER
# ============================================================

if cfg["training"]["optimizer"].lower() != "adamw":
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
        return float(current_step + 1) / float(warmup_steps)

    remaining = total_training_steps - current_step

    decay_steps = max(1, total_training_steps - warmup_steps)

    return max(0.0, float(remaining) / float(decay_steps))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)


# ============================================================
# 8. CHECKPOINT FUNCTIONS
# ============================================================

def checkpoint_files():

    files = list(CHECKPOINT_DIR.glob("checkpoint-epoch-*.pt"))

    return sorted(
        files,
        key=lambda p: int(
            p.stem.split("-")[-1]
        )
    )


def save_checkpoint(epoch, best_val_loss, best_epoch):

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"checkpoint-epoch-{epoch:03d}.pt"
    )

    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
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

    # Retain the latest N complete checkpoints.
    saved_files = checkpoint_files()

    while len(saved_files) > KEEP_LAST:

        old_file = saved_files.pop(0)

        old_file.unlink()

        print("Removed older checkpoint:", old_file.name)


def load_latest_checkpoint():

    files = checkpoint_files()

    if not files:
        return 0, float("inf"), 0

    checkpoint_path = files[-1]

    print("\nResuming from:", checkpoint_path)

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])

    if (
        torch.cuda.is_available()
        and state["cuda_rng_state"] is not None
    ):
        torch.cuda.set_rng_state_all(
            state["cuda_rng_state"]
        )

    completed_epoch = int(state["epoch"])

    best_val_loss = float(state["best_val_loss"])
    best_epoch = int(state["best_epoch"])

    print("Completed epochs:", completed_epoch)
    print("Best validation loss:", best_val_loss)

    return completed_epoch, best_val_loss, best_epoch


# ============================================================
# 9. VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    references = []
    predictions = []

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

            generated_ids = model.generate(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True
            )

            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            predictions.extend(
                [text.strip() for text in decoded]
            )

            references.extend(batch["vins"])

    model.config.use_cache = False

    val_loss = total_loss / max(1, total_samples)

    metrics = calculate_metrics(
        references,
        predictions
    )

    return val_loss, metrics


# ============================================================
# 10. SAVE BEST MODEL AND METRICS
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

    processor.save_pretrained(BEST_MODEL_DIR)

    print("Best model saved to:", BEST_MODEL_DIR)


METRICS_FILE = OUTPUT_DIR / "epoch_metrics.csv"

METRIC_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "validation_loss",
    "exact_string_accuracy",
    "character_accuracy",
    "CER",
    "string_error_rate",
    "epoch_training_time_seconds",
    "total_training_time_seconds"
]

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
# 11. RESUME OR START TRAINING
# ============================================================

start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0

if RESUME:
    start_epoch, best_val_loss, best_epoch = (
        load_latest_checkpoint()
    )

if cfg["reproducibility"]["save_config_copy_with_results"]:

    shutil.copy2(
        CONFIG_PATH,
        OUTPUT_DIR / "trocr_vin_config_used.yaml"
    )

start_time = time.time()
epochs_without_improvement = 0



# ============================================================
# 12. MAIN TRAINING LOOP
# ============================================================

for epoch in range(start_epoch, EPOCHS):

    epoch_number = epoch + 1
    epoch_start = time.time()

    model.train()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    total_train_samples = 0

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

            # Determine the actual number of batches in this
            # gradient accumulation group.
            group_start = (step // GRAD_ACCUM) * GRAD_ACCUM

            group_size = min(
                GRAD_ACCUM,
                len(train_loader) - group_start
            )

            # Divide by the actual group size.
            # This correctly handles the final incomplete group.
            scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            is_accumulation_step = (
                (step + 1) % GRAD_ACCUM == 0
            )

            is_last_batch = (
                step + 1 == len(train_loader)
            )

            if is_accumulation_step or is_last_batch:

                # Unscale only after all gradients in this group
                # have been accumulated.
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

        # Validate after every epoch.
        val_loss, metrics = validate()

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEPOCH RESULTS")
        print("-" * 45)
        print(f"Training loss:          {train_loss:.6f}")
        print(f"Validation loss:        {val_loss:.6f}")
        print(
            f"Exact-string accuracy:  "
            f"{metrics['exact_string_accuracy'] * 100:.2f}%"
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
        print(f"Learning rate:          {current_lr:.8f}")
        print(f"Epoch time:             {epoch_time:.2f} seconds")

        # Save epoch metrics.
        if cfg["reproducibility"]["save_epoch_metrics_csv"]:

            row = {
                "epoch": epoch_number,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "exact_string_accuracy":
                    metrics["exact_string_accuracy"],
                "character_accuracy":
                    metrics["character_accuracy"],
                "CER": metrics["CER"],
                "string_error_rate":
                    metrics["string_error_rate"],
                "epoch_training_time_seconds": epoch_time,
                "total_training_time_seconds": elapsed_time
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

        # Save best model based on validation loss.
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch_number
            epochs_without_improvement = 0

            if cfg["checkpoints"]["keep_best_model_separately"]:
                save_best_model()

            print("New best validation loss.")

        else:

            epochs_without_improvement += 1

        # Save full resumable checkpoint after every epoch.
        if cfg["checkpoints"]["save_after_every_epoch"]:

            save_checkpoint(
                epoch_number,
                best_val_loss,
                best_epoch
            )

        print("\nBest epoch:", best_epoch)
        print("Best validation loss:", best_val_loss)

        # Early stopping.
        if (
            EARLY_STOPPING
            and epochs_without_improvement >= PATIENCE
        ):

            print(
                "\nEarly stopping: validation loss did not "
                f"improve for {PATIENCE} epochs."
            )

            break

    except torch.cuda.OutOfMemoryError:

        print("\nCUDA OUT OF MEMORY ERROR.")
        print("Training stopped safely.")
        print("Try reducing batch_size in the YAML file to 1.")
        print("Then restart with resume_from_latest_if_available: true.")

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        raise

# ============================================================
# 13. FINAL TRAINING SUMMARY
# ============================================================

total_time = time.time() - start_time

summary = {
    "experiment": EXPERIMENT,
    "pretrained_model": MODEL_ID,
    "completed_epochs": epoch_number,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "total_training_time_seconds": total_time,
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE
}

with open(
    OUTPUT_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("TRAINING FINISHED")
print("=" * 65)
print(json.dumps(summary, indent=4))

print("\nMetrics file:", METRICS_FILE)
print("Checkpoint folder:", CHECKPOINT_DIR)
print("Best model folder:", BEST_MODEL_DIR)
