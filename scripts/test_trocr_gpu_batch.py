
from pathlib import Path
import yaml
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --------------------------------------------------
# 1. Read experiment configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "EXP-TROCR-VIN-001.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODEL_ID = cfg["model"]["pretrained_checkpoint"]
WIDTH = int(cfg["model"]["image_width"])
HEIGHT = int(cfg["model"]["image_height"])
MAX_LENGTH = int(cfg["model"]["max_target_length"])

DATA_ROOT = ROOT / cfg["data"]["dataset_root"]
MANIFEST = DATA_ROOT / cfg["data"]["train_manifest"]

BATCH_SIZE = int(cfg["data"]["batch_size"])
GRAD_ACCUM = int(cfg["training"]["gradient_accumulation_steps"])
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM

LR = float(cfg["training"]["learning_rate"])
WEIGHT_DECAY = float(cfg["training"]["weight_decay"])
GRAD_CLIP = float(cfg["training"]["gradient_clipping"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = (
    DEVICE.type == "cuda"
    and cfg["training"]["mixed_precision"] == "fp16"
)

if BATCH_SIZE < 1 or GRAD_ACCUM < 1:
    raise ValueError("Batch size and gradient accumulation must be at least 1.")

print("=" * 65)
print("TROCR GPU TRAINING-BATCH TEST")
print("=" * 65)

print("Experiment:", cfg["project"]["experiment_name"])
print("Device:", DEVICE)
print("Configured batch size:", BATCH_SIZE)
print("Gradient accumulation steps:", GRAD_ACCUM)
print("Effective batch size:", EFFECTIVE_BATCH_SIZE)
print("Image dimensions:", f"{WIDTH} x {HEIGHT}")
print("Mixed precision:", cfg["training"]["mixed_precision"])

if DEVICE.type != "cuda":
    raise RuntimeError("CUDA is unavailable. GPU test cannot proceed.")

print("\nGPU:", torch.cuda.get_device_name(0))

total_gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)

print("Total GPU memory:", round(total_gpu_memory, 2), "GB")

# --------------------------------------------------
# 2. Read first batch from training manifest
# --------------------------------------------------

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Training manifest not found: {MANIFEST}"
    )

samples = []

with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:

        line = line.rstrip("\r\n")

        if not line:
            continue

        parts = line.split("\t", 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid manifest row: {line}")

        image_path = DATA_ROOT / parts[0].strip()
        vin = parts[1].strip()

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if len(vin) != 17:
            raise ValueError(f"Invalid VIN length: {vin}")

        samples.append((image_path, vin))

        if len(samples) == BATCH_SIZE:
            break

if len(samples) != BATCH_SIZE:
    raise RuntimeError(
        f"Expected {BATCH_SIZE} training samples, "
        f"but found {len(samples)}."
    )

print("\nTraining samples loaded:", len(samples))

# --------------------------------------------------
# 3. Load processor and prepare image batch
# --------------------------------------------------

print("\nLoading TrOCR processor...")

processor = TrOCRProcessor.from_pretrained(MODEL_ID)

pixel_values_list = []
vins = []

for image_path, vin in samples:

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image = ImageOps.pad(
        image,
        (WIDTH, HEIGHT),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5)
    )

    pixels = processor.image_processor(
        images=image,
        return_tensors="pt",
        do_resize=False
    ).pixel_values.squeeze(0)

    pixel_values_list.append(pixels)
    vins.append(vin)

pixel_values = torch.stack(pixel_values_list)

tokenized = processor.tokenizer(
    vins,
    padding="max_length",
    max_length=MAX_LENGTH,
    truncation=True,
    return_tensors="pt"
)

labels = tokenized.input_ids

labels[
    labels == processor.tokenizer.pad_token_id
] = -100

print("Image tensor shape:", tuple(pixel_values.shape))
print("Label tensor shape:", tuple(labels.shape))
print("Ground-truth VINs:", vins)

# --------------------------------------------------
# 4. Load pretrained model
# --------------------------------------------------

print("\nLoading pretrained TrOCR weights...")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.use_cache = False

model.to(DEVICE)
model.train()

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

pixel_values = pixel_values.to(DEVICE)
labels = labels.to(DEVICE)

print("Model loaded on:", DEVICE)

# --------------------------------------------------
# 5. Prepare optimiser and reset GPU memory tracking
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    foreach=False
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_FP16
)

optimizer.zero_grad(set_to_none=True)

# Measure peak memory for this test only.
torch.cuda.reset_peak_memory_stats()

# --------------------------------------------------
# 6. Forward pass
# --------------------------------------------------

print("\nStarting GPU forward pass...")

with torch.autocast(
    device_type="cuda",
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

print("Forward pass successful.")
print("Training loss:", round(loss.item(), 6))

# --------------------------------------------------
# 7. Backward pass
# --------------------------------------------------

print("\nStarting backward pass...")

scaler.scale(loss).backward()

print("Backward pass successful.")

# --------------------------------------------------
# 8. Optimiser update
# --------------------------------------------------

print("\nPerforming one optimiser update...")

scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    GRAD_CLIP
)

scaler.step(optimizer)
scaler.update()

optimizer.zero_grad(set_to_none=True)

print("Optimiser update successful.")

# --------------------------------------------------
# 9. GPU memory report
# --------------------------------------------------

torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

print("\n" + "=" * 65)
print("GPU BATCH TEST RESULTS")
print("=" * 65)

print("Experiment:", cfg["project"]["experiment_name"])
print("GPU:", torch.cuda.get_device_name(0))

print("\nBATCH CONFIGURATION")
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation:", GRAD_ACCUM)
print("Effective batch size:", EFFECTIVE_BATCH_SIZE)

print("\nMODEL INPUT")
print("Image dimensions:", f"{WIDTH} x {HEIGHT}")
print("Image tensor:", tuple(pixel_values.shape))
print("Label tensor:", tuple(labels.shape))

print("\nTRAINING TEST")
print("Training loss:", round(loss.item(), 6))
print("Forward pass: SUCCESS")
print("Backward pass: SUCCESS")
print("Optimiser update: SUCCESS")

print("\nGPU MEMORY")
print("Total GPU memory:", round(total_gpu_memory, 3), "GB")
print("Currently allocated:", round(allocated, 3), "GB")
print("Currently reserved:", round(reserved, 3), "GB")
print("Peak allocated:", round(peak_allocated, 3), "GB")
print("Peak reserved:", round(peak_reserved, 3), "GB")

print("\n" + "=" * 65)
print("TEST PASSED: One GPU training batch completed.")
print("=" * 65)
