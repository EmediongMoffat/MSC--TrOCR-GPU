
import json
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Project configuration
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw"
TRAIN_IMAGES = DATA_ROOT / "train"
TRAIN_ANNOTATIONS = DATA_ROOT / "train_gt"
OUTPUT_DIR = ROOT / "results" / "EXP-TROCR-001"
CROP_DIR = OUTPUT_DIR / "crops"

CHECKPOINT = "microsoft/trocr-base-printed"
EXPERIMENT_ID = "EXP-TROCR-001"


def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def parse_annotation(line):
    # ICDAR2015 format:
    # x1,y1,x2,y2,x3,y3,x4,y4,transcription
    parts = line.strip().split(",", 8)

    if len(parts) != 9:
        return None

    try:
        coordinates = [int(value) for value in parts[:8]]
    except ValueError:
        return None

    transcription = parts[8].strip()

    # Ignore ICDAR do-not-care regions.
    if not transcription or transcription == "###":
        return None

    points = [
        (coordinates[i], coordinates[i + 1])
        for i in range(0, 8, 2)
    ]

    return points, transcription


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    # Find the first image with at least one valid text annotation.
    selected = None

    for annotation_path in sorted(
        TRAIN_ANNOTATIONS.glob("gt_img_*.txt"),
        key=lambda path: int(path.stem.replace("gt_img_", "")),
    ):
        image_number = annotation_path.stem.replace("gt_img_", "")
        image_path = TRAIN_IMAGES / f"img_{image_number}.jpg"

        if not image_path.exists():
            continue

        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            parsed = parse_annotation(line)

            if parsed is not None:
                points, transcription = parsed
                selected = (
                    image_path,
                    annotation_path,
                    line_number,
                    points,
                    transcription,
                )
                break

        if selected is not None:
            break

    if selected is None:
        raise RuntimeError(
            "No matching image and valid text annotation were found."
        )

    image_path, annotation_path, line_number, points, ground_truth = selected

    # Make a rectangular crop around the annotated quadrilateral.
    with Image.open(image_path) as source:
        scene = source.convert("RGB")
        width, height = scene.size

    x_min = max(0, min(point[0] for point in points))
    y_min = max(0, min(point[1] for point in points))
    x_max = min(width, max(point[0] for point in points) + 1)
    y_max = min(height, max(point[1] for point in points) + 1)

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Annotation produced an invalid crop.")

    crop = scene.crop((x_min, y_min, x_max, y_max))
    crop_path = CROP_DIR / "ground_truth_region.png"
    crop.save(crop_path)

    print("Experiment:", EXPERIMENT_ID)
    print("Git commit:", get_git_commit())
    print("Python/PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    )
    print("Checkpoint:", CHECKPOINT)
    print("Scene image:", image_path.name)
    print("Annotation:", annotation_path.name)
    print("Annotation line:", line_number)
    print("Ground truth:", ground_truth)
    print("Crop size:", crop.size)
    print("Crop saved:", crop_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading pretrained processor and model...")
    load_start = time.perf_counter()

    processor = TrOCRProcessor.from_pretrained(CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()

    model_load_seconds = time.perf_counter() - load_start

    inputs = processor(crop, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Warm-up is recorded separately from the measured inference.
    with torch.inference_mode():
        _ = model.generate(pixel_values, max_new_tokens=64)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = time.perf_counter() - inference_start
    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    record = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "Single-crop pretrained TrOCR smoke test",
        "git_commit": get_git_commit(),
        "dataset": "ICDAR2015 training split",
        "image": str(image_path.relative_to(ROOT)),
        "annotation": str(annotation_path.relative_to(ROOT)),
        "annotation_line": line_number,
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not available"
        ),
        "crop_method": "Axis-aligned bounding rectangle from ground-truth quadrilateral",
        "crop_path": str(crop_path.relative_to(ROOT)),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "model_load_seconds": model_load_seconds,
        "inference_seconds_after_one_warmup": inference_seconds,
        "recognition_accuracy_claimed": False,
        "notes": (
            "Single-region smoke test only. Not a dataset benchmark. "
            "No detection performance is measured."
        ),
    }

    result_path = OUTPUT_DIR / "smoke_test.json"
    result_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- SMOKE TEST RESULT ---")
    print("Ground truth :", ground_truth)
    print("Prediction   :", prediction)
    print(f"Model load   : {model_load_seconds:.3f} seconds")
    print(f"Inference    : {inference_seconds:.3f} seconds")
    print("Result saved :", result_path)


if __name__ == "__main__":
    main()
