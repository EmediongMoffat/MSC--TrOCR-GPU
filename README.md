# MSc VIN OCR Project — TrOCR on GPU

## Project purpose
This project investigates AI-based optical character recognition (OCR) for Vehicle Identification Numbers (VINs), following the OVST MSc AI Development Guide.

## Current stage
Environment setup and GPU verification.

## Hardware and software
- Operating system: Windows
- Python: 3.12
- PyTorch: 2.14.1+cu126
- PyTorch CUDA runtime: 12.6
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- GPU availability: Verified using `torch.cuda.is_available()`
- GPU computation: A 1000 × 1000 matrix multiplication completed successfully.

## Experiment records
- `EXP-TROCR-000`: Initial environment inspection and GPU verification.
- `EXP-TROCR-001`: Reserved for the first pretrained TrOCR inference test.

## Reproducibility
- Direct dependencies will be pinned in `requirements.txt`.
- Complete installed-package snapshots will be stored in `requirements-lock.txt`.
- Experiment logs, configurations, code versions, model checkpoints, and evaluation results will be maintained according to the OVST development guide.

## Data and model weights
Datasets and model checkpoints are excluded from Git. Do not commit confidential OVST or partner data.
