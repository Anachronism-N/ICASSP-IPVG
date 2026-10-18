# Wan2.2 Training Infrastructure Setup

## Overview
Set up complete training infrastructure for Wan2.2-TI2V-5B (text+image-to-video) using DiffSynth-Studio community framework.

## Model
- **Model**: Wan2.2-TI2V-5B (5B parameters, text+image-to-video)
- **Source**: ModelScope (pending download)
- **Resolution**: 1280x704, 24fps

## Training Framework
- **Framework**: DiffSynth-Studio (community framework for Wan2.2 LoRA/full training)
- **Method**: LoRA (rank 32, recommended for single-GPU training)
- **Precision**: bf16
- **GPU Requirement**: 1x A100 40GB (LoRA), 4x A100 80GB (full finetune)

## Configuration Files

### training_metadata/wan22_training_config.json
Complete training config:
- Model: Wan2.2-TI2V-5B
- LoRA rank: 32
- Learning rate: 1e-4
- Batch size: 1
- Gradient accumulation: 4
- Max sequence length: 24 frames
- Dataset metadata and hyperparameters

### training_metadata/accelerate_config.yaml
Accelerate config for single-GPU bf16 training.

### scripts/train_wan22.sh
Main training launch script with flags:
- `--smoke`: Run smoke test with 5,476 samples
- `--dataset PATH`: Specify dataset path
- `--lora-rank N`: Set LoRA rank (default: 32)

### scripts/smoke_test_wan22.sh
Smoke test with 5,476 samples (5,000 HOIGen + 476 Seamless Interaction).

### scripts/convert_manifest_to_diffsynth.py
Converts JSONL manifest to DiffSynth-Studio CSV format for training.

## Training Data Ready
| Dataset | Videos | Status |
|---------|--------|--------|
| HOIGen-1M | 106,100 | Extracted, captioned |
| Seamless Interaction | 476 | Extracted, captioned |
| DexYCB | 2,400 | Converted, needs captioning |
| H2O | 60 | Converted, needs captioning |
| Charades | 9,848 | Captions fixed |
| MIRA | 2.1M+ | Captions fixed, cached |
| NoXi | TBD | Downloaded, needs processing |
| OpenVidHD | TBD | Downloaded, needs processing |

## Pending
1. Download Wan2.2-TI2V-5B model from ModelScope
2. Run smoke test: `bash scripts/smoke_test_wan22.sh`
3. VLM captioning for remaining datasets (Qwen3-VL-235B deployed, GPUs occupied)

## GPU Requirements for VLM Captioning
- Qwen3-VL-235B-A22B: 4 GPUs per replica (TP=4), 8 replicas across 4 nodes
- Currently deployed but GPUs occupied by other workloads
