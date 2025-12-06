# Extreme Model Offloading Guide

This guide documents how to run and train massive models (like Mixtral 8x7B) on consumer hardware with limited RAM and VRAM.

## The Challenge
Running a 47B parameter model typically requires:
- **FP16**: ~94GB VRAM
- **4-bit**: ~26GB VRAM

Our Hardware:
- **GPU**: RTX 3060 (12GB VRAM)
- **RAM**: 16GB System RAM
- **Disk**: SSD

## The Solution: "The Triple Split"
We split the model across three tiers of storage:
1.  **GPU (12GB)**: Active layers and computation.
2.  **CPU (8GB)**: Fast access storage for upcoming layers.
3.  **Disk (SSD)**: Bulk storage for the rest of the model.

## Required Patches
To make this work, we had to patch the underlying libraries because they assume you have enough RAM to at least hold the quantized model (26GB), which we don't.

### 1. `bitsandbytes` Meta Tensor Fix
Fixes a crash when offloading 4-bit weights.
- **File**: `bitsandbytes/functional.py`
- **Fix**: Check for meta device before accessing `.item()`.

### 2. `accelerate` Offload Bypass
Allows training a 4-bit model even if parts of it are on disk.
- **File**: `accelerate/accelerator.py`
- **Fix**: Comment out the `ValueError` check for 4-bit + offload.

### 3. Fast Quantization Patch
Fixes extreme slowness when loading layers to CPU.
- **File**: `train_partitioned_mixtral.py` (Runtime patch)
- **Fix**: Route CPU quantization tasks through the GPU.

## Configuration
In `train_partitioned_mixtral.py`:
```python
max_memory = {
    0: "11.5GB",    # Maximize GPU usage
    "cpu": "8GB"    # Limit CPU usage to prevent OS crash
}
```
This forces `accelerate` to spill everything else to the `model_offload` folder on disk.

## Running
```bash
python train_partitioned_mixtral.py
```
