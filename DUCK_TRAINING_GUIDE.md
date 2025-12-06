# Duck Training Guide: Mixtral 47B on Retail Hardware

## Overview
This guide documents the successful configuration for training the **Mixtral 8x7B (47B Parameters)** model on a consumer PC with an **NVIDIA RTX 3060 (12GB VRAM)** and **16GB System RAM**.

## The Challenge
Training a 47B parameter model typically requires 80GB+ of VRAM (A100 GPUs). Even with 4-bit quantization (QLoRA), the model is ~26GB, which exceeds the 12GB VRAM of the RTX 3060 and the 16GB System RAM.

## The Solution: "Patched Offloading"
We achieved this by combining:
1.  **4-bit Quantization (NF4)**: Reduces model size from 90GB to 26GB.
2.  **Disk Offloading**: Uses the SSD as "slow RAM" to store layers that don't fit in VRAM/RAM.
3.  **Library Patching**: Fixed a critical bug in `bitsandbytes` that prevented offloading of 4-bit models.

## Configuration

### 1. Hardware Requirements
- **GPU**: NVIDIA RTX 3060 (12GB) or better.
- **RAM**: 16GB minimum (32GB recommended).
- **Disk**: SSD with >100GB free space (for model weights + offload buffer).
- **OS**: Windows 10/11 or Linux.

### 2. Software Setup
- **Python**: 3.10+
- **Libraries**: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `peft`.

### 3. The Critical Patch
The `bitsandbytes` library has a bug where it tries to access data from "meta tensors" during offloading, causing a crash.
**File**: `site-packages\bitsandbytes\functional.py`
**Location**: Inside `as_dict` method (approx line 525).
**Change**:
```python
# OLD
"nested_offset": self.offset.item(),

# NEW
"nested_offset": self.offset.item() if self.offset.device.type != "meta" else 0,
```

### 4. Training Script Settings
Use `train_partitioned_mixtral.py` with these settings:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

max_memory = {0: "11.5GB", "cpu": "12GB"} # Leave room for OS

model = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map="auto",
    offload_folder="model_offload", # Must be enabled!
    max_memory=max_memory
)
```

## Performance
- **Loading Time**: ~5 minutes (using SSD offloading).
- **Training Speed**: Slow (due to disk I/O swapping layers), but **functional**.
- **Stability**: Stable with the patch.

## Troubleshooting
- **OOM (Out of Memory)**: Reduce `cpu` limit in `max_memory` to force more offloading to disk.
- **Meta Tensor Error**: Re-apply the `bitsandbytes` patch.
