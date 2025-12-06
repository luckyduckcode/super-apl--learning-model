# Mixtral 47B Inference Guide

## Overview
We have successfully configured the **Mixtral 8x7B (47B parameter)** model to run on your **NVIDIA RTX 3060 (12GB VRAM)**.

This is achieved using **Ultra-Extreme Offloading**:
- **GPU VRAM**: Used only for active computation (one layer at a time).
- **System RAM**: Used as a buffer (limited to 4GB to prevent crashes).
- **Disk (SSD)**: Stores the bulk of the 90GB model weights.

## How to Run
To chat with the model, open a terminal and run:

```powershell
python chat_mixtral_47b.py
```

## Performance Expectations
- **Startup Time**: ~1-2 minutes (to load the offload map).
- **Generation Speed**: Slow. The model must read weights from disk for *every single token* generated.
- **Stability**: High. By disabling the training adapter (which was causing crashes), the base model runs reliably.

## Troubleshooting
- If you see `OOM` errors, ensure no other heavy applications are running.
- If the generation hangs, be patient. Disk I/O is the bottleneck.

## Note on "Duck" Personality
Since we had to skip the training adapter to ensure stability, the model is currently the **base Mixtral model**. It has vast general knowledge but hasn't been fine-tuned on the specific "Duck" personality yet.
