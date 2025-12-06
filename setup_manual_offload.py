import os
import sys
import torch
import gc
import glob
import json
from pathlib import Path
from transformers import AutoConfig
import bitsandbytes as bnb
from safetensors import safe_open
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configuration
MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = Path("manual_offload")
OFFLOAD_DIR.mkdir(exist_ok=True)

print("[*] Starting Manual Offload Setup...")

if not torch.cuda.is_available():
    print("❌ CUDA required for quantization.")
    sys.exit(1)

# Fast Quantization Helper
def fast_quantize(weight_tensor):
    """Quantizes a tensor to 4-bit NF4 using GPU."""
    # Move to GPU
    w_gpu = weight_tensor.to("cuda", dtype=torch.float16)
    # Quantize
    q_weight, state = bnb.functional.quantize_4bit(
        w_gpu, 
        quant_type="nf4", 
        compress_statistics=True
    )
    # Move back to CPU
    return q_weight.cpu(), state

# 1. Download Model
print("[*] Locating model checkpoint...")
# checkpoint_path = snapshot_download(MODEL_ID)
# Use existing path to avoid re-downloading or fetching .pt files
checkpoint_path = r"C:\Users\tenna\.cache\huggingface\hub\models--mistralai--Mixtral-8x7B-v0.1\snapshots\fc7ac94680e38d7348cfa806e51218e6273104b0"
print(f"   ✓ Checkpoint at {checkpoint_path}")

# 2. Process Shards
print("[*] Processing shards and quantizing to disk...")
shard_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))


# Store non-linear weights in a single dict (if they fit in RAM)
base_weights = {}
base_weights_path = OFFLOAD_DIR / "base_weights.pt"
if base_weights_path.exists():
    print("   [*] Loading existing base weights...")
    base_weights = torch.load(base_weights_path)

for shard in tqdm(shard_files, desc="Processing Shards"):
    # Use safe_open to iterate without loading full file to RAM
    with safe_open(shard, framework="pt", device="cpu") as f:
        for name in f.keys():
            # Check if we already have this weight (for base weights)
            if name in base_weights:
                continue

            # Check if it's a Linear layer weight that needs quantization
            is_linear_weight = any(x in name for x in ["proj", "w1", "w2", "w3", "lm_head"]) and "weight" in name
            
            if is_linear_weight:
                layer_name = name.replace(".weight", "")
                save_path = OFFLOAD_DIR / f"{layer_name}.pt"
                
                if save_path.exists():
                    continue # Skip if already processed
                
                # Load tensor only if needed
                tensor = f.get_tensor(name)
                
                if tensor.dim() != 2:
                    # Unexpected shape for linear weight?
                    if name not in base_weights:
                        base_weights[name] = tensor.cpu()
                    continue

                try:
                    q_weight, state = fast_quantize(tensor)
                    torch.save({"q_weight": q_weight, "state": state}, save_path)
                    del q_weight, state
                except Exception as e:
                    print(f"❌ Error quantizing {name}: {e}")
                
                del tensor
            else:
                # Keep other tensors (norm, embed, bias)
                if name not in base_weights:
                    tensor = f.get_tensor(name)
                    base_weights[name] = tensor.cpu()
                    del tensor
            
    gc.collect()
    torch.cuda.empty_cache()

# Save base weights
print("[*] Saving base weights...")
torch.save(base_weights, base_weights_path)
print("   ✓ Setup complete.")
