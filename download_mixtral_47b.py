#!/usr/bin/env python3
"""
Download Mixtral 8x7B (47B params) Model
Direct download from Hugging Face Hub
"""

import os
import sys
import torch
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install transformers torch")
    sys.exit(1)

print("="*80)
print("MIXTRAL 8x7B (47B Parameters) DOWNLOAD")
print("="*80)

model_id = "mistralai/Mixtral-8x7B-v0.1"

print(f"\nModel: {model_id}")
print(f"Size: ~47GB (unquantized)")
print(f"Description: Mixture of Experts - Most powerful Mistral model\n")

# GPU info
if torch.cuda.is_available():
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
else:
    print(f"⚠️  CUDA not available\n")

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"Cache location: {cache_dir}\n")

print("Starting download...")
print("(This will take 30-60 minutes depending on internet speed)\n")

try:
    print("[1] Loading config...")
    config = AutoConfig.from_pretrained(model_id)
    print(f"    ✓ Params: ~47B, Layers: {config.num_hidden_layers}, Heads: {config.num_attention_heads}")
    
    print("\n[2] Downloading and loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print(f"    ✓ Model loaded!")
    
    print("\n[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"    ✓ Tokenizer loaded")
    
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS! Mixtral 8x7B downloaded!")
    print(f"{'='*80}\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    if "gated" in str(e).lower() or "token" in str(e).lower():
        print("\nThis model may require HuggingFace token:")
        print("  1. Go to: https://huggingface.co/settings/tokens")
        print("  2. Create read token")
        print("  3. Run: huggingface-cli login")
    sys.exit(1)
