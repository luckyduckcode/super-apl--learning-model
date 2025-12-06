#!/usr/bin/env python3
"""
Manual Offloading Training for Mixtral 8x7B
This approach bypasses accelerate's broken offloading by implementing
our own layer-by-layer disk offloading with custom 4-bit Linear layers.
Supports Training via Gradient Checkpointing and Autograd Hooks.
"""

import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
import gc
import torch.utils.checkpoint

# Configuration
MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = Path("manual_offload")

print("[*] Manual Offload Training Approach...")
print(f"   Model: {MODEL_ID}")
print(f"   Offload Dir: {OFFLOAD_DIR}\n")

class OffloadTrigger(torch.autograd.Function):
    """
    Custom autograd function to trigger weight offloading AFTER the backward pass
    of the layer is complete.
    """
    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # This is called after the gradients have passed through the layer
        # (since this function was applied to the INPUT of the layer)
        # So the layer's backward is done. We can offload.
        if ctx.module.is_loaded:
            # print(f"DEBUG: Offloading {ctx.module.layer_name} after backward")
            ctx.module.offload_weights()
        return grad_output, None

# Custom Offloaded Layer
class OffloadedLinear4bit(nn.Module):
    """
    A linear layer that stores its quantized weights on disk and loads them
    on-demand during the forward pass. 
    
    For Training (with Gradient Checkpointing):
    1. Forward (no_grad): Loads weights, computes, offloads immediately.
    2. Backward (re-run forward with grad): Loads weights, registers hook, computes.
       Hook offloads weights after backward pass is done.
    """
    def __init__(self, in_features, out_features, bias=False, layer_name=""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = layer_name
        self.has_bias = bias
        self.bias_data = None
        
        self.weight_path = OFFLOAD_DIR / f"{layer_name}.pt"
        self.is_loaded = False
        self.bnb_layer = None

    def load_weights(self):
        """Load quantized weights from disk to GPU."""
        if self.is_loaded:
            return
        
        if not self.weight_path.exists():
            # Layer not quantized (possibly norm or embed - should not happen for Linear)
            return

        # print(f"DEBUG: Loading {self.layer_name}")
        data = torch.load(self.weight_path, map_location="cpu")
        q_weight = data["q_weight"].to("cuda")
        state = data["state"]
        
        # Create BnB layer on GPU
        if self.bnb_layer is None:
            self.bnb_layer = bnb.nn.Linear4bit(
                self.in_features, 
                self.out_features, 
                bias=self.has_bias,
                quant_type="nf4",
                compress_statistics=True,
                compute_dtype=torch.float16
            ).to("cuda")
            
        # Assign weights
        self.bnb_layer.weight.data = q_weight
        self.bnb_layer.weight.quant_state = state
        
        if self.has_bias and self.bias_data is not None:
            self.bnb_layer.bias.data = self.bias_data.to("cuda")
            
        self.is_loaded = True

    def offload_weights(self):
        """Offload weights from GPU to free memory."""
        if not self.is_loaded:
            return
        
        # Clear GPU memory
        self.bnb_layer.weight.data = torch.empty(0)
        self.bnb_layer = None
        self.is_loaded = False
        torch.cuda.empty_cache()

    def forward(self, x):
        """Forward pass with automatic weight loading/offloading."""
        self.load_weights()
        
        if self.bnb_layer is None:
            raise RuntimeError(f"Failed to load weights for {self.layer_name}")
            
        # If training (grad enabled), register hook to offload AFTER backward
        if torch.is_grad_enabled():
            x = OffloadTrigger.apply(x, self)
            out = self.bnb_layer(x)
            # Do NOT offload here. Hook will do it.
        else:
            # Inference or Checkpoint First Pass (no_grad)
            out = self.bnb_layer(x)
            self.offload_weights() # Immediate offload
            
        return out

def replace_linear_with_offloaded(module, name=""):
    """Recursively replace all nn.Linear layers with OffloadedLinear4bit."""
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear):
            offloaded = OffloadedLinear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                layer_name=full_name
            )
            # Save bias if present
            if child.bias is not None:
                offloaded.bias_data = child.bias.data.clone().cpu()
            setattr(module, child_name, offloaded)
        else:
            replace_linear_with_offloaded(child, full_name)

# 1. Create Skeleton
print("[*] Creating model skeleton...")
config = AutoConfig.from_pretrained(MODEL_ID)
# Enable gradient checkpointing in config if possible, or on model later
config.use_cache = False # Disable cache for training
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print("   ✓ Skeleton created\n")

# 2. Replace Linear Layers
print("[*] Replacing Linear layers with OffloadedLinear4bit...")
replace_linear_with_offloaded(model)
print("   ✓ Layers replaced\n")

# 3. Load Base Weights (Norms, Embeddings, etc.)
print("[*] Loading base weights (norms, embeddings)...")
base_weights_path = OFFLOAD_DIR / "base_weights.pt"

if not base_weights_path.exists():
    print(f"❌ Base weights not found at {base_weights_path}")
    print("   Run `python setup_manual_offload.py` first to quantize the model.")
    sys.exit(1)

base_weights = torch.load(base_weights_path)
print(f"   Loaded {len(base_weights)} tensors")

# Assign to model
for name, param in model.named_parameters():
    if name in base_weights:
        parent = model
        path = name.split(".")
        for p in path[:-1]:
            parent = getattr(parent, p)
        new_param = nn.Parameter(base_weights[name].to("cuda"), requires_grad=True)
        setattr(parent, path[-1], new_param)

for name, buffer in model.named_buffers():
    if name in base_weights:
        parent = model
        path = name.split(".")
        for p in path[:-1]:
            parent = getattr(parent, p)
        parent.register_buffer(path[-1], base_weights[name].to("cuda"))

print("   ✓ Base weights loaded\n")

# 4. Enable Gradient Checkpointing
print("[*] Enabling Gradient Checkpointing...")
model.gradient_checkpointing_enable()
print("   ✓ Gradient Checkpointing enabled\n")

# 5. Tokenizer
print("[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("   ✓ Tokenizer loaded\n")

# 6. Test Training Step
print("[*] Testing Training Step (Forward + Backward)...")
model.train()
text = "Mixtral is a large language model that requires offloading."
inputs = tokenizer(text, return_tensors="pt").to("cuda")
labels = inputs.input_ids.clone()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

try:
    print("   > Forward pass...")
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    print(f"   ✓ Forward pass successful! Loss: {loss.item():.4f}")
    
    print("   > Backward pass...")
    loss.backward()
    print("   ✓ Backward pass successful!")
    
    print("   > Optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    print("   ✓ Optimizer step successful!")
    
except Exception as e:
    print(f"   ❌ Training step failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*80)
print("Manual offloading training setup complete!")
print("You can now implement your full training loop.")
print("="*80)
