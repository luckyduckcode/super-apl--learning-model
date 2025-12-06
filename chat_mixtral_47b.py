#!/usr/bin/env python3
"""
Chat with the trained Mixtral 47B model.
Uses the same offloading strategy as training to fit on 12GB VRAM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import sys

print("\n" + "="*80)
print("MIXTRAL 47B CHAT INTERFACE")
print("="*80)

model_id = "mistralai/Mixtral-8x7B-v0.1"
adapter_path = "./model_trained/checkpoint-2" # Use the checkpoint saved during training

if not Path(adapter_path).exists():
    print(f"❌ Adapter not found at {adapter_path}")
    print("   Please wait for training to complete.")
    # Try to find any checkpoint
    checkpoints = list(Path("./model_trained").glob("checkpoint-*"))
    if checkpoints:
        adapter_path = str(checkpoints[0])
        print(f"   Found alternative checkpoint: {adapter_path}")
    else:
        sys.exit(1)

print(f"\n[*] Loading base model: {model_id}")
print("    (This will take time due to disk offloading...)")

# Offload directory
offload_dir = Path("./model_offload_chat")
offload_dir.mkdir(exist_ok=True)

# Memory settings (Same as training for stability)
max_memory = {0: "0GB", "cpu": "4GB"}

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder=str(offload_dir),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory=max_memory
    )
    print("   ✓ Base model loaded")

    # Adapter loading disabled for stability testing
    # print(f"[*] Loading LoRA adapter: {adapter_path}")
    # try:
    #     model = PeftModel.from_pretrained(model, adapter_path)
    #     print("   ✓ Adapter loaded")
    # except Exception as e:
    #     print(f"   ⚠️ Failed to load adapter: {e}")
    #     print("   ⚠️ Proceeding with base model only (untrained)...")

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("   ✓ Tokenizer loaded")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("CHAT READY (Type 'quit' to exit)")
print("="*80)

while True:
    try:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            break
            
        if user_input.lower() in ["quit", "exit"]:
            break
        
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        
        print("Duck: ", end="", flush=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(user_input):].strip()
        print(response)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\nGoodbye!")
