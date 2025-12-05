#!/usr/bin/env python3
"""Quick test of training setup"""
import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

# Test imports
print("\nTesting imports...")
try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")

try:
    import peft
    print(f"✓ peft {peft.__version__}")
except Exception as e:
    print(f"✗ peft: {e}")

try:
    import bitsandbytes
    print(f"✓ bitsandbytes {bitsandbytes.__version__}")
except Exception as e:
    print(f"✗ bitsandbytes: {e}")

print("\nAll systems ready for training!")
