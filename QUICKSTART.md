# Quick Start Guide - Super APL Learning Model

## Running the Executables

### Option 1: Standard GUI (SuperAPLModel.exe)
```
dist/SuperAPLModel.exe
```

### Option 2: Duck Personality Variant (Duck.exe)
```
dist/Duck.exe
```

Both come pre-configured with C++ engine support and fallback to Python emulator if DLL unavailable.

## Using the GUI

### 1. Load Data
- **Method A (CSV Files)**:
  - Click "Load Weights" to load weights matrix W from CSV
  - Click "Load Input (CSV)" to load input matrix A from CSV
  
- **Method B (Manual Entry)**:
  ```
  A:
  1.0 2.0 3.0
  4.0 5.0 6.0

  W:
  1.0 0.0
  0.0 1.0
  1.0 1.0
  ```

### 2. Configure Expression
Default: `Result ← Input +.× Weights`

Supported operators:
- `+.x` or `+.×` (matrix multiply / inner product)
- `phi` or `φ` (transpose)

### 3. Run Inference
Click "RUN INFERENCE" to execute.

The system automatically:
1. Tries C++ engine (native computation)
2. Falls back to Python emulator if needed
3. Reports computation time and results

## CSV File Format

### Simple Matrix
```
1.0, 2.0, 3.0
4.0, 5.0, 6.0
```

### With Headers (Optional)
```
A:
1.0 2.0
3.0 4.0

W:
5.0 6.0
7.0 8.0
```

## System Architecture

```
┌─────────────────────┐
│   GUI (Tkinter)     │
├─────────────────────┤
│   APLEmulator       │
│   (Python/NumPy)    │
├─────────────────────┤
│ C++ Engine (DLL)    │ ← ctypes binding
│ SimpleMatrixMultiply│
└─────────────────────┘
```

## File Structure

```
dist/
  ├── SuperAPLModel.exe        ← Main executable
  ├── Duck.exe                 ← Personality variant
  └── super_apl_engine.dll     ← C++ compute engine
```

## Testing

To validate the system:
```
python test_integration.py
```

Expected output:
```
[Test] CSV Loading ✓
[Test] Engine Binding ✓
[Test] Matrix Multiply ✓
All integration tests passed!
```

## Troubleshooting

### "Engine not available"
- System automatically falls back to Python emulator
- All computation still works correctly
- Check `build/Release/super_apl_engine.dll` exists

### CSV load error
- Ensure file is comma or space-separated
- Check for non-numeric values
- Try simpler format (just numbers, no headers)

### Matrix dimension mismatch
- For matrix multiply (A @ W): A columns must equal W rows
- Example: (2×3) @ (3×2) → (2×2) ✓

## Command Line Usage

Create a Python script:
```python
import sys
sys.path.insert(0, 'src/gui')
from app import APLEmulator

apl = APLEmulator()
result = apl.execute_apl_expression("Result +.× W", "A:\n1 2 3\n4 5 6\n\nW:\n1 0\n0 1\n1 1")
print(result)
```

## Building from Source

### Build C++ Engine
```
build_engine.bat
```

### Build SuperAPLModel.exe
```
build_exe.bat
```

### Build Duck.exe
```
build_duck.bat
```

## Specifications

- **Language**: Python 3.12 (GUI), C++17 (engine), CUDA (optional)
- **Memory**: ~50MB (Python + DLL)
- **Platform**: Windows 10+ (x64)
- **Dependencies**: None (self-contained)

## Documentation

See `INTEGRATION_COMPLETE.md` for detailed technical information.
