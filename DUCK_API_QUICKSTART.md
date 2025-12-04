# Duck API - Quick Start Guide

Duck now has a complete API with three interfaces: **Python Module**, **CLI**, and **REST**.

## 🚀 Quick Start

### Python Module (Simplest)
```python
from src.api.duck_api import DuckAPI

duck = DuckAPI()
result = duck.matrix_multiply([[1, 2, 3]], [[1], [2], [3]])
print(result['result'])  # [[14]]
```

### CLI (Command Line)
```bash
python src/api/duck_api.py multiply --input A.csv --weights W.csv
```

### REST (Web Service)
```bash
# Start server
python src/api/duck_api.py server

# In another terminal
curl -X POST http://localhost:5000/api/v1/matrix-multiply \
  -H "Content-Type: application/json" \
  -d '{"A": [[1,2,3]], "W": [[1], [2], [3]]}'
```

---

## 📚 Complete Documentation

See **[DUCK_API.md](DUCK_API.md)** for comprehensive API reference including:
- All Python methods with examples
- REST endpoint documentation
- CLI command reference
- Error handling
- Performance notes

---

## 🎯 Common Tasks

### Task 1: Multiply Two Matrices (Python)
```python
from src.api.duck_api import DuckAPI

duck = DuckAPI()

A = [[1, 2, 3], [4, 5, 6]]
W = [[1, 0], [0, 1], [1, 1]]

result = duck.matrix_multiply(A, W)
print(result['result'])  # [[4, 5], [10, 11]]
```

### Task 2: Transpose Matrix (CLI)
```bash
python src/api/duck_api.py transpose --input matrix.csv --output transposed.json
```

### Task 3: Batch Process (Python)
```python
from src.api.duck_api import DuckAPI

duck = DuckAPI()

datasets = [
    {"A": [[1, 2]], "W": [[3], [4]]},
    {"A": [[5, 6]], "W": [[7], [8]]}
]

results = duck.batch_compute(datasets)
for r in results:
    print(f"Item {r['index']}: {r['result']}")
```

### Task 4: Load CSV and Compute (Python)
```python
from src.api.duck_api import DuckAPI

duck = DuckAPI()

# Load from CSV files
data = duck.load_csv('input.csv', 'weights.csv')

# Compute
result = duck.matrix_multiply(data['data']['A'], data['data']['W'])
print(result['result'])
```

### Task 5: Start Web Service (REST)
```bash
# Start server on port 5000
python src/api/duck_api.py server

# Or custom port
python src/api/duck_api.py server --port 8000

# Check status
curl http://localhost:5000/api/v1/status
```

---

## 📋 API Methods

### Python Module

| Method | Purpose | Example |
|--------|---------|---------|
| `matrix_multiply(A, W)` | Multiply two matrices | `duck.matrix_multiply([[1,2]], [[3],[4]])` |
| `transpose(A)` | Transpose matrix | `duck.transpose([[1,2],[3,4]])` |
| `compute(A, W, expr)` | Custom APL expression | `duck.compute(A, W, "Result ← Input +.× Weights")` |
| `batch_compute(data)` | Process multiple matrices | `duck.batch_compute([{...}, {...}])` |
| `load_csv(path_a, path_w)` | Load from CSV files | `duck.load_csv('A.csv', 'W.csv')` |
| `get_status()` | Get API status | `duck.get_status()` |
| `get_personality()` | Get Duck personality | `duck.get_personality()` |

### REST Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/status` | API status |
| GET | `/api/v1/personality` | Duck personality config |
| POST | `/api/v1/compute` | Custom expression |
| POST | `/api/v1/matrix-multiply` | Matrix multiply |
| POST | `/api/v1/transpose` | Transpose |
| POST | `/api/v1/batch` | Batch processing |

### CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `multiply` | Matrix multiply | `duck_api.py multiply --input A.csv --weights W.csv` |
| `transpose` | Transpose | `duck_api.py transpose --input A.csv` |
| `compute` | Custom expression | `duck_api.py compute --input A.csv --expr "..."` |
| `server` | REST server | `duck_api.py server --port 5000` |
| `status` | API status | `duck_api.py status` |
| `personality` | Duck config | `duck_api.py personality` |

---

## 🔧 Installation

### Requirements
```bash
pip install numpy flask
```

### Verify Installation
```bash
python src/api/duck_api.py status
```

Expected output:
```json
{
  "status": "online",
  "api_version": "1.0.0",
  "model_name": "Duck (Super APL Model)",
  "cpp_engine_available": true,
  "python_emulator": true
}
```

---

## 📖 Examples

Run example scripts:
```bash
# List all examples
python src/api/duck_api_examples.py

# Run specific example
python src/api/duck_api_examples.py 1    # Simple Python usage
python src/api/duck_api_examples.py 5    # Batch processing
python src/api/duck_api_examples.py 12   # REST with Python
```

---

## 🌐 REST API Examples

### Start Server
```bash
python src/api/duck_api.py server &
sleep 2
```

### Check Status
```bash
curl http://localhost:5000/api/v1/status
```

### Matrix Multiply
```bash
curl -X POST http://localhost:5000/api/v1/matrix-multiply \
  -H "Content-Type: application/json" \
  -d '{
    "A": [[1, 2, 3], [4, 5, 6]],
    "W": [[1, 0], [0, 1], [1, 1]]
  }'
```

### Transpose
```bash
curl -X POST http://localhost:5000/api/v1/transpose \
  -H "Content-Type: application/json" \
  -d '{"A": [[1, 2, 3], [4, 5, 6]]}'
```

### Batch Processing
```bash
curl -X POST http://localhost:5000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
      {"A": [[4, 5, 6]], "W": [[1], [1], [1]]}
    ]
  }'
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
```bash
pip install flask
```

### "API is offline"
Make sure the server is running:
```bash
python src/api/duck_api.py server
```

### "C++ Engine not available"
The API will automatically fall back to Python emulator. This is fine and all computations will still work.

### Dimension Mismatch Error
Make sure matrix dimensions are compatible:
- For multiply: A (M×K) @ W (K×N) → (M×N)
- For transpose: (M×N) → (N×M)

---

## 📊 Performance

| Operation | Time | Engine |
|-----------|------|--------|
| Small matrix (2×3 @ 3×2) | <1ms | C++ or Python |
| CSV load | ~5ms | Python |
| Transpose | <1ms | Python |
| Batch 100 items | ~50ms | Mixed |

---

## 🔐 API Security Notes

- REST API runs on `127.0.0.1:5000` by default (localhost only)
- For external access, use `--host 0.0.0.0`
- No authentication by default (add if deploying publicly)
- Input validation performed on matrix dimensions

---

## 📝 Response Format

All endpoints return JSON with consistent structure:

### Success Response
```json
{
  "status": "success",
  "result": [[...], [...]],
  "shape": [2, 2],
  "engine": "C++ Native",
  "time_ms": 0.5
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Dimension mismatch",
  "error_type": "ValueError"
}
```

---

## 🚀 Next Steps

1. **Read Full Documentation**: [DUCK_API.md](DUCK_API.md)
2. **Run Examples**: `python src/api/duck_api_examples.py`
3. **Try REST Server**: `python src/api/duck_api.py server`
4. **Integrate into Your App**: Import `DuckAPI` and use Python module interface

---

## 📞 Support

- See [DUCK_API.md](DUCK_API.md) for comprehensive documentation
- Check [DUCK_API_EXAMPLES.md](src/api/duck_api_examples.py) for usage patterns
- Review error messages for debugging tips

---

## ✨ Features

✅ **Three Interfaces**: Python module, CLI, REST API
✅ **Batch Processing**: Process multiple matrices at once
✅ **CSV Support**: Load matrices from CSV files
✅ **C++ Acceleration**: Uses native engine when available, falls back to Python
✅ **Error Handling**: Comprehensive error messages with types
✅ **Performance Metrics**: Timing data for each operation
✅ **Duck Personality**: R2-D2 humor + C-3PO versatility
✅ **Fully Documented**: Complete API reference with examples

---

**Version**: 1.0.0
**Last Updated**: December 2024
