# Duck API Documentation

Complete API reference for Duck Super APL Learning Model with REST, CLI, and Python module interfaces.

## Quick Start

### Python Module (Programmatic)
```python
from duck_api import DuckAPI

duck = DuckAPI()

# Simple matrix multiply
result = duck.matrix_multiply([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])
print(result['result'])  # [[4, 5], [10, 11]]
```

### CLI (Command-line)
```bash
# Matrix multiply from CSV files
python duck_api.py multiply --input A.csv --weights W.csv

# Transpose
python duck_api.py transpose --input A.csv

# Start REST server
python duck_api.py server --port 5000
```

### REST API (HTTP)
```bash
# POST request to compute
curl -X POST http://localhost:5000/api/v1/matrix-multiply \
  -H "Content-Type: application/json" \
  -d '{"A": [[1,2,3],[4,5,6]], "W": [[1,0],[0,1],[1,1]]}'
```

---

## Python Module API

### DuckAPI Class

#### `__init__(dll_path=None, personality_path=None)`
Initialize Duck API instance.

**Parameters:**
- `dll_path` (str, optional): Path to super_apl_engine.dll. Auto-detected if None.
- `personality_path` (str, optional): Path to duck_personality.json. Auto-detected if None.

**Example:**
```python
duck = DuckAPI()
duck = DuckAPI(dll_path='path/to/super_apl_engine.dll')
```

#### `compute(A, W=None, expression="Result ← Input +.× Weights")`
Execute an APL expression.

**Parameters:**
- `A` (array-like): Input matrix (M×K)
- `W` (array-like, optional): Weight matrix (K×N)
- `expression` (str): APL expression to evaluate

**Returns:**
```python
{
    "status": "success",
    "result": [[4, 5], [10, 11]],
    "shape": [2, 2],
    "engine": "C++ Native",
    "time_ms": 0.5,
    "expression": "Result ← Input +.× Weights"
}
```

**Example:**
```python
result = duck.compute(
    [[1, 2, 3], [4, 5, 6]],
    [[1, 0], [0, 1], [1, 1]]
)
print(result['result'])  # [[4, 5], [10, 11]]
print(result['engine'])  # "C++ Native" or "Python Emulator"
print(result['time_ms']) # Computation time
```

#### `matrix_multiply(A, W)`
Shortcut for matrix multiplication.

**Parameters:**
- `A` (array-like): Input matrix (M×K)
- `W` (array-like): Weight matrix (K×N)

**Returns:** Same as `compute()`

**Example:**
```python
result = duck.matrix_multiply([[1, 2, 3]], [[1], [2], [3]])
print(result['result'])  # [[14]]
```

#### `transpose(A)`
Transpose a matrix.

**Parameters:**
- `A` (array-like): Input matrix

**Returns:** Same as `compute()`

**Example:**
```python
result = duck.transpose([[1, 2, 3], [4, 5, 6]])
print(result['result'])  # [[1, 4], [2, 5], [3, 6]]
```

#### `batch_compute(data_list, expression="Result ← Input +.× Weights")`
Process multiple matrices in batch.

**Parameters:**
- `data_list` (list): List of dicts with "A" (required) and "W" (optional) keys
- `expression` (str): APL expression to execute for each

**Returns:**
```python
[
    {
        "index": 0,
        "status": "success",
        "result": [...],
        "shape": [...],
        "engine": "C++ Native",
        "time_ms": 0.5
    },
    ...
]
```

**Example:**
```python
data = [
    {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
    {"A": [[4, 5, 6]], "W": [[1], [1], [1]]}
]
results = duck.batch_compute(data)
for r in results:
    print(f"Item {r['index']}: {r['result']}")
```

#### `load_csv(filepath_a, filepath_w=None)`
Load matrices from CSV files.

**Parameters:**
- `filepath_a` (str): Path to CSV file for matrix A
- `filepath_w` (str, optional): Path to CSV file for matrix W

**Returns:**
```python
{
    "status": "success",
    "data": {"A": [...], "W": [...]},
    "A_shape": (2, 3),
    "W_shape": (3, 2)
}
```

**Example:**
```python
data = duck.load_csv('input.csv', 'weights.csv')
if data['status'] == 'success':
    result = duck.matrix_multiply(data['data']['A'], data['data']['W'])
```

#### `get_status()`
Get API and engine status.

**Returns:**
```python
{
    "status": "online",
    "api_version": "1.0.0",
    "model_name": "Duck (Super APL Model)",
    "cpp_engine_available": true,
    "python_emulator": true
}
```

#### `get_personality()`
Get Duck personality configuration.

**Returns:**
```python
{
    "status": "success",
    "personality": {
        "model_name": "Duck (Super APL Model)",
        "personality_profile": {
            "humor_style": "R2-D2",
            "versatility_style": "C-3PO",
            "system_prompt": "..."
        }
    }
}
```

---

## REST API Endpoints

### Base URL
```
http://localhost:5000/api/v1
```

### GET /status
Get API status.

**Response:**
```json
{
    "status": "online",
    "api_version": "1.0.0",
    "model_name": "Duck (Super APL Model)",
    "cpp_engine_available": true,
    "python_emulator": true
}
```

### GET /personality
Get Duck personality configuration.

**Response:**
```json
{
    "status": "success",
    "personality": {
        "model_name": "Duck (Super APL Model)",
        "personality_profile": {
            "humor_style": "R2-D2",
            "humor_description": "Sassy, expressive, beep-boop sarcasm...",
            "versatility_style": "C-3PO",
            "versatility_description": "Fluent in 6 million forms...",
            "system_prompt": "You are Duck..."
        }
    }
}
```

### POST /compute
Execute an APL expression.

**Request:**
```json
{
    "A": [[1, 2, 3], [4, 5, 6]],
    "W": [[1, 0], [0, 1], [1, 1]],
    "expression": "Result ← Input +.× Weights"
}
```

**Response:**
```json
{
    "status": "success",
    "result": [[4, 5], [10, 11]],
    "shape": [2, 2],
    "engine": "C++ Native",
    "time_ms": 0.5,
    "expression": "Result ← Input +.× Weights"
}
```

### POST /matrix-multiply
Matrix multiply shortcut.

**Request:**
```json
{
    "A": [[1, 2, 3], [4, 5, 6]],
    "W": [[1, 0], [0, 1], [1, 1]]
}
```

**Response:**
```json
{
    "status": "success",
    "result": [[4, 5], [10, 11]],
    "shape": [2, 2],
    "engine": "C++ Native",
    "time_ms": 0.5,
    "expression": "Result ← Input +.× Weights"
}
```

### POST /transpose
Transpose matrix.

**Request:**
```json
{
    "A": [[1, 2, 3], [4, 5, 6]]
}
```

**Response:**
```json
{
    "status": "success",
    "result": [[1, 4], [2, 5], [3, 6]],
    "shape": [3, 2],
    "engine": "Python Emulator",
    "time_ms": 0.2
}
```

### POST /batch
Batch process multiple matrices.

**Request:**
```json
{
    "data": [
        {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
        {"A": [[4, 5, 6]], "W": [[1], [1], [1]]}
    ],
    "expression": "Result ← Input +.× Weights"
}
```

**Response:**
```json
{
    "status": "success",
    "batch_size": 2,
    "results": [
        {
            "index": 0,
            "status": "success",
            "result": [[14]],
            "shape": [1, 1],
            "engine": "C++ Native",
            "time_ms": 0.5
        },
        {
            "index": 1,
            "status": "success",
            "result": [[15]],
            "shape": [1, 1],
            "engine": "C++ Native",
            "time_ms": 0.4
        }
    ]
}
```

---

## CLI Commands

### Setup
```bash
# Navigate to project
cd "super apl learning model"

# Ensure dependencies are installed
pip install numpy flask
```

### matrix-multiply
Multiply two matrices from CSV files.

```bash
python src/api/duck_api.py multiply --input input.csv --weights weights.csv

# With output file
python src/api/duck_api.py multiply --input input.csv --weights weights.csv --output result.json
```

**Example:**
```bash
$ python src/api/duck_api.py multiply --input A.csv --weights W.csv
{
  "status": "success",
  "result": [[4, 5], [10, 11]],
  "shape": [2, 2],
  "engine": "C++ Native",
  "time_ms": 0.5
}
```

### transpose
Transpose a matrix from CSV file.

```bash
python src/api/duck_api.py transpose --input input.csv

# With output file
python src/api/duck_api.py transpose --input input.csv --output result.json
```

### compute
Execute custom APL expression.

```bash
python src/api/duck_api.py compute \
    --input input.csv \
    --weights weights.csv \
    --expr "Result ← Input +.× Weights" \
    --output result.json
```

### server
Start REST API server.

```bash
# Default: http://127.0.0.1:5000
python src/api/duck_api.py server

# Custom host/port
python src/api/duck_api.py server --host 0.0.0.0 --port 8000
```

### status
Get API status.

```bash
python src/api/duck_api.py status
```

**Output:**
```json
{
  "status": "online",
  "api_version": "1.0.0",
  "model_name": "Duck (Super APL Model)",
  "cpp_engine_available": true,
  "python_emulator": true
}
```

### personality
Get Duck personality configuration.

```bash
python src/api/duck_api.py personality
```

---

## Complete Examples

### Python - Simple Computation
```python
from duck_api import DuckAPI

duck = DuckAPI()

# Matrix multiply
A = [[1, 2, 3], [4, 5, 6]]
W = [[1, 0], [0, 1], [1, 1]]

result = duck.matrix_multiply(A, W)

print(f"Result: {result['result']}")
print(f"Engine: {result['engine']}")
print(f"Time: {result['time_ms']:.2f}ms")
```

### Python - Batch Processing
```python
from duck_api import DuckAPI

duck = DuckAPI()

# Process multiple datasets
datasets = [
    {"A": [[1, 2]], "W": [[1], [2]]},
    {"A": [[3, 4]], "W": [[1], [2]]},
    {"A": [[5, 6]], "W": [[1], [2]]}
]

results = duck.batch_compute(datasets)

for r in results:
    if r['status'] == 'success':
        print(f"Item {r['index']}: {r['result']} ({r['time_ms']:.2f}ms)")
    else:
        print(f"Item {r['index']}: Error - {r['error']}")
```

### Python - CSV Loading
```python
from duck_api import DuckAPI

duck = DuckAPI()

# Load from CSV
data = duck.load_csv('input.csv', 'weights.csv')

if data['status'] == 'success':
    A = data['data']['A']
    W = data['data']['W']
    result = duck.matrix_multiply(A, W)
    print(result['result'])
```

### CLI - Batch Processing
```bash
# Create CSV files
echo "1,2,3" > A.csv
echo "4,5,6" >> A.csv

echo "1,0" > W.csv
echo "0,1" >> W.csv
echo "1,1" >> W.csv

# Run computation
python src/api/duck_api.py multiply --input A.csv --weights W.csv --output result.json

# View result
cat result.json
```

### REST - Using curl
```bash
# Start server
python src/api/duck_api.py server &

# Wait for startup
sleep 2

# Status check
curl http://localhost:5000/api/v1/status

# Matrix multiply
curl -X POST http://localhost:5000/api/v1/matrix-multiply \
  -H "Content-Type: application/json" \
  -d '{
    "A": [[1, 2, 3], [4, 5, 6]],
    "W": [[1, 0], [0, 1], [1, 1]]
  }'

# Batch processing
curl -X POST http://localhost:5000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
      {"A": [[4, 5, 6]], "W": [[1], [1], [1]]}
    ]
  }'
```

### REST - Using Python requests
```python
import requests
import json

# Start server first: python src/api/duck_api.py server

url = 'http://localhost:5000/api/v1/matrix-multiply'
data = {
    'A': [[1, 2, 3], [4, 5, 6]],
    'W': [[1, 0], [0, 1], [1, 1]]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Result: {result['result']}")
print(f"Engine: {result['engine']}")
print(f"Time: {result['time_ms']:.2f}ms")
```

---

## Error Handling

### Python
```python
from duck_api import DuckAPI

duck = DuckAPI()
result = duck.matrix_multiply([[1, 2]], [[1, 2, 3]])  # Dimension mismatch

if result['status'] == 'error':
    print(f"Error: {result['error']}")
    print(f"Type: {result['error_type']}")
```

### REST
```bash
# Invalid request (missing W)
curl -X POST http://localhost:5000/api/v1/matrix-multiply \
  -H "Content-Type: application/json" \
  -d '{"A": [[1, 2, 3]]}'

# Returns 400 Bad Request:
# {"status": "error", "error": "Missing 'A' or 'W' in request body"}
```

---

## Requirements

- Python 3.7+
- NumPy
- Flask (for REST API)
- super_apl_engine.dll (for C++ acceleration)

---

## API Versioning

Current version: **1.0.0**

- API base path: `/api/v1`
- Backwards compatibility maintained for all 1.x versions

---

## Performance Notes

- **C++ Engine**: ~0.5ms for small matrices when DLL available
- **Python Emulator**: ~0.2-1ms depending on matrix size
- **Batch processing**: Linear scaling with number of matrices
- **CSV loading**: Depends on file size (~1-10ms for typical files)

---

## Support & Examples

See `DUCK_API_EXAMPLES.md` for additional usage examples and integration patterns.
