"""
Duck API Usage Examples

Quick reference for common usage patterns
"""

# ============================================================================
# Example 1: Simple Python Module Usage
# ============================================================================

def example_python_simple():
    """Basic matrix multiplication using Python module"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    # Define matrices
    A = [[1, 2, 3], [4, 5, 6]]      # 2×3 matrix
    W = [[1, 0], [0, 1], [1, 1]]    # 3×2 matrix
    
    # Compute
    result = duck.matrix_multiply(A, W)
    
    # Check result
    print(f"Result: {result['result']}")
    print(f"Shape: {result['shape']}")
    print(f"Engine: {result['engine']}")
    print(f"Time: {result['time_ms']:.2f}ms")
    # Output: Result: [[4, 5], [10, 11]], Engine: C++ Native, Time: 0.5ms


# ============================================================================
# Example 2: Transpose Matrix
# ============================================================================

def example_python_transpose():
    """Transpose a matrix"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    A = [[1, 2, 3], [4, 5, 6]]
    result = duck.transpose(A)
    
    print(f"Original: {A}")
    print(f"Transposed: {result['result']}")
    # Output: Transposed: [[1, 4], [2, 5], [3, 6]]


# ============================================================================
# Example 3: Custom APL Expression
# ============================================================================

def example_python_custom_expr():
    """Execute custom APL expression"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    A = [[1, 2, 3], [4, 5, 6]]
    
    # Custom transpose expression
    result = duck.compute(A, None, "Result ← φ Input")
    print(f"Transposed: {result['result']}")


# ============================================================================
# Example 4: Load from CSV Files
# ============================================================================

def example_python_csv_load():
    """Load matrices from CSV files"""
    from duck_api import DuckAPI
    
    # Create sample CSV files
    import csv
    
    # Create input.csv
    with open('input.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[1, 2, 3], [4, 5, 6]])
    
    # Create weights.csv
    with open('weights.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[1, 0], [0, 1], [1, 1]])
    
    # Load and compute
    duck = DuckAPI()
    data = duck.load_csv('input.csv', 'weights.csv')
    
    if data['status'] == 'success':
        result = duck.matrix_multiply(data['data']['A'], data['data']['W'])
        print(f"Result: {result['result']}")


# ============================================================================
# Example 5: Batch Processing
# ============================================================================

def example_python_batch():
    """Process multiple matrices at once"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    # Multiple datasets
    datasets = [
        {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
        {"A": [[4, 5, 6]], "W": [[1], [1], [1]]},
        {"A": [[7, 8, 9]], "W": [[2], [1], [1]]}
    ]
    
    # Process all
    results = duck.batch_compute(datasets)
    
    # Display results
    for r in results:
        if r['status'] == 'success':
            print(f"Item {r['index']}: {r['result']} ({r['time_ms']:.2f}ms)")
        else:
            print(f"Item {r['index']}: Error - {r['error']}")


# ============================================================================
# Example 6: Error Handling
# ============================================================================

def example_python_error_handling():
    """Handle errors gracefully"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    # Dimension mismatch error
    A = [[1, 2]]
    W = [[1, 2, 3]]  # W dimensions don't match A
    
    result = duck.matrix_multiply(A, W)
    
    if result['status'] == 'error':
        print(f"Error: {result['error']}")
        print(f"Error type: {result['error_type']}")
    else:
        print(f"Success: {result['result']}")


# ============================================================================
# Example 7: CLI - Matrix Multiply
# ============================================================================

def example_cli_multiply():
    """
    Command-line matrix multiply
    
    Usage:
        python duck_api.py multiply --input A.csv --weights W.csv
        python duck_api.py multiply --input A.csv --weights W.csv --output result.json
    
    Examples:
        python duck_api.py multiply --input input.csv --weights weights.csv
        python duck_api.py multiply --input data/A.csv --weights data/W.csv --output output.json
    """
    pass


# ============================================================================
# Example 8: CLI - Transpose
# ============================================================================

def example_cli_transpose():
    """
    Command-line transpose
    
    Usage:
        python duck_api.py transpose --input A.csv
        python duck_api.py transpose --input A.csv --output result.json
    
    Examples:
        python duck_api.py transpose --input input.csv
        python duck_api.py transpose --input data/A.csv --output output.json
    """
    pass


# ============================================================================
# Example 9: CLI - Custom Expression
# ============================================================================

def example_cli_compute():
    """
    Command-line custom expression
    
    Usage:
        python duck_api.py compute --input A.csv --expr "Result ← φ Input"
        python duck_api.py compute --input A.csv --weights W.csv --expr "Result ← Input +.× Weights"
    
    Examples:
        python duck_api.py compute --input data/A.csv --expr "Result ← φ Input"
        python duck_api.py compute --input A.csv --weights W.csv --expr "Result ← Input +.× Weights" --output result.json
    """
    pass


# ============================================================================
# Example 10: CLI - Start REST Server
# ============================================================================

def example_cli_server():
    """
    Start REST API server
    
    Usage:
        python duck_api.py server
        python duck_api.py server --port 8000
        python duck_api.py server --host 0.0.0.0 --port 5000
    
    Examples:
        python duck_api.py server                          # localhost:5000
        python duck_api.py server --port 8000              # localhost:8000
        python duck_api.py server --host 0.0.0.0           # All interfaces
    """
    pass


# ============================================================================
# Example 11: REST API with curl
# ============================================================================

def example_rest_curl():
    """
    REST API usage with curl
    
    First, start the server:
        python duck_api.py server
    
    Then in another terminal:
    
    Check status:
        curl http://localhost:5000/api/v1/status
    
    Matrix multiply:
        curl -X POST http://localhost:5000/api/v1/matrix-multiply \\
          -H "Content-Type: application/json" \\
          -d '{"A": [[1,2,3],[4,5,6]], "W": [[1,0],[0,1],[1,1]]}'
    
    Transpose:
        curl -X POST http://localhost:5000/api/v1/transpose \\
          -H "Content-Type: application/json" \\
          -d '{"A": [[1,2,3],[4,5,6]]}'
    
    Batch processing:
        curl -X POST http://localhost:5000/api/v1/batch \\
          -H "Content-Type: application/json" \\
          -d '{
            "data": [
              {"A": [[1,2,3]], "W": [[1],[2],[3]]},
              {"A": [[4,5,6]], "W": [[1],[1],[1]]}
            ]
          }'
    """
    pass


# ============================================================================
# Example 12: REST API with Python requests
# ============================================================================

def example_rest_python():
    """REST API usage with Python requests library"""
    import requests
    
    # Make sure server is running: python duck_api.py server
    
    url = 'http://localhost:5000/api/v1/matrix-multiply'
    
    data = {
        'A': [[1, 2, 3], [4, 5, 6]],
        'W': [[1, 0], [0, 1], [1, 1]]
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if result['status'] == 'success':
        print(f"Result: {result['result']}")
        print(f"Engine: {result['engine']}")
        print(f"Time: {result['time_ms']:.2f}ms")
    else:
        print(f"Error: {result['error']}")


# ============================================================================
# Example 13: Batch REST API with requests
# ============================================================================

def example_rest_batch():
    """Batch processing via REST API"""
    import requests
    
    url = 'http://localhost:5000/api/v1/batch'
    
    data = {
        'data': [
            {'A': [[1, 2, 3]], 'W': [[1], [2], [3]]},
            {'A': [[4, 5, 6]], 'W': [[1], [1], [1]]},
            {'A': [[7, 8, 9]], 'W': [[2], [1], [1]]}
        ]
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    print(f"Processed {result['batch_size']} items:")
    for r in result['results']:
        print(f"  Item {r['index']}: {r['result']}")


# ============================================================================
# Example 14: Getting API Status
# ============================================================================

def example_api_status():
    """Check API and engine status"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    status = duck.get_status()
    print(f"API Version: {status['api_version']}")
    print(f"Model: {status['model_name']}")
    print(f"C++ Engine: {'Available' if status['cpp_engine_available'] else 'Not available'}")
    print(f"Python Emulator: {'Available' if status['python_emulator'] else 'Not available'}")


# ============================================================================
# Example 15: Duck Personality
# ============================================================================

def example_duck_personality():
    """Get Duck personality configuration"""
    from duck_api import DuckAPI
    
    duck = DuckAPI()
    
    personality = duck.get_personality()
    profile = personality['personality']['personality_profile']
    
    print(f"Humor Style: {profile['humor_style']}")
    print(f"  {profile['humor_description']}")
    print()
    print(f"Versatility: {profile['versatility_style']}")
    print(f"  {profile['versatility_description']}")
    print()
    print(f"Prompt: {profile['system_prompt']}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Simple Python Usage', example_python_simple),
        '2': ('Transpose', example_python_transpose),
        '3': ('Custom Expression', example_python_custom_expr),
        '4': ('CSV Loading', example_python_csv_load),
        '5': ('Batch Processing', example_python_batch),
        '6': ('Error Handling', example_python_error_handling),
        '7': ('CLI Multiply', example_cli_multiply),
        '8': ('CLI Transpose', example_cli_transpose),
        '9': ('CLI Compute', example_cli_compute),
        '10': ('CLI Server', example_cli_server),
        '11': ('REST with curl', example_rest_curl),
        '12': ('REST with Python', example_rest_python),
        '13': ('REST Batch', example_rest_batch),
        '14': ('API Status', example_api_status),
        '15': ('Duck Personality', example_duck_personality),
    }
    
    print("Duck API Examples")
    print("=" * 50)
    print()
    
    for key, (name, func) in examples.items():
        print(f"{key:2}. {name}")
    
    print()
    print("Usage: python duck_api_examples.py [example_number]")
    print("       python duck_api_examples.py 1")
    print()
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"Running Example {example_num}: {name}")
            print("=" * 50)
            print()
            func()
        else:
            print(f"Unknown example: {example_num}")
    else:
        print("(Run without arguments to see usage)")
