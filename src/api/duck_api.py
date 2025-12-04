"""
Duck API - REST and programmatic interface to Duck Super APL Learning Model

Provides three interfaces:
1. REST API (Flask) - HTTP endpoints for web services
2. CLI - Command-line interface for batch processing
3. Python Module - Direct function calls
"""

import json
import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add src/gui to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))

from app import APLEmulator

# ============================================================================
# Core Duck API Class
# ============================================================================

class DuckAPI:
    """Main API interface for Duck Super APL Learning Model"""
    
    def __init__(self, dll_path=None, personality_path=None):
        """
        Initialize Duck API
        
        Args:
            dll_path: Path to super_apl_engine.dll (auto-detected if None)
            personality_path: Path to duck_personality.json (auto-detected if None)
        """
        self.emulator = APLEmulator(dll_path)
        self.personality = self._load_personality(personality_path)
        self.version = "1.0.0"
    
    def _load_personality(self, personality_path=None):
        """Load Duck personality configuration"""
        if personality_path is None:
            # Search for personality file
            search_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'training', 'duck_personality.json'),
                os.path.join(os.getcwd(), 'duck_personality.json'),
                os.path.join(os.getcwd(), 'src', 'training', 'duck_personality.json'),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    personality_path = path
                    break
        
        if personality_path and os.path.exists(personality_path):
            try:
                with open(personality_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load personality file: {e}")
        
        return self._default_personality()
    
    @staticmethod
    def _default_personality():
        """Return default Duck personality if file not found"""
        return {
            "model_name": "Duck (Super APL Model)",
            "personality_profile": {
                "humor_style": "R2-D2",
                "humor_description": "Sassy, expressive, beep-boop sarcasm, brave, cheeky, situational comedy.",
                "versatility_style": "C-3PO",
                "versatility_description": "Fluent in over 6 million forms of communication, protocol-focused, anxious but helpful.",
                "system_prompt": "You are Duck. You possess C-3PO versatility with R2-D2 attitude: sassy, brave, helpful, and witty."
            },
            "training_parameters": {
                "quantization": "NF4",
                "context_window": 4096
            }
        }
    
    def compute(self, A, W=None, expression="Result ← Input +.× Weights"):
        """
        Perform APL computation
        
        Args:
            A: Input matrix (numpy array or list)
            W: Weight matrix (numpy array or list). If None, generates random weights.
            expression: APL expression to execute
        
        Returns:
            dict with keys:
            - result: numpy array result
            - shape: result shape
            - engine: "C++ Native" or "Python Emulator"
            - time_ms: computation time in milliseconds
            - status: "success" or "error"
        """
        try:
            import time
            
            # Convert to numpy arrays
            A = np.array(A, dtype=np.float32)
            if W is not None:
                W = np.array(W, dtype=np.float32)
            
            # Format input for emulator
            input_str = self._format_input(A, W)
            
            # Time the computation
            start = time.time()
            result = self.emulator.execute_apl_expression(expression, input_str)
            elapsed_ms = (time.time() - start) * 1000
            
            # Determine which engine was used
            engine = "C++ Native" if (self.emulator.cpp_engine and self.emulator.cpp_engine.available) else "Python Emulator"
            
            return {
                "status": "success",
                "result": result.tolist() if isinstance(result, np.ndarray) else result,
                "shape": list(result.shape) if isinstance(result, np.ndarray) else None,
                "engine": engine,
                "time_ms": elapsed_ms,
                "expression": expression
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def matrix_multiply(self, A, W):
        """
        Simple matrix multiply shortcut
        
        Args:
            A: Input matrix (M×K)
            W: Weight matrix (K×N)
        
        Returns:
            dict with result (M×N) and metadata
        """
        return self.compute(A, W, "Result ← Input +.× Weights")
    
    def transpose(self, A):
        """
        Transpose matrix
        
        Args:
            A: Input matrix
        
        Returns:
            dict with transposed result and metadata
        """
        return self.compute(A, None, "Result ← φ Input")
    
    def batch_compute(self, data_list, expression="Result ← Input +.× Weights"):
        """
        Process multiple matrices in batch
        
        Args:
            data_list: List of dicts with keys "A" (required) and "W" (optional)
            expression: APL expression to execute
        
        Returns:
            List of result dicts, each with status, result, engine, time_ms
        """
        results = []
        for i, data in enumerate(data_list):
            A = data.get("A")
            W = data.get("W")
            if A is None:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": "Missing required key 'A'"
                })
                continue
            
            result = self.compute(A, W, expression)
            result["index"] = i
            results.append(result)
        
        return results
    
    def load_csv(self, filepath_a, filepath_w=None):
        """
        Load matrices from CSV files
        
        Args:
            filepath_a: Path to CSV file for matrix A
            filepath_w: Path to CSV file for matrix W (optional)
        
        Returns:
            dict with keys "A" (and "W" if provided)
        """
        try:
            A = np.loadtxt(filepath_a, delimiter=',', dtype=np.float32)
            result = {"A": A.tolist()}
            
            if filepath_w:
                W = np.loadtxt(filepath_w, delimiter=',', dtype=np.float32)
                result["W"] = W.tolist()
            
            return {
                "status": "success",
                "data": result,
                "A_shape": A.shape,
                "W_shape": np.array(W).shape if filepath_w else None
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def get_personality(self):
        """Get Duck personality configuration"""
        return {
            "status": "success",
            "personality": self.personality,
            "api_version": self.version
        }
    
    def get_status(self):
        """Get API and engine status"""
        cpp_available = (self.emulator.cpp_engine and self.emulator.cpp_engine.available) if self.emulator.cpp_engine else False
        
        return {
            "status": "online",
            "api_version": self.version,
            "model_name": self.personality.get("model_name", "Duck"),
            "cpp_engine_available": cpp_available,
            "python_emulator": True
        }
    
    @staticmethod
    def _format_input(A, W=None):
        """Format matrix data for emulator"""
        A_str = "\n".join([" ".join(f"{x:.6f}" for x in row) for row in A])
        
        if W is not None:
            W_str = "\n".join([" ".join(f"{x:.6f}" for x in row) for row in W])
            return f"A:\n{A_str}\n\nW:\n{W_str}"
        
        return A_str


# ============================================================================
# REST API (Flask)
# ============================================================================

def create_flask_app(duck_api):
    """Create Flask REST API application"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Error: Flask not installed. Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route('/api/v1/status', methods=['GET'])
    def status():
        """GET /api/v1/status - Get API status"""
        return jsonify(duck_api.get_status())
    
    @app.route('/api/v1/personality', methods=['GET'])
    def personality():
        """GET /api/v1/personality - Get Duck personality"""
        return jsonify(duck_api.get_personality())
    
    @app.route('/api/v1/compute', methods=['POST'])
    def compute():
        """POST /api/v1/compute - Execute APL expression
        
        Request body:
        {
            "A": [[1, 2, 3], [4, 5, 6]],
            "W": [[1, 0], [0, 1], [1, 1]],
            "expression": "Result ← Input +.× Weights"
        }
        """
        try:
            data = request.get_json()
            A = data.get('A')
            W = data.get('W')
            expression = data.get('expression', "Result ← Input +.× Weights")
            
            if A is None:
                return jsonify({"status": "error", "error": "Missing 'A' in request body"}), 400
            
            result = duck_api.compute(A, W, expression)
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400
    
    @app.route('/api/v1/matrix-multiply', methods=['POST'])
    def matrix_multiply():
        """POST /api/v1/matrix-multiply - Matrix multiply shortcut
        
        Request body:
        {
            "A": [[1, 2, 3], [4, 5, 6]],
            "W": [[1, 0], [0, 1], [1, 1]]
        }
        """
        try:
            data = request.get_json()
            A = data.get('A')
            W = data.get('W')
            
            if A is None or W is None:
                return jsonify({"status": "error", "error": "Missing 'A' or 'W' in request body"}), 400
            
            result = duck_api.matrix_multiply(A, W)
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400
    
    @app.route('/api/v1/transpose', methods=['POST'])
    def transpose():
        """POST /api/v1/transpose - Transpose matrix
        
        Request body:
        {
            "A": [[1, 2, 3], [4, 5, 6]]
        }
        """
        try:
            data = request.get_json()
            A = data.get('A')
            
            if A is None:
                return jsonify({"status": "error", "error": "Missing 'A' in request body"}), 400
            
            result = duck_api.transpose(A)
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400
    
    @app.route('/api/v1/batch', methods=['POST'])
    def batch():
        """POST /api/v1/batch - Batch processing
        
        Request body:
        {
            "data": [
                {"A": [[1, 2, 3], [4, 5, 6]], "W": [[1, 0], [0, 1], [1, 1]]},
                {"A": [[7, 8, 9], [10, 11, 12]], "W": [[2, 1], [1, 2], [0, 1]]}
            ],
            "expression": "Result ← Input +.× Weights"
        }
        """
        try:
            data = request.get_json()
            data_list = data.get('data', [])
            expression = data.get('expression', "Result ← Input +.× Weights")
            
            if not data_list:
                return jsonify({"status": "error", "error": "Missing 'data' in request body"}), 400
            
            results = duck_api.batch_compute(data_list, expression)
            return jsonify({
                "status": "success",
                "batch_size": len(results),
                "results": results
            })
        
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400
    
    return app


# ============================================================================
# CLI Interface
# ============================================================================

def main_cli():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Duck Super APL Learning Model API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Matrix multiply
  python duck_api.py multiply --input A.csv --weights W.csv
  
  # Transpose
  python duck_api.py transpose --input A.csv
  
  # Custom expression
  python duck_api.py compute --input A.csv --weights W.csv --expr "Result ← Input +.× Weights"
  
  # REST server
  python duck_api.py server --port 5000
  
  # Get status
  python duck_api.py status
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # multiply command
    multiply_parser = subparsers.add_parser('multiply', help='Matrix multiply')
    multiply_parser.add_argument('--input', required=True, help='Input CSV file (A)')
    multiply_parser.add_argument('--weights', required=True, help='Weights CSV file (W)')
    multiply_parser.add_argument('--output', help='Output JSON file')
    
    # transpose command
    transpose_parser = subparsers.add_parser('transpose', help='Transpose matrix')
    transpose_parser.add_argument('--input', required=True, help='Input CSV file')
    transpose_parser.add_argument('--output', help='Output JSON file')
    
    # compute command
    compute_parser = subparsers.add_parser('compute', help='Custom APL expression')
    compute_parser.add_argument('--input', required=True, help='Input CSV file (A)')
    compute_parser.add_argument('--weights', help='Weights CSV file (W)')
    compute_parser.add_argument('--expr', required=True, help='APL expression')
    compute_parser.add_argument('--output', help='Output JSON file')
    
    # server command
    server_parser = subparsers.add_parser('server', help='Start REST API server')
    server_parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    server_parser.add_argument('--host', default='127.0.0.1', help='Host (default: 127.0.0.1)')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Get API status')
    
    # personality command
    personality_parser = subparsers.add_parser('personality', help='Get Duck personality')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize API
    duck = DuckAPI()
    
    if args.command == 'multiply':
        result = duck.load_csv(args.input, args.weights)
        if result.get('status') == 'success':
            data = result['data']
            result = duck.matrix_multiply(data['A'], data['W'])
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result.get('error')}")
    
    elif args.command == 'transpose':
        result = duck.load_csv(args.input)
        if result.get('status') == 'success':
            data = result['data']
            result = duck.transpose(data['A'])
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result.get('error')}")
    
    elif args.command == 'compute':
        result = duck.load_csv(args.input, args.weights)
        if result.get('status') == 'success':
            data = result['data']
            result = duck.compute(data['A'], data.get('W'), args.expr)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result.get('error')}")
    
    elif args.command == 'server':
        app = create_flask_app(duck)
        if app:
            print(f"Starting Duck API server on {args.host}:{args.port}")
            print(f"Endpoints:")
            print(f"  GET  /api/v1/status")
            print(f"  GET  /api/v1/personality")
            print(f"  POST /api/v1/compute")
            print(f"  POST /api/v1/matrix-multiply")
            print(f"  POST /api/v1/transpose")
            print(f"  POST /api/v1/batch")
            print()
            app.run(host=args.host, port=args.port, debug=False)
    
    elif args.command == 'status':
        status = duck.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'personality':
        personality = duck.get_personality()
        print(json.dumps(personality, indent=2))


if __name__ == '__main__':
    main_cli()
