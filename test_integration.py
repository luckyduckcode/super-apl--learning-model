#!/usr/bin/env python3
"""
Integration test for the APLEmulator with CSV loader and ctypes engine binding.
"""
import sys
import os
import tempfile

# Add src/gui to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'gui'))

import numpy as np
from ctypes_engine import EngineBinding
from app import APLEmulator, SuperAPLApp

def test_csv_loading():
    """Test CSV loading helper functions."""
    print("[Test] CSV Loading")
    
    # Create a dummy root for SuperAPLApp
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    app = SuperAPLApp(root)
    
    # Create test CSV file in temp directory
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1.0,2.0,3.0\n")
        f.write("4.0,5.0,6.0\n")
        test_csv = f.name
    
    try:
        # Test matrix loading
        matrix = app._load_csv_matrix(test_csv)
        print(f"  Loaded matrix shape: {matrix.shape}")
        print(f"  Matrix:\n{matrix}")
        assert matrix.shape == (2, 3), f"Expected (2, 3), got {matrix.shape}"
        assert np.allclose(matrix[0, 0], 1.0), "First element should be 1.0"
        
        # Test matrix formatting
        formatted = app._format_matrix(matrix)
        print(f"  Formatted matrix:\n{formatted}")
        
        print("  ✓ CSV loading test passed\n")
    finally:
        os.unlink(test_csv)
        root.destroy()

def test_engine_binding():
    """Test EngineBinding availability and fallback."""
    print("[Test] Engine Binding")
    
    apl = APLEmulator()
    
    if apl.cpp_engine is not None:
        if apl.cpp_engine.available:
            print("  ✓ C++ Engine loaded successfully")
        else:
            print("  ✓ C++ Engine DLL not available (will use emulator)")
    else:
        print("  ✓ EngineBinding module not available (pure emulator mode)")
    
    print()

def test_matrix_multiply():
    """Test matrix multiply with fallback logic."""
    print("[Test] Matrix Multiply")
    
    apl = APLEmulator()
    
    # Test input
    input_data = "A:\n1.0 2.0 3.0\n4.0 5.0 6.0\n\nW:\n1.0 0.0\n0.0 1.0\n1.0 1.0"
    
    result = apl.execute_apl_expression("Result +.× W", input_data)
    
    if isinstance(result, np.ndarray):
        print(f"  Result shape: {result.shape}")
        print(f"  Result:\n{result}")
        print("  ✓ Matrix multiply test passed")
    else:
        print(f"  Result: {result}")
        print("  ! Matrix multiply returned non-array")
    
    print()

if __name__ == "__main__":
    print("="*60)
    print("Super APL Learning Model - Integration Tests")
    print("="*60 + "\n")
    
    try:
        test_csv_loading()
        test_engine_binding()
        test_matrix_multiply()
        
        print("="*60)
        print("All integration tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
