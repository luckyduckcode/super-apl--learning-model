#!/usr/bin/env python3
"""
Verify that the C++ engine DLL is properly callable and functional.
"""
import ctypes
import os
import sys
import numpy as np

def test_dll_loading():
    """Test DLL can be loaded and function found."""
    print("[DLL Verification] Testing super_apl_engine.dll\n")
    
    # Search for DLL
    dll_paths = [
        os.path.join(os.getcwd(), 'build', 'Release', 'super_apl_engine.dll'),
        os.path.join(os.getcwd(), 'dist', 'super_apl_engine.dll'),
        os.path.join(os.getcwd(), 'super_apl_engine.dll'),
    ]
    
    dll_path = None
    for path in dll_paths:
        if os.path.exists(path):
            dll_path = path
            print(f"✓ Found DLL: {path}")
            break
    
    if not dll_path:
        print("✗ DLL not found in:")
        for path in dll_paths:
            print(f"  - {path}")
        return False
    
    # Load DLL
    try:
        dll = ctypes.CDLL(dll_path)
        print(f"✓ Loaded DLL successfully")
    except Exception as e:
        print(f"✗ Failed to load DLL: {e}")
        return False
    
    # Test SimpleMatrixMultiply function
    try:
        simple_multiply = dll.SimpleMatrixMultiply
        simple_multiply.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # W
            ctypes.c_int,                     # M
            ctypes.c_int,                     # N
            ctypes.c_int,                     # K
        ]
        simple_multiply.restype = None
        print(f"✓ Found SimpleMatrixMultiply function")
    except AttributeError as e:
        print(f"✗ Function not found: {e}")
        return False
    
    # Test computation
    print("\n[Computation Test]\n")
    
    M, K, N = 2, 3, 2
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    
    print(f"A shape: {A.shape}")
    print(f"W shape: {W.shape}")
    print(f"Expected C shape: ({M}, {N})")
    
    # Call C++ function
    try:
        simple_multiply(
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K
        )
        print(f"\n✓ C++ function executed successfully")
    except Exception as e:
        print(f"✗ Function call failed: {e}")
        return False
    
    # Verify result
    expected = np.array([[4.0, 5.0], [10.0, 11.0]], dtype=np.float32)
    print(f"\nResult C:\n{C}")
    print(f"\nExpected:\n{expected}")
    
    if np.allclose(C, expected, rtol=1e-5):
        print(f"\n✓ Result matches expected output")
        return True
    else:
        print(f"\n✗ Result does NOT match expected output")
        diff = np.abs(C - expected)
        print(f"Max difference: {np.max(diff)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Super APL Learning Model - DLL Verification")
    print("=" * 60 + "\n")
    
    success = test_dll_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ DLL verification PASSED")
        print("✓ Engine is ready for use")
    else:
        print("✗ DLL verification FAILED")
        sys.exit(1)
    print("=" * 60)
