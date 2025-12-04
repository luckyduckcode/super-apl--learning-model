"""
ctypes_engine.py - Python ctypes binding for super_apl_engine.dll

This module provides a Python interface to the C++ engine via ctypes.
It handles data marshalling, struct layout, and error handling.
"""

import ctypes
import numpy as np
import os
import sys

class EngineBinding:
    def __init__(self, dll_path=None):
        """Initialize the ctypes binding to the C++ engine.
        
        Args:
            dll_path: Path to super_apl_engine.dll. If None, searches in common locations.
        """
        self.dll = None
        self.available = False
        
        if dll_path is None:
            # Search in common locations
            search_paths = [
                os.path.join(os.getcwd(), 'super_apl_engine.dll'),
                os.path.join(os.getcwd(), 'build', 'Release', 'super_apl_engine.dll'),
                os.path.join(os.getcwd(), 'dist', 'super_apl_engine.dll'),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    dll_path = path
                    break
        
        if dll_path and os.path.exists(dll_path):
            try:
                self.dll = ctypes.CDLL(dll_path)
                
                # Define function signature
                # void SimpleMatrixMultiply(float* C, const float* A, const float* W, int M, int N, int K)
                self.simple_multiply = self.dll.SimpleMatrixMultiply
                self.simple_multiply.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # C (output)
                    ctypes.POINTER(ctypes.c_float),  # A (input)
                    ctypes.POINTER(ctypes.c_float),  # W (weights)
                    ctypes.c_int,                     # M
                    ctypes.c_int,                     # N
                    ctypes.c_int                      # K
                ]
                self.simple_multiply.restype = None
                
                self.available = True
            except Exception as e:
                print(f"[Engine] Failed to load C++ engine: {e}")
                self.available = False
    
    def matrix_multiply(self, A, W):
        """Call SimpleMatrixMultiply from the C++ engine.
        
        Args:
            A: numpy array (M x K)
            W: numpy array (K x N)
        
        Returns:
            numpy array (M x N) with the result
        """
        if not self.available:
            return None
        
        # Ensure inputs are float32 and C-contiguous
        A = np.ascontiguousarray(A, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
        
        M, K = A.shape
        K2, N = W.shape
        
        if K != K2:
            raise ValueError(f"Shape mismatch: A is {A.shape} but W is {W.shape}")
        
        # Allocate output buffer
        C = np.zeros((M, N), dtype=np.float32)
        
        try:
            # Call the C++ function
            self.simple_multiply(
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(M),
                ctypes.c_int(N),
                ctypes.c_int(K)
            )
            return C
        except Exception as e:
            print(f"[Engine] Error calling C++ function: {e}")
            return None
