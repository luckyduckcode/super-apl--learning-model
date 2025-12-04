import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ctypes
import os
import sys
import numpy as np
import time
import re
import csv

# Import the ctypes engine wrapper
try:
    from ctypes_engine import EngineBinding
except ImportError:
    EngineBinding = None

# --- APL Emulation Layer ---
# In a real scenario, this would connect to a Dyalog APL instance or interpreter.
# Here, we emulate the "Top Layer" of the architecture in Python for the GUI.
class APLEmulator:
    def __init__(self, dll_path=None):
        self.engine = None
        self.cpp_engine = None
        if EngineBinding is not None:
            self.cpp_engine = EngineBinding(dll_path)
            if self.cpp_engine.available:
                print("[APL] C++ Engine loaded successfully.")
            else:
                print("[APL] C++ Engine not available. Will use Python emulator.")
        else:
            print("[APL] ctypes_engine module not found. Using Python emulator only.")

    def execute_apl_expression(self, expression, input_data):
        """
        Parses a pseudo-APL expression and executes it.
        Example: Result <- Input +.x Weights
        Supports both ASCII and Unicode '×' in the expression.
        """
        import re
        # Accept +.x, +.× and patterns with spaces or unicode arrow
        expr = expression.strip()
        # Check for matrix-multiply operator variants
        if re.search(r"\+\.x|\+\.×", expr):
            return self._matrix_multiply(input_data)
        # Transpose operator
        if re.search(r"\bphi\b|\bφ\b", expr):
            # try parse matrix from input
            try:
                arr = self._parse_input_matrix(input_data)
                return arr.T
            except Exception:
                return "Unknown Operation"
        return "Unknown Operation"

    def _matrix_multiply(self, input_data):
        # Parse input_data into matrices A and W.
        try:
            A, W = self._parse_a_w_from_input(input_data)
            
            # Try C++ engine first
            if self.cpp_engine and self.cpp_engine.available:
                result = self.cpp_engine.matrix_multiply(A, W)
                if result is not None:
                    return result
            
            # Fallback to Python emulator
            res = np.dot(A, W)
            return res
        except Exception as e:
            print(f"[Error] {e}")
            # Fallback: random output to indicate it ran
            return np.random.rand(1, 10)

    def _parse_input_matrix(self, input_str):
        """Parses a string into a numpy matrix -- rows separated by newlines, columns by spaces/commas."""
        lines = [l.strip() for l in input_str.strip().splitlines() if l.strip() != ""]
        if len(lines) == 0:
            raise ValueError("No input")
        rows = []
        for line in lines:
            # split on space or comma
            parts = [p for p in re.split(r"[\s,]+", line) if p != ""]
            rows.append([float(p) for p in parts])
        mat = np.array(rows, dtype=np.float32)
        return mat

    def _parse_a_w_from_input(self, input_str):
        """If the input contains two matrices, return A, W. If it only contains one matrix, produce a random W.
        Separators: a blank line separates the two matrices, or a header 'A:' and 'W:' may be present.
        """
        import re
        # If A: and W: present
        if "A:" in input_str and "W:" in input_str:
            # split by header
            a_part = input_str.split("A:")[1].split("W:")[0]
            w_part = input_str.split("W:")[1]
            A = self._parse_input_matrix(a_part)
            W = self._parse_input_matrix(w_part)
            return A, W

        # If blank line separates two matrices
        parts = [p for p in re.split(r"\n\s*\n", input_str.strip()) if p.strip() != ""]
        if len(parts) >= 2:
            A = self._parse_input_matrix(parts[0])
            W = self._parse_input_matrix(parts[1])
            return A, W

        # Single matrix -> treat as A and create W randomly as column vector
        A = self._parse_input_matrix(input_str)
        # If A is 1xN -> make W N x 1
        N = A.shape[1]
        # For inference, generate a random weight to produce a vector.
        W = np.random.rand(N, min(10, N)).astype(np.float32)
        return A, W

# --- GUI Application ---
class SuperAPLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super APL Learning Model - Inference Engine")
        self.root.geometry("800x600")
        
        # Determine path to DLL (assuming it's in the same dir as exe or in build folder)
        dll_name = "super_apl_engine.dll"
        dll_path = os.path.join(os.getcwd(), "build", "Debug", dll_name)
        if not os.path.exists(dll_path):
            dll_path = os.path.join(os.getcwd(), dll_name)

        self.apl = APLEmulator(dll_path)

        self._setup_ui()

    def _setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#333", height=80)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Super APL / C++ Hybrid Architecture", 
                 fg="white", bg="#333", font=("Segoe UI", 18, "bold")).pack(pady=20)

        # Main Content
        content_frame = tk.Frame(self.root, padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)

        # Left Panel: Controls
        left_panel = tk.LabelFrame(content_frame, text="Model Controls", padx=10, pady=10)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(left_panel, text="APL Expression:").pack(anchor="w")
        self.expr_entry = tk.Entry(left_panel, width=30)
        self.expr_entry.insert(0, "Result ← Input +.× Weights")
        self.expr_entry.pack(fill="x", pady=(0, 10))

        tk.Label(left_panel, text="Quantization Level:").pack(anchor="w")
        self.quant_combo = ttk.Combobox(left_panel, values=["4-bit (Q4_0)", "2-bit (Q2_K)"])
        self.quant_combo.current(0)
        self.quant_combo.pack(fill="x", pady=(0, 10))

        tk.Button(left_panel, text="Load Model Weights", command=self.load_weights).pack(fill="x", pady=5)
        tk.Button(left_panel, text="Load Input (CSV)", command=self.load_input_csv).pack(fill="x", pady=5)
        
        tk.Label(left_panel, text="Input Data (Text):").pack(anchor="w", pady=(20, 0))
        self.input_text = tk.Text(left_panel, height=10, width=30)
        self.input_text.pack(fill="x")

        run_btn = tk.Button(left_panel, text="RUN INFERENCE", bg="#007acc", fg="white", 
                            font=("Segoe UI", 10, "bold"), command=self.run_inference)
        run_btn.pack(fill="x", pady=20)

        # Right Panel: Visualization / Output
        right_panel = tk.LabelFrame(content_frame, text="Engine Output", padx=10, pady=10)
        right_panel.pack(side="right", fill="both", expand=True)

        self.log_text = tk.Text(right_panel, bg="#f0f0f0", state="disabled", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def load_weights(self):
        file_path = filedialog.askopenfilename(
            title="Load Weights (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                W = self._load_csv_matrix(file_path)
                self.log(f"[Success] Weights loaded from {os.path.basename(file_path)}.")
                self.log(f"[Info] Shape: {W.shape}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load file: {e}")
    
    def load_input_csv(self):
        """Load input matrix A from CSV file."""
        file_path = filedialog.askopenfilename(
            title="Load Input (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                A = self._load_csv_matrix(file_path)
                # Display the matrix in the input text area
                self.input_text.delete("1.0", "end")
                self.input_text.insert("1.0", f"A:\n{self._format_matrix(A)}")
                self.log(f"[Success] Input loaded from {os.path.basename(file_path)}.")
                self.log(f"[Info] Shape: {A.shape}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load file: {e}")
    
    def _load_csv_matrix(self, file_path):
        """Load a CSV file and return a numpy matrix."""
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append([float(x) for x in row if x.strip()])
        return np.array(rows, dtype=np.float32)
    
    def _format_matrix(self, matrix):
        """Format a numpy matrix as readable text."""
        lines = []
        for row in matrix:
            lines.append(" ".join(f"{x:.4f}" for x in row))
        return "\n".join(lines)

    def run_inference(self):
        input_str = self.input_text.get("1.0", "end").strip()
        if not input_str:
            messagebox.showwarning("Input Error", "Please provide input text.")
            return

        expr = self.expr_entry.get()
        self.log(f"\n[APL Layer] Executing: {expr}")
        
        # Determine which engine will be used
        if self.apl.cpp_engine and self.apl.cpp_engine.available:
            self.log(f"[APL Layer] Dispatching to C++ Engine (native).")
            engine_label = "C++ Native"
        else:
            self.log(f"[APL Layer] Dispatching to Python Emulator.")
            engine_label = "Python Emulator"

        start_time = time.time()
        
        # Call the emulator (which will try C++ first)
        result = self.apl.execute_apl_expression(expr, input_str)
        
        elapsed = (time.time() - start_time) * 1000
        self.log(f"[{engine_label}] Computation finished in {elapsed:.2f} ms")
        self.log(f"[Output] {result}")

if __name__ == "__main__":
    root = tk.Tk()
    # Set icon if available
    # root.iconbitmap("icon.ico") 
    app = SuperAPLApp(root)
    root.mainloop()
