# Build Instructions

## Prerequisites
1.  **Python 3.x** installed.
2.  **CMake** and a C++ Compiler (Visual Studio or MinGW) for the engine.
3.  **PyInstaller** for creating the EXE (`pip install pyinstaller`).
4.  **(Optional) NVIDIA CUDA Toolkit**: Install this to enable GPU acceleration. If installed, the build system will automatically detect it and compile the GPU kernels.

## Steps

1.  **Build the C++ Engine:**
    You can use the provided script `build_engine.bat` which enables multi-core compilation, or run manually:
    ```powershell
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release --parallel
    cd ..
    ```
    This creates `super_apl_engine.dll` in `build/Release`.

2.  **Build the GUI EXE:**
    Run the `build_exe.bat` script:
    ```powershell
    .\build_exe.bat
    ```

3.  **Run:**
    The executable will be in the `dist` folder. 
    *Note:* You must copy `super_apl_engine.dll` to the same folder as `SuperAPLModel.exe` for the hybrid architecture to work fully, otherwise it runs in emulation mode.
