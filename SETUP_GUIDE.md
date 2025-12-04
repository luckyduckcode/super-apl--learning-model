# Environment Setup Guide

## 1. Install Visual Studio Build Tools (Required for C++)
The C++ engine requires a compiler. The easiest way on Windows is to install the **Visual Studio Build Tools**.

1.  **Download**: [Direct Link to Installer (vs_BuildTools.exe)](https://aka.ms/vs/17/release/vs_BuildTools.exe)
2.  **Run the Installer**.
3.  **Select Workload**: 
    *   When the installer opens, you will see a "Workloads" tab.
    *   Look for **"Desktop development with C++"** (it has a C++ icon).
    *   **Check the box** next to it.
    *   On the right side, ensure "MSVC ... C++ x64/x86 build tools" and "Windows 10/11 SDK" are selected (they are by default).
4.  **Install**: Click the "Install" button in the bottom right.

## 2. Install CMake (Required for Build System)
CMake is used to generate the build files.

1.  **Download**: [Direct Link to Installer (cmake-msi)](https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-windows-x86_64.msi)
2.  **Install**:
    *   Run the MSI installer.
    *   **IMPORTANT**: During installation, you will see "Install Options".
    *   Select **"Add CMake to the system PATH for all users"**.
    *   If you miss this, the build script won't find CMake.

## 3. Install CUDA (Optional - For GPU Support)
If you have an NVIDIA GPU and want to use it:

1.  **Download**: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2.  **Install**: Follow the standard installation steps.

## 4. Restart
After installing these tools, **restart your terminal** (or VS Code) so the new PATH settings take effect.

## 5. Run Build Again
Run `.\build_engine.bat` again.
