@echo off
setlocal enabledelayedexpansion

echo [Setup] Checking environment...

:: 1. Check for CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] CMake is not found in your PATH.
    echo.
    echo Please install CMake: https://cmake.org/download/
    echo During installation, select "Add CMake to the system PATH".
    echo.
    pause
    exit /b 1
)

:: 2. Check for C++ Compiler (cl.exe)
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo [Setup] C++ compiler not found in PATH. Searching for Visual Studio...
    
    set "found_vs="
    
    :: List of common paths for vcvars64.bat
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        set "found_vs=1"
        goto :build_start
    )
    
    if not defined found_vs (
        echo [Error] C++ Compiler not found.
        echo.
        echo Please install "Visual Studio Build Tools" with the "Desktop development with C++" workload.
        echo Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
        pause
        exit /b 1
    )
)

:build_start
echo.
echo [Build] Creating build directory...
if not exist build mkdir build
cd build

echo [Build] Configuring with CMake...
cmake ..
if %errorlevel% neq 0 (
    echo [Error] CMake configuration failed.
    pause
    exit /b 1
)

echo [Build] Compiling C++ Engine (using all CPU cores)...
cmake --build . --config Release --parallel
if %errorlevel% neq 0 (
    echo [Error] Compilation failed.
    pause
    exit /b 1
)

echo [Build] Done.
cd ..
