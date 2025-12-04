@echo off
REM DuckChat Service Launcher (Windows)
REM Starts the Duck Chat API server with external model and auto-reindex
REM Usage: duck_service.bat [--skip-reindex]

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set CONFIG=%REPO_ROOT%\deploy\duckchat_engine.json
set ADAPTER=testmylora
set HOST=0.0.0.0
set PORT=5000

REM Parse command-line args
set SKIP_REINDEX=
for %%A in (%*) do (
    if "%%A"=="--skip-reindex" set SKIP_REINDEX=--skip-reindex
)

echo.
echo ========================================
echo Duck Chat Service Launcher
echo ========================================
echo Config: %CONFIG%
echo Adapter: %ADAPTER%
echo Host/Port: %HOST%:%PORT%
echo.

REM Activate venv if present
if exist "%REPO_ROOT%\.venv\Scripts\activate.bat" (
    call "%REPO_ROOT%\.venv\Scripts\activate.bat"
    echo [*] Virtual environment activated
)

REM Launch bootstrap
cd /d "%REPO_ROOT%"
python scripts\duck_server_bootstrap.py ^
  --config "%CONFIG%" ^
  --adapter "%ADAPTER%" ^
  --host %HOST% ^
  --port %PORT% ^
  %SKIP_REINDEX%

REM On exit, display status
if %ERRORLEVEL% EQU 0 (
    echo [*] Service exited cleanly
) else (
    echo [!] Service exited with error %ERRORLEVEL%
)
pause
