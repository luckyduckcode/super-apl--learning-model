@echo off
echo [Build] Building Duck.exe...

:: Ensure the dist directory exists
if not exist dist mkdir dist

:: Build using PyInstaller
:: We include the app.py (logic), duck_personality.json (data), and name it "Duck"
pyinstaller --noconfirm --onefile --windowed ^
    --name "Duck" ^
    --add-data "src/gui;." ^
    --add-data "src/training/duck_personality.json;." ^
    --paths "src/gui" ^
    src/gui/duck_app.py

echo.
echo [Build] Copying Engine DLL...
:: Copy the C++ engine if it exists
if exist "build\Release\super_apl_engine.dll" (
    copy "build\Release\super_apl_engine.dll" "dist\"
    echo [Success] Engine DLL copied.
) else (
    echo [Warning] C++ Engine DLL not found in build/Release. Duck will run in emulation mode.
)

echo.
echo [Build] Duck.exe created in 'dist' folder.
pause
