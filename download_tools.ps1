# This script helps you download the necessary tools.

Write-Host "Opening Visual Studio Build Tools Download Page..." -ForegroundColor Cyan
Start-Process "https://aka.ms/vs/17/release/vs_BuildTools.exe"

Write-Host "Opening CMake Download Page..." -ForegroundColor Cyan
Start-Process "https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-windows-x86_64.msi"

Write-Host "`n==================================================================" -ForegroundColor Yellow
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Run the 'vs_BuildTools.exe' that just downloaded."
Write-Host "2. In the installer, check the box for 'Desktop development with C++'."
Write-Host "   (It is usually the top-left option under 'Desktop & Mobile')."
Write-Host "3. Click 'Install' (bottom right)."
Write-Host "4. Run the 'cmake-*-windows-x86_64.msi' that just downloaded."
Write-Host "5. IMPORTANT: Select 'Add CMake to the system PATH for all users'."
Write-Host "6. Restart your computer or VS Code after installation."
Write-Host "==================================================================`n" -ForegroundColor Yellow
