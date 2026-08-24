# Build the TEMPEST Analysis Suite into a standalone Windows executable.
# Usage:  ./build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import PyInstaller" 2>$null
if (-not $?) { Write-Host "Installing PyInstaller..."; pip install pyinstaller }

Write-Host "Building TempestSuite.exe ..."
pyinstaller --noconfirm tempest_suite.spec

Write-Host ""
Write-Host "Done. Executable: dist\TempestSuite\TempestSuite.exe"
