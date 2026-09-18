$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install -r crypto_bot\requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onedir --name INTRA `
  --add-data "crypto_bot\web;crypto_bot\web" `
  --hidden-import crypto_bot.terminal_bridge `
  intra_launcher.py

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
  Write-Warning "PyInstaller build completed. Install Inno Setup 6 to create the installer."
  exit 0
}
& $iscc packaging\INTRA.iss
