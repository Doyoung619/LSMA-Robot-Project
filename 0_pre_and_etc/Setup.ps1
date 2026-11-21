$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.venv/Scripts/python.exe' rita.py analyze
if ($LASTEXITCODE -ne 0) { throw 'Audio analysis failed; verify ffmpeg is installed' }
