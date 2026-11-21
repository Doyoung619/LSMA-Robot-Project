$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
}
& '.venv/Scripts/python.exe' -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependency installation failed' }
& '.venv/Scripts/python.exe' prepare.py --assets deployment --refresh-metadata
if ($LASTEXITCODE -ne 0) { throw 'Pinned source/artifact preparation failed' }
