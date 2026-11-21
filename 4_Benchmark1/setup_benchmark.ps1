$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated dance benchmark environment.' }
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependencies failed to install.' }
& '.venv/Scripts/python.exe' setup_external.py
if ($LASTEXITCODE -ne 0) { throw 'Pinned Reachy source setup failed.' }
