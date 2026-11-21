$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required" }
& '.venv\Scripts\python.exe' -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& '.venv\Scripts\python.exe' download_models.py
if ($LASTEXITCODE -ne 0) { throw "Checkpoint preparation failed" }
