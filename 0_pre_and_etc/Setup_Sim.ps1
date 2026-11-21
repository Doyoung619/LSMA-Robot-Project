param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv-sim/Scripts/python.exe')) {
    & $Python -m venv --system-site-packages .venv-sim
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the simulation environment' }
}
& '.venv-sim/Scripts/python.exe' -m pip install --no-cache-dir -r requirements-sim.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install simulation dependencies' }
& '.venv-sim/Scripts/python.exe' -X utf8 sim.py setup
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare robot models' }
& '.venv-sim/Scripts/python.exe' -X utf8 sim.py check
if ($LASTEXITCODE -ne 0) { throw 'Model checks failed' }
Write-Output 'Ready. Double-click Play_Sim.cmd to choose a robot.'
