@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-sim\Scripts\python.exe" (
  echo Run Setup_Sim.ps1 first.
  pause
  exit /b 1
)
echo.
echo Rita - Expanded virtual robot dance
echo 1. Kinova Gen3 6-DOF
echo 2. Unitree G1 humanoid
echo 3. Misty simplified
echo 4. Unitree Go2 quadruped
echo.
choice /c 1234 /n /m "Choose robot [1-4]: "
if errorlevel 4 (set "RITA_SIM_ROBOT=go2") else if errorlevel 3 (set "RITA_SIM_ROBOT=misty") else if errorlevel 2 (set "RITA_SIM_ROBOT=g1") else (set "RITA_SIM_ROBOT=kinova")
".venv-sim\Scripts\python.exe" -X utf8 sim.py dance --robot %RITA_SIM_ROBOT% --profile expressive --bpm 117.02546482191966 --seconds 120
pause
