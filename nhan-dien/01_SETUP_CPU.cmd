@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto install
echo Creating Python environment...
py -3.11 -m venv ".venv" >nul 2>&1
if not errorlevel 1 goto install
py -3.12 -m venv ".venv" >nul 2>&1
if not errorlevel 1 goto install
python -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)]" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv ".venv"
if errorlevel 1 goto failed
:install
echo Installing CPU PyTorch. Internet is required for setup.
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__)"
if errorlevel 1 goto failed
echo.
echo SETUP OK. Run 02_TEST_FULL.cmd to test the configured dataset folder.
pause
exit /b 0
:python_missing
echo Install Python 3.11 or 3.12 for Windows, with Add python.exe to PATH enabled.
echo Then run this file again.
pause
exit /b 1
:failed
echo.
echo SETUP FAILED. Read the error above. Check Internet and available disk space.
pause
exit /b 1
