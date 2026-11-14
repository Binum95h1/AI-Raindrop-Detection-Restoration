@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto setup_missing
echo Grid 16x3 is an inferred comparison mode; original preprocessing code is not in the uploaded ZIP.
".venv\Scripts\python.exe" test_raindrop.py --mode grid --cols 16 --rows 3 --imgsz 256 --conf 0.25 --device cpu
if errorlevel 1 goto failed
start "" "%~dp0output"
pause
exit /b 0
:setup_missing
echo Run 01_SETUP_CPU.cmd first.
pause
exit /b 1
:failed
echo TEST FAILED. Read the error above. Put images in the input folder or pass --input to test_raindrop.py.
pause
exit /b 1
