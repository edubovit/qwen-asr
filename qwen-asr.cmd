@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project environment is missing. Run setup.cmd first. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m qwen_asr_cli %*
exit /b %errorlevel%
