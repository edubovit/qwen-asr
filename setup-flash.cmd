@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_flash.ps1"
exit /b %errorlevel%
