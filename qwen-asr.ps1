$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\environment.ps1')
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Project environment is missing. Run .\setup.ps1 first.'
}
& $python -m qwen_asr_cli @args
exit $LASTEXITCODE
