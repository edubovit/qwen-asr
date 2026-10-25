$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$root = $env:QWEN_ASR_HOME
$env:UV_PROJECT_ENVIRONMENT = Join-Path $root '.local\venv-flash'
Push-Location $root
try {
    & uv python install (Get-Content '.python-version' -Raw).Trim() --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv sync --locked --extra flash --managed-python
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & '.\.local\venv-flash\Scripts\python.exe' '.\scripts\check_runtime.py'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & '.\.local\venv-flash\Scripts\python.exe' '.\scripts\prefetch_flash.py'
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
