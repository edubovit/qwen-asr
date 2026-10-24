$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\environment.ps1')
foreach ($command in @('uv', 'ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required executable not found on PATH: $command"
    }
}
Push-Location $PSScriptRoot
try {
    & uv python install (Get-Content '.python-version' -Raw).Trim() --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv sync --locked --managed-python
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & '.\.venv\Scripts\python.exe' '.\scripts\check_runtime.py'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable. Check the Windows NVIDIA driver and CUDA-enabled PyTorch installation.' }
} finally {
    Pop-Location
}
