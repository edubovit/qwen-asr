$env:QWEN_ASR_HOME = Split-Path $PSScriptRoot -Parent
$local = Join-Path $env:QWEN_ASR_HOME '.local'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $local 'python'
$env:UV_PYTHON_BIN_DIR = Join-Path $local 'bin'
$env:UV_CACHE_DIR = Join-Path $local 'uv-cache'
$env:UV_PYTHON_CACHE_DIR = Join-Path $local 'python-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:QWEN_ASR_HOME '.venv'
$env:HF_HOME = Join-Path $local 'models'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
$env:TORCH_HOME = Join-Path $local 'torch'
$env:NUMBA_CACHE_DIR = Join-Path $local 'numba-cache'
$env:CUDA_CACHE_PATH = Join-Path $local 'cuda-cache'
$env:XDG_CACHE_HOME = Join-Path $local 'cache'
$env:MPLCONFIGDIR = Join-Path $local 'matplotlib'
$env:TEMP = Join-Path $local 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
foreach ($path in @($env:UV_PYTHON_INSTALL_DIR, $env:UV_CACHE_DIR, $env:HF_HOME,
                    $env:TORCH_HOME, $env:NUMBA_CACHE_DIR, $env:XDG_CACHE_HOME,
                    $env:MPLCONFIGDIR, $env:TEMP)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}
