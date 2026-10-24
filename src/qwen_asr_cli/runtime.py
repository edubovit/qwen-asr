import os
import tempfile
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.environ.get("QWEN_ASR_HOME", Path(__file__).resolve().parents[2])).resolve()


def configure_runtime(offline: bool = False) -> Path:
    local = workspace_root() / ".local"
    directories = {
        "HF_HOME": local / "models",
        "HF_HUB_CACHE": local / "models" / "hub",
        "HF_XET_CACHE": local / "models" / "xet",
        "UV_CACHE_DIR": local / "uv-cache",
        "UV_PYTHON_INSTALL_DIR": local / "python",
        "TORCH_HOME": local / "torch",
        "CUDA_CACHE_PATH": local / "cuda-cache",
        "XDG_CACHE_HOME": local / "cache",
        "NUMBA_CACHE_DIR": local / "numba-cache",
        "MPLCONFIGDIR": local / "matplotlib",
        "TMP": local / "tmp",
        "TEMP": local / "tmp",
        "TMPDIR": local / "tmp",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    tempfile.tempdir = str(local / "tmp")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return local
