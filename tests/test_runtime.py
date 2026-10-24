import os
import tempfile
from pathlib import Path

from qwen_asr_cli import runtime


def test_default_workspace_comes_from_package_not_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("QWEN_ASR_HOME")
    monkeypatch.chdir(tmp_path)
    assert runtime.workspace_root() == Path(runtime.__file__).resolve().parents[2]


def test_runtime_overrides_external_caches_before_imports(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "C:/unwanted-external-cache")
    local = runtime.configure_runtime(offline=True)
    assert local == tmp_path / ".local"
    for name in (
        "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "NUMBA_CACHE_DIR", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "TMP", "TEMP", "TMPDIR",
    ):
        value = Path(os.environ[name])
        assert value.is_relative_to(local) and value.is_dir()
    assert tempfile.gettempdir() == str(local / "tmp")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
