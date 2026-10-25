import json
import sys
from types import SimpleNamespace

import pytest

from qwen_asr_cli import flash


@pytest.mark.parametrize("version,cuda", [("2.9.1", "12.8"), ("2.10.0", "13.0")])
def test_kernel_rejects_wrong_abi(monkeypatch, version, cuda):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__=version, version=SimpleNamespace(cuda=cuda),
    ))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=None))
    with pytest.raises(RuntimeError, match="requires PyTorch 2.10"):
        flash.prepare_flash_kernel(True)


def test_legacy_metadata_is_adapted_without_changing_cached_binary(tmp_path, monkeypatch):
    source = tmp_path / "cache"
    build = source / "build" / "torch210-cu128-x86_64-windows"
    build.mkdir(parents=True)
    metadata = {"version": 1, "python-depends": []}
    (build / "metadata.json").write_text(json.dumps(metadata))
    (build / "_flash_attn2_d6b301e.pyd").write_bytes(b"binary fixture")
    monkeypatch.setattr(flash, "workspace_root", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_KERNELS", "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"),
    ))

    def download(repo, **options):
        assert repo == flash.KERNEL_REPO
        assert options["revision"] == flash.KERNEL_REVISION
        assert options["local_files_only"]
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(flash.hashlib, "file_digest", lambda *args: SimpleNamespace(
        hexdigest=lambda: "d6a62d14386a4a156301af046a57bb718dc8ef7f103dd01b9529fc7c276b5e8a",
    ))
    flash.prepare_flash_kernel(True)
    target = tmp_path / ".local" / "flash-kernel" / flash.KERNEL_REVISION
    assert json.loads((build / "metadata.json").read_text()) == metadata
    assert (target / "_flash_attn2_d6b301e.pyd").read_bytes() == b"binary fixture"
    assert json.loads((target / "metadata.json").read_text())["name"] == "flash-attn2"
    assert (target / "ready").exists()
