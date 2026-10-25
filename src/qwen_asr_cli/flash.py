import hashlib
import json
import os
import shutil
from pathlib import Path

from .runtime import workspace_root

KERNEL_REPO = "kernels-community/flash-attn2"
KERNEL_REVISION = "c3f3c93a4a39e3d2cf49c13eac49229bc726ea13"


def prepare_flash_kernel(offline):
    import torch
    from huggingface_hub import snapshot_download

    if not torch.__version__.startswith("2.10.") or torch.version.cuda != "12.8":
        raise RuntimeError("This pinned Windows kernel requires PyTorch 2.10 / CUDA 12.8. "
                           "Use setup-flash.cmd and .local/venv-flash/Scripts/python.exe.")
    directory = snapshot_download(
        KERNEL_REPO, revision=KERNEL_REVISION, repo_type="model",
        allow_patterns=["build/torch210-cu128-x86_64-windows/**"],
        local_files_only=offline,
    )
    source = Path(directory) / "build" / "torch210-cu128-x86_64-windows"
    target = workspace_root() / ".local" / "flash-kernel" / KERNEL_REVISION
    if not (target / "ready").exists():
        with (source / "_flash_attn2_d6b301e.pyd").open("rb") as binary:
            digest = hashlib.file_digest(binary, "sha256").hexdigest()
        if digest != "d6a62d14386a4a156301af046a57bb718dc8ef7f103dd01b9529fc7c276b5e8a":
            raise ValueError("FlashAttention binary hash differs from the pinned Hub artifact")
        shutil.copytree(source, target, dirs_exist_ok=True)
        metadata = json.loads((target / "metadata.json").read_text())
        metadata.update(name="flash-attn2", id="_flash_attn2_d6b301e", license="BSD-3-Clause",
                        backend={"type": "cuda", "archs": []})
        (target / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (target / "ready").touch()
    os.environ["LOCAL_KERNELS"] = f"{KERNEL_REPO}={os.path.relpath(target)}"
