import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["HF_HOME"] = str(root / ".local" / "models")
    os.environ["HF_HUB_CACHE"] = str(root / ".local" / "models" / "hub")
    os.environ["HF_XET_CACHE"] = str(root / ".local" / "models" / "xet")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import snapshot_download

    for model, revision in (
        ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"),
        ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "c07281df297b9905d24a508279258cccf987a064"),
    ):
        path = snapshot_download(
            model,
            revision=revision,
            allow_patterns=["*.json", "*.jinja", "*.safetensors"],
            max_workers=2,
        )
        print(f"{model}: {path}", flush=True)


if __name__ == "__main__":
    main()
