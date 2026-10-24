import sys

import torch


def main():
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}; available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    left = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    assert (left @ left).float().mean().item() == 32.0
    print("CUDA BF16 matrix multiplication: passed")


if __name__ == "__main__":
    main()
