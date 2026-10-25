from qwen_asr_cli.runtime import configure_runtime


def main():
    configure_runtime()

    from qwen_asr_cli.flash import prepare_flash_kernel

    prepare_flash_kernel(offline=False)
    print("Pinned Windows FlashAttention 2 kernel cached locally.")


if __name__ == "__main__":
    main()
