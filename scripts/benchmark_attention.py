import argparse
import json
import time
from pathlib import Path

from qwen_asr_cli.audio import audio_info, decode_media, is_silent, iter_chunks
from qwen_asr_cli.model import NativeQwen
from qwen_asr_cli.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser(description="Local matched-backend ASR benchmark")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--name", default="run")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--limit-chunks", type=int)
    args = parser.parse_args()
    local = configure_runtime(offline=not args.online)
    directory = local / "benchmark-attention"
    directory.mkdir(exist_ok=True)
    audio = directory / "decoded.wav"
    if args.prepare:
        started = time.perf_counter()
        decode_media(args.prepare.resolve(), directory)
        metadata = {"audio_seconds": audio_info(audio).duration,
                    "decode_seconds": time.perf_counter() - started}
        (directory / "input.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata))
        return

    import torch

    started = time.perf_counter()
    model = NativeQwen(offline=not args.online, attention=args.attention)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    first = next(chunk for chunk in iter_chunks(audio) if not is_silent(chunk.samples))
    model.transcribe(first.samples, None, 1024)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model.transcribe(first.samples, None, 1024)
    operators = {event.key: event.count for event in profile.key_averages()
                 if any(part in event.key.lower() for part in ("attention", "flash", "flashattn"))}
    if args.attention == "flash_attention_2" and not any(
        "flash" in name.lower() for name in operators
    ):
        raise RuntimeError(f"No FlashAttention operator observed: {operators}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    started = time.perf_counter()
    with (directory / f"{args.name}.jsonl").open("w", encoding="utf-8") as log:
        for index, chunk in enumerate(iter_chunks(audio)):
            if args.limit_chunks and index >= args.limit_chunks:
                break
            begin = time.perf_counter()
            text = ""
            if not is_silent(chunk.samples):
                text = model.transcribe(chunk.samples, None, 1024).text
            torch.cuda.synchronize()
            row = {"index": index, "start": chunk.start, "end": chunk.end,
                   "seconds": time.perf_counter() - begin, "text": text}
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
    elapsed = time.perf_counter() - started
    report = {
        "backend": args.attention, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "chunks": len(rows), "audio_seconds": rows[-1]["end"], "load_seconds": load_seconds,
        "inference_seconds": elapsed, "audio_per_wall_second": rows[-1]["end"] / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "operators": operators,
    }
    (directory / f"{args.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / f"{args.name}.txt").write_text(
        "\n".join(row["text"] for row in rows if row["text"]) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
