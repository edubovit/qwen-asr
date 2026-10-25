import argparse
import math
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from .alignment import aligned_segments, format_segment, format_time
from .audio import audio_info, decode_media, is_silent, iter_chunks
from .errors import CliError, InputError
from .output import publish_output, validate_output
from .runtime import configure_runtime


def language_value(value: str) -> str | None:
    languages = {
        "auto": None, "ru": "Russian", "russian": "Russian",
        "en": "English", "english": "English",
    }
    try:
        return languages[value.casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError("choose auto, ru/Russian, or en/English") from None


def chunk_value(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number between 1 and 60") from None
    if not math.isfinite(number) or not 1 <= number <= 60:
        raise argparse.ArgumentTypeError("must be between 1 and 60 seconds")
    return number


def token_value(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 8192") from None
    if not 1 <= number <= 8192:
        raise argparse.ArgumentTypeError("must be between 1 and 8192")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="qwen-asr",
        description="Transcribe one local audio/video file with Qwen3-ASR on NVIDIA CUDA."
    )
    result.add_argument("media", type=Path, help="local audio or video file")
    result.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa",
                        help="attention backend (default: sdpa; FlashAttention needs flash extras)")
    result.add_argument(
        "--timestamps", action="store_true", help="add forced-aligned segment start/end times"
    )
    result.add_argument(
        "--language", type=language_value, default="auto", metavar="auto|ru|en",
        help="auto-detect (default), or force Russian/English",
    )
    output = result.add_mutually_exclusive_group()
    output.add_argument(
        "--output", "-o", type=Path, help="output UTF-8 text file (default: input stem + .txt)"
    )
    output.add_argument(
        "--stdout", action="store_true", help="write only the finished transcript to stdout"
    )
    result.add_argument(
        "--force", action="store_true", help="replace an existing output, never the source"
    )
    result.add_argument(
        "--offline", action="store_true",
        help="use pinned models from the local cache; no downloads",
    )
    result.add_argument(
        "--chunk-seconds", type=chunk_value, default=30.0, metavar="SECONDS",
        help="maximum chunk duration, 1–60 (default: 30); prefer silence boundaries",
    )
    result.add_argument(
        "--max-new-tokens", type=token_value, default=1024, metavar="N",
        help="generation cap per chunk (default: 1024); reaching it is an error",
    )
    return result


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def transcribe_to_file(args, source: Path, directory: Path, staged: Path) -> None:
    progress(f"Decoding {source.name} to mono 16 kHz audio…")
    decoded = decode_media(source, directory)
    info = audio_info(decoded)
    progress(
        f"Audio duration: {format_time(info.duration)}. Maximum chunk: {args.chunk_seconds:g}s."
    )
    model = None
    with staged.open("w", encoding="utf-8", newline="\n") as output:
        for index, chunk in enumerate(iter_chunks(decoded, args.chunk_seconds), 1):
            progress(
                f"Chunk {index}: {format_time(chunk.start)}–{format_time(chunk.end)} "
                f"/ {format_time(info.duration)}"
            )
            if is_silent(chunk.samples):
                progress("  Digital silence; skipped.")
                continue
            if model is None:
                from .model import NativeQwen

                options = {"offline": args.offline, "progress": progress}
                if args.attention != "sdpa":
                    options["attention"] = args.attention
                model = NativeQwen(**options)
            transcript = model.transcribe(chunk.samples, args.language, args.max_new_tokens)
            if not transcript.text:
                continue
            if args.timestamps:
                words = model.align(chunk.samples, transcript)
                segments = aligned_segments(transcript.text, words, chunk.duration, chunk.start)
                for segment in segments:
                    output.write(format_segment(segment))
            else:
                output.write(transcript.text + "\n")


def run(args) -> None:
    source = args.media.expanduser().absolute()
    if not source.is_file():
        raise InputError(f"Input is not an existing local file: {source}")
    destination = None
    if not args.stdout:
        destination = (args.output or source.with_suffix(".txt")).expanduser().absolute()
    if destination is not None:
        validate_output(source, destination, args.force)
    local = configure_runtime(args.offline)
    with tempfile.TemporaryDirectory(prefix="asr-", dir=local / "tmp") as temporary:
        directory = Path(temporary)
        staged = directory / "transcript.txt"
        with redirect_stdout(sys.stderr):
            transcribe_to_file(args, source, directory, staged)
        if destination is None:
            with staged.open("r", encoding="utf-8") as transcript:
                shutil.copyfileobj(transcript, sys.stdout)
            sys.stdout.flush()
        else:
            publish_output(staged, source, destination, args.force)
            progress(f"Saved {destination}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        progress("Cancelled. No partial transcript was saved.")
        return 130
    except CliError as exc:
        progress(f"Error: {exc}")
        return exc.exit_code
    except OSError as exc:
        progress(f"File I/O error: {exc}")
        return 2
    return 0
