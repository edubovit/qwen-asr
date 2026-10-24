import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".local" / "fixtures"
RESULTS = ROOT / ".local" / "verification"


def normalized_words(text):
    text = text.casefold().replace("ё", "е")
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return text.split()


def word_error_rate(reference, hypothesis):
    expected, actual = normalized_words(reference), normalized_words(hypothesis)
    row = list(range(len(actual) + 1))
    for index, word in enumerate(expected, 1):
        following = [index]
        for column, other in enumerate(actual, 1):
            following.append(min(following[-1] + 1, row[column] + 1,
                                 row[column - 1] + (word != other)))
        row = following
    return row[-1] / len(expected)


def seconds(value):
    hours, minutes, remaining = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(remaining)


def timed_text(text, duration):
    previous = 0.0
    lines = []
    for line in text.splitlines():
        match = re.fullmatch(r"\[([\d:.]+) --> ([\d:.]+)\] (.+)", line)
        assert match, f"Invalid timestamp line: {line}"
        start, end = seconds(match[1]), seconds(match[2])
        assert previous <= start < end <= duration + 0.081, (previous, start, end, duration)
        previous = end
        lines.append(match[3])
    assert lines, "No timed segments"
    return " ".join(lines), len(lines)


def main():
    import torch

    RESULTS.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA unavailable"
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    fixtures = {Path(item["path"]).stem: item for item in manifest["fixtures"]}
    records = []
    completed = False

    def run(name, arguments, *, reference=None, duration=None, output=None, success=True,
            extra_env=None, command=None):
        started = time.perf_counter()
        process = subprocess.run(
            command or [sys.executable, "-m", "qwen_asr_cli", *map(str, arguments)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600,
            env={**os.environ, "PYTHONUTF8": "1", **(extra_env or {})},
        )
        record = {"name": name, "seconds": round(time.perf_counter() - started, 3),
                  "exit_code": process.returncode, "passed": False}
        records.append(record)
        (RESULTS / f"{name}.stdout.txt").write_text(process.stdout, encoding="utf-8")
        (RESULTS / f"{name}.stderr.txt").write_text(process.stderr, encoding="utf-8")
        assert (process.returncode == 0) == success, (name, process.returncode, process.stderr)
        if not success:
            assert not process.stdout, (name, "Failed command emitted a partial transcript")
            assert process.returncode in {2, 3, 4, 5}, (name, "Unexpected error code", process)
            assert "Traceback (most recent call last)" not in process.stderr, (name, process.stderr)
        if reference is not None:
            text = output.read_text(encoding="utf-8") if output else process.stdout
            if output:
                assert not process.stdout, (name, "Unexpected stdout", process.stdout)
            if duration is not None:
                text, record["segments"] = timed_text(text, duration)
            record["wer"] = round(word_error_rate(reference, text), 5)
            record["transcript"] = text.strip()
            assert record["wer"] <= 0.35, (name, record)
        record["passed"] = True
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return process

    try:
        for name, fixture in fixtures.items():
            source = FIXTURES / fixture["path"]
            for timestamps in (False, True):
                case = name + ("_timed" if timestamps else "_plain")
                destination = RESULTS / f"{case}.txt"
                options = [source, "--offline", "--force", "--output", destination]
                if timestamps:
                    options += ["--timestamps"]
                run(case, options, reference=fixture["reference_text"], output=destination,
                    duration=fixture["duration_seconds"] if timestamps else None)

        media = RESULTS / "media"
        for filename, fixture_name in (
            ("English recording.mp3", "en_fleurs"),
            ("English video.mp4", "en_fleurs"),
            ("Русская запись.wav", "ru_fleurs"),
        ):
            case = "media_" + Path(filename).suffix[1:] + "_" + fixture_name
            run(case, [media / filename, "--offline", "--stdout"],
                reference=fixtures[fixture_name]["reference_text"])

        mixed = fixtures["mixed_ru_en"]
        run("repeated_timed", [media / "mixed repeated.wav", "--offline", "--stdout",
                               "--timestamps"],
            reference=" ".join([mixed["reference_text"]] * 4),
            duration=mixed["duration_seconds"] * 4)
        run("forced_english", [FIXTURES / "en_fleurs.wav", "--offline", "--stdout",
                               "--language", "en"],
            reference=fixtures["en_fleurs"]["reference_text"])
        delayed = run(
            "delayed_video", [media / "Delayed English video.mkv", "--offline", "--stdout",
                              "--timestamps"],
            reference=fixtures["en_fleurs"]["reference_text"],
            duration=fixtures["en_fleurs"]["duration_seconds"] + 2,
        )
        assert seconds(re.match(r"\[([\d:.]+)", delayed.stdout)[1]) >= 2

        destination = RESULTS / "protected.txt"
        destination.write_text("KEEP EXISTING CONTENT", encoding="utf-8")
        run("no_overwrite", [FIXTURES / "en_fleurs.wav", "--output", destination], success=False)
        run("generation_cap", [FIXTURES / "en_fleurs.wav", "--output", destination,
                               "--force", "--offline", "--max-new-tokens", "1"], success=False)
        assert destination.read_text(encoding="utf-8") == "KEEP EXISTING CONTENT"
        for name in ("empty.wav", "corrupt.wav", "missing.wav"):
            run("invalid_" + name, [media / name, "--offline", "--stdout"], success=False)
        run("cuda_unavailable", [FIXTURES / "en_fleurs.wav", "--offline", "--stdout"],
            success=False, extra_env={"CUDA_VISIBLE_DEVICES": ""})
        empty_home = RESULTS / "empty-home"
        run("offline_missing_models", [FIXTURES / "en_fleurs.wav", "--offline", "--stdout"],
            success=False, extra_env={"QWEN_ASR_HOME": str(empty_home)})
        default_source = media / "Русская запись.wav"
        default_output = default_source.with_suffix(".txt")
        default_output.unlink(missing_ok=True)
        run("default_destination", [default_source, "--offline"], output=default_output,
            reference=fixtures["ru_fleurs"]["reference_text"])
        run("source_protection", [default_source, "--output", default_source, "--force"],
            success=False)
        empty = run("two_hours_silence", [media / "two hours silence.wav", "--offline",
                                          "--stdout"])
        assert empty.stdout == ""
        run("cmd_launcher", [], reference=fixtures["en_fleurs"]["reference_text"],
            command=["cmd", "/d", "/c", str(ROOT / "qwen-asr.cmd"),
                     str(FIXTURES / "en_fleurs.wav"), "--offline", "--stdout"])
        assert not list((ROOT / ".local" / "tmp").glob("asr-*")), "Leaked audio temp directory"
        assert not list((empty_home / ".local" / "tmp").glob("asr-*")), "Leaked error temp"
        completed = True
    finally:
        report = {
            "python": sys.version, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "cases": records, "completed": completed,
            "normalization": "casefold; ё to е; remove Unicode punctuation; split whitespace",
            "limitations": "Tiny read-speech samples; mixed audio is concatenated, not natural "
                           "code-switching. Two-hour test is silent decoding/chunking, not "
                           "two hours of model inference.",
        }
        (RESULTS / "gpu-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
