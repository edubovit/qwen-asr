import os
import subprocess
import sys

import pytest

from qwen_asr_cli import cli, model
from qwen_asr_cli.alignment import Word
from qwen_asr_cli.errors import AudioError, ModelError, TranscriptionError


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch, write_audio):
    source = tmp_path / "запись с пробелами.wav"
    source.write_bytes(b"source must remain unchanged")
    state = {"loaded": 0, "transcribed": 0, "aligned": 0, "silent": False, "error": None}

    def decode(source, directory):
        print("dependency diagnostic on stdout")
        return write_audio(directory / "decoded.wav", seconds=2.5, silent=state["silent"])

    class FakeModel:
        def __init__(self, offline, progress):
            state["loaded"] += 1
            state["offline"] = offline
            progress("Loading fake model")

        def transcribe(self, samples, language, max_new_tokens):
            state["transcribed"] += 1
            state["language"] = language
            if state["error"] and state["transcribed"] >= 2:
                raise state["error"]
            return model.Transcript("Привет, world!", language or "Russian")

        def align(self, samples, transcript):
            state["aligned"] += 1
            return [Word("Привет", 0.08, 0.2), Word("world", 0.2, 0.4)]

    monkeypatch.setattr(cli, "decode_media", decode)
    monkeypatch.setattr(model, "NativeQwen", FakeModel)
    return source, state


def assert_clean(tmp_path):
    assert not list((tmp_path / ".local" / "tmp").glob("asr-*"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_default_utf8_output_and_single_model(fake_pipeline, tmp_path, capsys):
    source, state = fake_pipeline
    assert cli.main([str(source), "--chunk-seconds", "1"]) == 0
    output = source.with_suffix(".txt")
    assert output.read_bytes() == ("Привет, world!\n" * 3).encode("utf-8")
    assert state["loaded"] == 1 and state["transcribed"] == 3
    assert state["aligned"] == 0
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "Chunk 3:" in stderr and "dependency diagnostic" in stderr
    assert source.read_bytes() == b"source must remain unchanged"
    assert_clean(tmp_path)


def test_stdout_contains_only_finished_transcript(fake_pipeline, tmp_path, capsys):
    source, state = fake_pipeline
    assert cli.main([str(source), "--stdout", "--offline", "--language", "ru"]) == 0
    stdout, stderr = capsys.readouterr()
    assert stdout == "Привет, world!\n"
    assert "Loading fake model" in stderr
    assert state["offline"] is True and state["language"] == "Russian"
    assert not source.with_suffix(".txt").exists()
    assert_clean(tmp_path)


def test_timestamps_use_aligned_word_offsets(fake_pipeline, tmp_path, capsys):
    source, state = fake_pipeline
    assert cli.main([str(source), "--stdout", "--timestamps", "--chunk-seconds", "1"]) == 0
    stdout, _ = capsys.readouterr()
    assert stdout.splitlines() == [
        "[00:00:00.080 --> 00:00:00.400] Привет, world!",
        "[00:00:01.080 --> 00:00:01.400] Привет, world!",
        "[00:00:02.080 --> 00:00:02.400] Привет, world!",
    ]
    assert state["aligned"] == 3
    assert_clean(tmp_path)


def test_silence_never_loads_model_or_invents_text(fake_pipeline, tmp_path, capsys):
    source, state = fake_pipeline
    state["silent"] = True
    assert cli.main([str(source), "--stdout", "--timestamps"]) == 0
    stdout, stderr = capsys.readouterr()
    assert stdout == "" and "Digital silence" in stderr
    assert state["loaded"] == 0
    assert_clean(tmp_path)


@pytest.mark.parametrize("exception,code", [
    (KeyboardInterrupt(), 130), (TranscriptionError("generation cap"), 5),
    (ModelError("missing offline aligner"), 4),
])
def test_failure_keeps_existing_file_and_cleans_everything(
    fake_pipeline, tmp_path, capsys, exception, code
):
    source, state = fake_pipeline
    destination = source.with_suffix(".txt")
    destination.write_bytes(b"old transcript")
    state["error"] = exception
    assert cli.main([str(source), "--force", "--chunk-seconds", "1"]) == code
    assert destination.read_bytes() == b"old transcript"
    assert capsys.readouterr().out == ""
    assert_clean(tmp_path)


def test_failure_does_not_emit_partial_stdout(fake_pipeline, tmp_path, capsys):
    source, state = fake_pipeline
    state["error"] = TranscriptionError("truncated")
    assert cli.main([str(source), "--stdout", "--chunk-seconds", "1"]) == 5
    assert capsys.readouterr().out == ""
    assert_clean(tmp_path)


def test_explicit_output_and_existing_file_protection(fake_pipeline, capsys):
    source, state = fake_pipeline
    destination = source.parent / "explicit.txt"
    destination.write_text("old", encoding="utf-8")
    assert cli.main([str(source), "-o", str(destination)]) == 2
    assert state["loaded"] == 0 and destination.read_text() == "old"
    assert cli.main([str(source), "-o", str(destination), "--force"]) == 0
    assert destination.read_text(encoding="utf-8") == "Привет, world!\n"
    assert capsys.readouterr().out == ""


def test_default_output_with_txt_input_cannot_destroy_source(fake_pipeline):
    source, _ = fake_pipeline
    renamed = source.with_suffix(".txt")
    source.rename(renamed)
    assert cli.main([str(renamed), "--force"]) == 2
    assert renamed.read_bytes() == b"source must remain unchanged"


def test_missing_input_is_useful_error(tmp_path, capsys):
    assert cli.main([str(tmp_path / "missing.wav")]) == 2
    stdout, stderr = capsys.readouterr()
    assert stdout == "" and "existing local file" in stderr


def test_decode_failure_cleans_temp(fake_pipeline, tmp_path, monkeypatch, capsys):
    source, _ = fake_pipeline

    def fail(source, directory):
        (directory / "decoded.wav").write_bytes(b"partial audio")
        raise AudioError("corrupt input")

    monkeypatch.setattr(cli, "decode_media", fail)
    assert cli.main([str(source)]) == 3
    assert "corrupt input" in capsys.readouterr().err
    assert_clean(tmp_path)


@pytest.mark.parametrize("arguments", [
    ["--language", "xx"], ["--chunk-seconds", "nan"], ["--chunk-seconds", "0"],
    ["--chunk-seconds", "61"], ["--max-new-tokens", "0"], ["--max-new-tokens", "8193"],
    ["--output", "output.txt", "--stdout"],
])
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(["input.wav", *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("value,expected", [
    ("auto", None), ("RU", "Russian"), ("Russian", "Russian"),
    ("en", "English"), ("english", "English"),
])
def test_language_aliases(value, expected):
    assert cli.parser().parse_args(["input.wav", "--language", value]).language == expected


def test_help_does_not_import_heavy_libraries():
    code = """
import sys
from qwen_asr_cli.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
heavy = ('torch', 'transformers', 'numpy', 'soundfile', 'librosa')
assert not any(name in sys.modules for name in heavy)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert "--timestamps" in result.stdout


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "qwen_asr_cli", "--help"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "--offline" in result.stdout


def test_offline_tests_do_not_import_actual_torch_or_transformers():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_stdout_is_utf8_in_redirected_console(tmp_path, write_audio):
    source = write_audio(tmp_path / "silence.wav", silent=True)
    environment = os.environ.copy()
    environment["QWEN_ASR_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "qwen_asr_cli", str(source), "--stdout"],
        capture_output=True, env=environment,
    )
    if b"FFmpeg was not found" in result.stderr:
        pytest.skip("FFmpeg is not installed")
    assert result.returncode == 0, result.stderr.decode("utf-8")
    assert result.stdout == b""
    assert "Digital silence" in result.stderr.decode("utf-8")
