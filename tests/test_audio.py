import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from qwen_asr_cli import audio
from qwen_asr_cli.errors import AudioError


@pytest.mark.parametrize(
    "frames,seconds", [(1, 1), (16000, 1), (32000, 1), (170003, 3), (99001, 1.23)]
)
def test_chunks_cover_every_sample_exactly(tmp_path, frames, seconds):
    samples = np.random.default_rng(18).uniform(-0.2, 0.2, frames).astype(np.float32)
    samples[35000:47000] = 0
    path = tmp_path / "coverage.wav"
    sf.write(path, samples, audio.SAMPLE_RATE, subtype="PCM_16")
    expected, _ = sf.read(path, dtype="float32")
    chunks = list(audio.iter_chunks(path, seconds))
    position = 0
    for chunk in chunks:
        assert chunk.start_sample == position
        assert 0 < len(chunk.samples) <= round(seconds * audio.SAMPLE_RATE)
        position += len(chunk.samples)
    assert position == frames
    np.testing.assert_array_equal(np.concatenate([chunk.samples for chunk in chunks]), expected)


def test_boundary_favors_sustained_silence():
    samples = np.full(30 * audio.SAMPLE_RATE, 0.2, dtype=np.float32)
    samples[27 * audio.SAMPLE_RATE : 28 * audio.SAMPLE_RATE] = 0
    boundary = audio.silence_boundary(samples)
    assert 27 * audio.SAMPLE_RATE < boundary < 28 * audio.SAMPLE_RATE


def test_continuous_speech_keeps_nominal_boundary():
    samples = np.full(30 * audio.SAMPLE_RATE, 0.2, dtype=np.float32)
    assert audio.silence_boundary(samples) == len(samples)


def test_short_click_is_not_a_silence_boundary():
    samples = np.full(30 * audio.SAMPLE_RATE, 0.2, dtype=np.float32)
    samples[27 * audio.SAMPLE_RATE : 27 * audio.SAMPLE_RATE + 100] = 0
    assert audio.silence_boundary(samples) == len(samples)


def test_multi_hour_reader_is_bounded(monkeypatch):
    total = 3 * 3600 * audio.SAMPLE_RATE + 17
    reads = []

    class SyntheticAudio:
        def __enter__(self):
            self.position = 0
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return total

        def seek(self, position):
            self.position = position

        def read(self, frames, **kwargs):
            reads.append(frames)
            return np.zeros(min(frames, total - self.position), dtype=np.float32)

    monkeypatch.setattr(audio, "audio_info", lambda path: audio.AudioInfo(total))
    monkeypatch.setattr(sf, "SoundFile", lambda path: SyntheticAudio())
    covered = 0
    for chunk in audio.iter_chunks("synthetic", 30):
        assert chunk.start_sample == covered
        covered += len(chunk.samples)
    assert covered == total
    assert max(reads) == 30 * audio.SAMPLE_RATE
    assert len(reads) > 360


@pytest.mark.parametrize("seconds", [0, -1, 61, float("nan"), float("inf")])
def test_chunk_parameter_rejected(seconds):
    with pytest.raises(ValueError):
        next(audio.iter_chunks("unused", seconds))


def test_silence_is_conservative():
    assert audio.is_silent(np.zeros(16000, dtype=np.float32))
    assert audio.is_silent(np.full(16000, 1 / 32768, dtype=np.float32))
    assert not audio.is_silent(np.full(16000, 0.0001, dtype=np.float32))


def test_substantial_pause_splits_even_a_short_final_window(tmp_path):
    samples = np.full(20 * audio.SAMPLE_RATE, 0.2, dtype=np.float32)
    samples[10 * audio.SAMPLE_RATE : int(10.5 * audio.SAMPLE_RATE)] = 0
    source = tmp_path / "language change.wav"
    sf.write(source, samples, audio.SAMPLE_RATE, subtype="PCM_16")
    chunks = list(audio.iter_chunks(source, 30))
    assert len(chunks) == 2
    assert 10 < chunks[0].end < 10.5
    assert chunks[0].end == chunks[1].start
    assert sum(len(chunk.samples) for chunk in chunks) == len(samples)


def test_short_pauses_and_edge_silence_do_not_fragment_utterances():
    samples = np.full(20 * audio.SAMPLE_RATE, 0.2, dtype=np.float32)
    samples[:2 * audio.SAMPLE_RATE] = 0
    samples[-2 * audio.SAMPLE_RATE:] = 0
    samples[10 * audio.SAMPLE_RATE : int(10.1 * audio.SAMPLE_RATE)] = 0
    assert audio.pause_boundary(samples) == len(samples)
    assert audio.pause_boundary(np.zeros_like(samples)) == len(samples)


def test_audio_info_rejects_empty_and_wrong_format(tmp_path, write_audio):
    with pytest.raises(AudioError, match="no audio samples"):
        audio.audio_info(write_audio(tmp_path / "empty.wav", seconds=0))
    with pytest.raises(AudioError, match="mono 16 kHz"):
        audio.audio_info(write_audio(tmp_path / "stereo.wav", channels=2))
    with pytest.raises(AudioError, match="Cannot read"):
        audio.audio_info(tmp_path / "absent.wav")


def test_ffmpeg_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(AudioError, match="FFmpeg was not found"):
        audio.decode_media(tmp_path / "input.wav", tmp_path)


def test_ffmpeg_cancel_kills_child(tmp_path, monkeypatch):
    class Process:
        killed = False
        waits = 0

        def wait(self):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -1

        def kill(self):
            self.killed = True

    process = Process()
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return process

    monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(audio, "_container_start", lambda source, directory: 7.25)
    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        audio.decode_media(tmp_path / "запись с пробелами.wav", tmp_path)
    assert process.killed and process.waits == 2
    assert "-nostdin" in commands[0]
    assert "-rf64" in commands[0]
    assert "-copyts" in commands[0] and "-start_at_zero" not in commands[0]
    assert commands[0][commands[0].index("-itsoffset") + 1] == "-7.250000000"
    assert commands[0][commands[0].index("-protocol_whitelist") + 1] == "file,pipe"
    assert commands[0][commands[0].index("-i") + 1].endswith("запись с пробелами.wav")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.parametrize("kind", ["wav", "mp3", "mkv"])
def test_real_ffmpeg_audio_and_video(tmp_path, write_audio, kind):
    source = write_audio(
        tmp_path / "исходник с пробелами.wav", seconds=0.5, samplerate=44100, channels=2
    )
    executable = shutil.which("ffmpeg")
    if kind != "wav":
        target = tmp_path / f"видео и звук.{kind}"
        command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if kind == "mkv":
            command += [
                "-f", "lavfi", "-i", "color=c=black:s=16x16:r=10", "-i", str(source),
                "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-shortest",
            ]
        else:
            command += ["-i", str(source)]
        subprocess.run(command + [str(target)], check=True, capture_output=True)
        source = target
    decoded = audio.decode_media(source, tmp_path)
    info = audio.audio_info(decoded)
    assert 0.45 <= info.duration <= 0.65
    assert not audio.is_silent(next(audio.iter_chunks(decoded)).samples)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_real_ffmpeg_rejects_corrupt_media(tmp_path):
    source = tmp_path / "corrupt.mp3"
    source.write_bytes(b"not an audio file")
    with pytest.raises(AudioError, match="FFprobe could not inspect"):
        audio.decode_media(source, tmp_path)


def test_ffprobe_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "ffprobe" else "ffmpeg.exe")
    with pytest.raises(AudioError, match="FFprobe was not found"):
        audio.decode_media(tmp_path / "input.wav", tmp_path)


@pytest.mark.parametrize("metadata,expected", [
    ('{"format": {"start_time": "7.250000"}}', 7.25),
    ('{"format": {"start_time": "-0.007000"}}', -0.007),
    ('{"format": {}}', 0),
])
def test_probe_uses_container_origin_and_local_protocols(tmp_path, monkeypatch, metadata, expected):
    def run(command, **kwargs):
        assert command[command.index("-show_entries") + 1] == "format=start_time"
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert kwargs["stdin"] == subprocess.DEVNULL
        return SimpleNamespace(returncode=0, stdout=metadata)

    monkeypatch.setattr(shutil, "which", lambda name: "ffprobe.exe")
    monkeypatch.setattr(subprocess, "run", run)
    assert audio._container_start(tmp_path / "input.mkv", tmp_path) == expected


@pytest.mark.parametrize("metadata", ["not JSON", '{"format":{"start_time":"nan"}}'])
def test_invalid_container_time_is_rejected(tmp_path, monkeypatch, metadata):
    monkeypatch.setattr(shutil, "which", lambda name: "ffprobe.exe")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=metadata)
    )
    with pytest.raises(AudioError, match="container start time"):
        audio._container_start(tmp_path / "input.mkv", tmp_path)


def run_ffmpeg(arguments):
    subprocess.run(
        [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", *arguments],
        check=True, capture_output=True,
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/FFprobe not installed"
)
@pytest.mark.parametrize("container_start", [0, 7.25])
@pytest.mark.parametrize("kind", ["mkv", "ts"])
def test_real_delayed_audio_uses_container_timeline(tmp_path, write_audio, container_start, kind):
    source = write_audio(tmp_path / "tone.wav", seconds=1)
    media = tmp_path / f"delayed.{kind}"
    codecs = ["-c:v", "mpeg4", "-c:a", "pcm_s16le"]
    if kind == "ts":
        codecs = [
            "-c:v", "mpeg2video", "-bf", "0", "-c:a", "mp2", "-ar", "48000",
            "-muxdelay", "0", "-muxpreload", "0", "-mpegts_copyts", "1",
            "-avoid_negative_ts", "disabled",
        ]
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=16x16:r=10:d=4",
        "-itsoffset", "2", "-i", str(source), "-map", "0:v:0", "-map", "1:a:0",
        *codecs, "-output_ts_offset", str(container_start), str(media),
    ])
    assert audio._container_start(media, tmp_path) == pytest.approx(container_start, abs=0.001)
    decoded = audio.decode_media(media, tmp_path)
    samples, sample_rate = sf.read(decoded, dtype="float32")
    assert sample_rate == audio.SAMPLE_RATE
    assert len(samples) / sample_rate == pytest.approx(3, abs=0.02)
    assert audio.is_silent(samples[: int(1.95 * sample_rate)])
    assert not audio.is_silent(samples[int(2.05 * sample_rate) : int(2.95 * sample_rate)])
    first_signal = np.flatnonzero(np.abs(samples) > 0.01)[0] / sample_rate
    assert first_signal == pytest.approx(2, abs=0.002)
    if kind == "mkv":
        original, _ = sf.read(source, dtype="float32")
        assert len(samples) == 3 * sample_rate
        np.testing.assert_array_equal(samples[: 2 * sample_rate], 0)
        np.testing.assert_array_equal(samples[2 * sample_rate :], original)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/FFprobe not installed"
)
@pytest.mark.parametrize("gap", [0, 0.02, 1, 3, 12])
@pytest.mark.parametrize("container_start", [0, 7.25])
def test_real_internal_packet_gap_is_silence_not_removed(
    tmp_path, write_audio, gap, container_start
):
    source = write_audio(tmp_path / "tone.wav", seconds=2)
    media = tmp_path / "packet gap.mka"
    run_ffmpeg([
        "-i", str(source), "-af",
        f"asetnsamples=n=160:p=0,asetpts=PTS+gte(N\\,16000)*{gap}/TB",
        "-c:a", "pcm_s16le", "-output_ts_offset", str(container_start), str(media),
    ])
    assert audio._container_start(media, tmp_path) == pytest.approx(container_start, abs=0.001)
    decoded = audio.decode_media(media, tmp_path)
    samples, sample_rate = sf.read(decoded, dtype="float32")
    original, _ = sf.read(source, dtype="float32")
    one_second = audio.SAMPLE_RATE
    after_gap = one_second + round(gap * one_second)
    assert sample_rate == one_second
    assert len(samples) == 2 * one_second + round(gap * one_second)
    np.testing.assert_array_equal(samples[:one_second], original[:one_second])
    np.testing.assert_array_equal(samples[one_second:after_gap], 0)
    np.testing.assert_array_equal(samples[after_gap:], original[one_second:])


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/FFprobe not installed"
)
def test_real_transport_stream_gap_is_not_sanitized(tmp_path, write_audio):
    source = write_audio(tmp_path / "tone.wav", seconds=2, samplerate=48000)
    media = tmp_path / "discontinuous.ts"
    run_ffmpeg([
        "-i", str(source), "-af", "asetnsamples=n=480:p=0,asetpts=PTS+gte(N\\,48000)*12/TB",
        "-c:a", "mp2", "-muxdelay", "0", "-muxpreload", "0", "-mpegts_copyts", "1",
        "-avoid_negative_ts", "disabled", "-output_ts_offset", "7.25", str(media),
    ])
    decoded = audio.decode_media(media, tmp_path)
    samples, sample_rate = sf.read(decoded, dtype="float32")
    assert len(samples) / sample_rate == pytest.approx(14, abs=0.05)
    assert not audio.is_silent(samples[int(0.1 * sample_rate) : int(0.9 * sample_rate)])
    assert audio.is_silent(samples[int(1.1 * sample_rate) : int(12.9 * sample_rate)])
    assert not audio.is_silent(samples[int(13.1 * sample_rate) : int(13.9 * sample_rate)])
