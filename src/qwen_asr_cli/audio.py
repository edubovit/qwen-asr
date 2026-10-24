import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import AudioError

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioInfo:
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / SAMPLE_RATE


@dataclass(frozen=True)
class AudioChunk:
    start_sample: int
    samples: "np.ndarray"

    @property
    def start(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.start + self.duration


def _log_tail(path: Path) -> str:
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 8_000))
        return log.read().decode("utf-8", errors="replace").strip()


def _container_start(source: Path, directory: Path) -> float:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise AudioError("FFprobe was not found. It is required to preserve the media timeline.")
    log_path = directory / "ffprobe.log"
    command = [
        executable, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_entries", "format=start_time", "-of", "json", "-i", str(source),
    ]
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            raise AudioError(f"Could not start FFprobe: {exc}") from exc
    if result.returncode:
        raise AudioError(
            f"FFprobe could not inspect the input (exit {result.returncode}): {_log_tail(log_path)}"
        )
    try:
        start = float(json.loads(result.stdout).get("format", {}).get("start_time", 0))
    except (ValueError, TypeError) as exc:
        raise AudioError(f"FFprobe returned an invalid container start time: {exc}") from exc
    if not math.isfinite(start):
        raise AudioError("FFprobe returned a non-finite container start time.")
    return start


def decode_media(source: Path, directory: Path) -> Path:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise AudioError(
            "FFmpeg was not found. Install FFmpeg or make its existing bin directory available."
        )
    decoded = directory / "decoded.wav"
    log_path = directory / "ffmpeg.log"
    start = _container_start(source, directory)
    command = [
        executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
        "-copyts", "-itsoffset", f"{-start:.9f}",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-af", f"aresample={SAMPLE_RATE}:async=1:first_pts=0:min_comp=0.001:min_hard_comp=0.001",
        "-c:a", "pcm_s16le", "-rf64", "auto", "-y", str(decoded),
    ]
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
            )
        except OSError as exc:
            raise AudioError(f"Could not start FFmpeg: {exc}") from exc
        try:
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if return_code:
        raise AudioError(
            f"FFmpeg could not decode the input (exit {return_code}): {_log_tail(log_path)}"
        )
    return decoded


def audio_info(path: Path) -> AudioInfo:
    import soundfile as sf

    try:
        info = sf.info(path)
    except (OSError, sf.LibsndfileError) as exc:
        raise AudioError(f"Cannot read decoded audio: {exc}") from exc
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise AudioError("Decoded audio must be mono 16 kHz.")
    if info.frames == 0:
        raise AudioError("The input contains no audio samples.")
    return AudioInfo(info.frames)


def silence_boundary(samples: "np.ndarray") -> int:
    import numpy as np

    frame_size = SAMPLE_RATE // 50
    lookback = min(5 * SAMPLE_RATE, len(samples) // 4)
    start = len(samples) - lookback
    frame_count = lookback // frame_size
    if frame_count < 2:
        return len(samples)
    frames = samples[start : start + frame_count * frame_size].reshape(-1, frame_size)
    energy = np.mean(frames * frames, axis=1)
    threshold = min(0.01, max(0.0001, float(np.sqrt(np.mean(samples * samples))) * 0.1))
    quiet = energy <= threshold * threshold
    edges = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 8]
    if runs:
        a, b = runs[-1]
        return start + ((a + b) * frame_size // 2)
    return len(samples)


def pause_boundary(samples: "np.ndarray") -> int:
    import numpy as np

    frame_size = SAMPLE_RATE // 50
    count = len(samples) // frame_size
    if len(samples) < 7 * SAMPLE_RATE:
        return len(samples)
    frames = samples[: count * frame_size].reshape(-1, frame_size)
    threshold = min(0.01, max(0.0001, float(np.sqrt(np.mean(samples * samples))) * 0.1))
    quiet = np.mean(frames * frames, axis=1) <= threshold * threshold
    edges = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        middle = (int(first) + int(last)) * frame_size // 2
        if (
            last - first >= 20
            and middle >= 5 * SAMPLE_RATE
            and last * frame_size <= len(samples) - SAMPLE_RATE
            and first > 0
        ):
            return middle
    return len(samples)


def iter_chunks(path: Path, chunk_seconds: float = 30.0):
    import soundfile as sf

    if not math.isfinite(chunk_seconds) or not 1 <= chunk_seconds <= 60:
        raise ValueError("chunk_seconds must be between 1 and 60.")
    audio_info(path)
    max_frames = round(chunk_seconds * SAMPLE_RATE)
    with sf.SoundFile(path) as audio:
        position = 0
        while position < len(audio):
            audio.seek(position)
            samples = audio.read(max_frames, dtype="float32", always_2d=False)
            if not len(samples):
                raise AudioError("Decoded audio ended unexpectedly.")
            length = pause_boundary(samples)
            if length == len(samples) and position + length < len(audio):
                length = silence_boundary(samples)
            yield AudioChunk(position, samples[:length])
            position += length


def is_silent(samples: "np.ndarray") -> bool:
    import numpy as np

    return not len(samples) or float(np.max(np.abs(samples))) <= 1 / 32768
