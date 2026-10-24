import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / ".local/tmp/pytest")


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    environment = os.environ.copy()
    old_temp = tempfile.tempdir
    monkeypatch.setenv("QWEN_ASR_HOME", str(tmp_path))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment)
        tempfile.tempdir = old_temp


@pytest.fixture
def write_audio():
    def write(path, seconds=1, samplerate=16000, silent=False, channels=1):
        samples = np.zeros(round(seconds * samplerate), dtype=np.float32)
        if not silent:
            phase = np.arange(len(samples)) * 2 * np.pi * 220 / samplerate
            samples = (0.2 * np.sin(phase)).astype(np.float32)
        if channels > 1:
            samples = np.tile(samples[:, None], (1, channels))
        sf.write(path, samples, samplerate, subtype="PCM_16")
        return path

    return write
