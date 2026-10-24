import os
from pathlib import Path

import pytest

from qwen_asr_cli import output
from qwen_asr_cli.errors import InputError


@pytest.fixture
def paths(tmp_path):
    source, staged, destination = [
        tmp_path / name for name in ("source.wav", "staged.txt", "output.txt")
    ]
    source.write_bytes(b"original media")
    staged.write_text("Привет, world!\n", encoding="utf-8")
    return source, staged, destination


def test_atomic_publication_and_explicit_overwrite(paths):
    source, staged, destination = paths
    output.publish_output(staged, source, destination, False)
    assert destination.read_bytes() == staged.read_bytes()
    staged.write_text("replacement", encoding="utf-8")
    with pytest.raises(InputError, match="already exists"):
        output.publish_output(staged, source, destination, False)
    assert destination.read_text(encoding="utf-8") == "Привет, world!\n"
    output.publish_output(staged, source, destination, True)
    assert destination.read_text(encoding="utf-8") == "replacement"
    assert not list(destination.parent.glob(".*.tmp"))


@pytest.mark.parametrize("force", [False, True])
def test_source_and_hardlinks_never_overwritten(paths, force):
    source, staged, destination = paths
    with pytest.raises(InputError, match="source media"):
        output.publish_output(staged, source, source, force)
    os.link(source, destination)
    with pytest.raises(InputError, match="source media"):
        output.publish_output(staged, source, destination, force)
    assert source.read_bytes() == b"original media"


def test_symbolic_link_source_is_protected(paths):
    source, staged, destination = paths
    try:
        destination.symlink_to(source)
    except OSError:
        pytest.skip("Windows symbolic links require Developer Mode or privilege")
    with pytest.raises(InputError, match="source media"):
        output.publish_output(staged, source, destination, True)
    assert source.read_bytes() == b"original media"


def test_non_source_symbolic_link_is_not_followed(paths):
    source, staged, destination = paths
    try:
        destination.symlink_to(staged)
    except OSError:
        pytest.skip("Windows symbolic links require Developer Mode or privilege")
    with pytest.raises(InputError, match="symbolic-link"):
        output.publish_output(staged, source, destination, True)


def test_cancellation_preserves_old_output_and_cleans_sibling(paths, monkeypatch):
    source, staged, destination = paths
    destination.write_text("old", encoding="utf-8")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        output.publish_output(staged, source, destination, True)
    assert destination.read_text() == "old"
    assert not list(destination.parent.glob(".*.tmp"))


def test_destination_is_rechecked_after_staging(paths, monkeypatch):
    source, staged, destination = paths
    real_validate = output.validate_output
    calls = 0

    def validate(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.write_text("another process", encoding="utf-8")
        real_validate(*args)

    monkeypatch.setattr(output, "validate_output", validate)
    with pytest.raises(InputError, match="already exists"):
        output.publish_output(staged, source, destination, False)
    assert destination.read_text() == "another process"
    assert not list(destination.parent.glob(".*.tmp"))


def test_no_clobber_even_after_final_check(paths, monkeypatch):
    source, staged, destination = paths
    method = "rename" if os.name == "nt" else "link"
    real_publish = getattr(os, method)

    def concurrent_writer(temp, target):
        Path(target).write_text("concurrent", encoding="utf-8")
        real_publish(temp, target)

    monkeypatch.setattr(os, method, concurrent_writer)
    with pytest.raises(FileExistsError):
        output.publish_output(staged, source, destination, False)
    assert destination.read_text() == "concurrent"
    assert not list(destination.parent.glob(".*.tmp"))


def test_missing_output_directory(paths):
    source, staged, destination = paths
    with pytest.raises(InputError, match="directory does not exist"):
        output.publish_output(staged, source, destination / "missing" / "out.txt", False)
