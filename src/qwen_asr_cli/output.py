import os
import shutil
import tempfile
from pathlib import Path

from .errors import InputError


def validate_output(source: Path, destination: Path, force: bool) -> None:
    if destination.resolve() == source.resolve() or (
        destination.exists() and destination.samefile(source)
    ):
        raise InputError(
            "The output is the source media (possibly a link). The source can never be overwritten."
        )
    if destination.is_symlink():
        raise InputError(
            "Refusing to replace a symbolic-link output; choose a regular output path."
        )
    if destination.exists():
        if not destination.is_file():
            raise InputError(f"Output is not a regular file: {destination}")
        if not force:
            raise InputError(f"Output already exists: {destination}. Use --force to replace it.")
    if not destination.parent.is_dir():
        raise InputError(f"Output directory does not exist: {destination.parent}")


def publish_output(staged: Path, source: Path, destination: Path, force: bool) -> None:
    validate_output(source, destination, force)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            with staged.open("rb") as transcript:
                shutil.copyfileobj(transcript, output)
            output.flush()
            os.fsync(output.fileno())
        validate_output(source, destination, force)
        if force:
            os.replace(temporary, destination)
        elif os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
