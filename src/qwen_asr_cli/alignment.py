import math
import re
import unicodedata
from dataclasses import dataclass

from .errors import TranscriptionError


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float


def kept_character(char: str) -> bool:
    return char == "'" or unicodedata.category(char).startswith(("L", "N"))


def aligned_segments(text: str, words: list[Word], duration: float, offset: float) -> list[Segment]:
    if not text.strip():
        if words:
            raise TranscriptionError("Alignment returned words for an empty transcript.")
        return []
    if not words:
        raise TranscriptionError(
            "Alignment returned no words; no transcript was saved. Retry without --timestamps."
        )
    positions = [i for i, char in enumerate(text) if kept_character(char)]
    canonical = "".join(text[i] for i in positions)
    aligned = ["".join(c for c in word.text if kept_character(c)) for word in words]
    if not all(aligned) or "".join(aligned) != canonical:
        raise TranscriptionError(
            "Alignment did not preserve all transcript words. Retry without --timestamps."
        )
    bounded = []
    previous_end = 0.0
    for word in words:
        start, end = word.start, word.end
        if (
            not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end < start or start < previous_end - 1e-6
            or end > duration + 0.08 + 1e-6
        ):
            raise TranscriptionError(
                f"Invalid alignment time for {word.text!r}: {start:.3f}–{end:.3f}s "
                f"in {duration:.3f}s audio. Retry without --timestamps."
            )
        bounded.append(Word(word.text, min(start, duration), min(end, duration)))
        previous_end = end
    ends = []
    consumed = 0
    for cleaned in aligned:
        consumed += len(cleaned)
        boundary = positions[consumed] if consumed < len(positions) else len(text)
        if consumed < len(positions):
            gap = text[positions[consumed - 1] + 1 : boundary]
            opening = re.search(r'[([{«“‘"]+$', gap)
            if opening:
                boundary -= len(opening[0])
        ends.append(boundary)
    segments = []
    first_word = 0
    text_start = 0
    last_text_start = 0
    for i, word in enumerate(bounded):
        span = text[text_start : ends[i]].strip()
        elapsed = word.end - bounded[first_word].start
        last = i == len(bounded) - 1
        sentence_end = elapsed >= 1 and re.search(r"[.!?…][\"'»”)]*$", span)
        pause = not last and bounded[i + 1].start - word.end >= 0.7
        if last or sentence_end or pause or elapsed >= 8 or i - first_word >= 19:
            if elapsed <= 0:
                if not last:
                    continue
                if not segments:
                    raise TranscriptionError(
                        "Alignment produced no positive-duration segment. "
                        "Retry without --timestamps."
                    )
                previous = segments[-1]
                segments[-1] = Segment(
                    text[last_text_start : ends[i]].strip(), previous.start,
                    max(previous.end, offset + word.end),
                )
                continue
            segments.append(Segment(span, offset + bounded[first_word].start, offset + word.end))
            last_text_start = text_start
            first_word = i + 1
            text_start = ends[i]
    return segments


def format_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_segment(segment: Segment) -> str:
    return f"[{format_time(segment.start)} --> {format_time(segment.end)}] {segment.text}\n"
