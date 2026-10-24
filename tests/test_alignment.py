import math

import pytest

from qwen_asr_cli.alignment import Segment, Word, aligned_segments, format_segment, format_time
from qwen_asr_cli.errors import TranscriptionError


def test_segments_preserve_punctuation_words_and_offsets():
    text = "Привет, мир! This is a test."
    words = [
        Word("Привет", 0.2, 0.8), Word("мир", 0.8, 1.4),
        Word("This", 2.1, 2.3), Word("is", 2.3, 2.5),
        Word("a", 2.5, 2.6), Word("test", 2.6, 3.5),
    ]
    segments = aligned_segments(text, words, 4, 63.5)
    assert " ".join(segment.text for segment in segments) == text
    assert segments == [Segment("Привет, мир!", 63.7, 64.9), Segment("This is a test.", 65.6, 67.0)]
    assert format_segment(segments[0]) == "[00:01:03.700 --> 00:01:04.900] Привет, мир!\n"


def test_unpunctuated_long_text_is_segmented_without_loss():
    text = " ".join(f"word{i}" for i in range(60))
    words = [Word(f"word{i}", i * 0.4, (i + 1) * 0.4) for i in range(60)]
    segments = aligned_segments(text, words, 24, 100)
    assert len(segments) == 3
    assert " ".join(segment.text for segment in segments) == text
    assert all(a.end <= b.start for a, b in zip(segments, segments[1:]))
    assert segments[-1].end == 124


@pytest.mark.parametrize("text,words", [
    ("Don't drop punctuation—please!", ["Don't", "drop", "punctuationplease"]),
    ("«Привет», по-русски.", ["Привет", "порусски"]),
    ("你好世界。", ["你", "好", "世", "界"]),
])
def test_alignment_token_cleanup_does_not_rewrite_original(text, words):
    aligned = [Word(word, i * 0.2, (i + 1) * 0.2) for i, word in enumerate(words)]
    segments = aligned_segments(text, aligned, 5, 0)
    assert "".join(segment.text for segment in segments) == text


@pytest.mark.parametrize("words", [
    [], [Word("wrong", 0, 1)], [Word("hello", 0, 1), Word("extra", 1, 2)],
    [Word("hello", math.nan, 1)], [Word("hello", 0, math.inf)],
    [Word("hello", -0.1, 1)], [Word("hello", 1, 0.5)],
    [Word("hello", 0, 2.2)], [Word("hello", 0.5, 0.5)],
])
def test_bad_alignment_fails_instead_of_dropping_text(words):
    with pytest.raises(TranscriptionError):
        aligned_segments("hello", words, 2, 0)


def test_out_of_order_words_are_rejected():
    with pytest.raises(TranscriptionError, match="Invalid alignment time"):
        aligned_segments("hello world", [Word("hello", 0, 1), Word("world", 0.5, 1.5)], 2, 0)


def test_one_quantization_tick_at_end_is_bounded():
    result = aligned_segments("hello", [Word("hello", 0.1, 1.04)], 1, 3)
    assert result == [Segment("hello", 3.1, 4)]


def test_empty_alignment():
    assert aligned_segments("", [], 1, 0) == []
    with pytest.raises(TranscriptionError):
        aligned_segments("", [Word("invented", 0, 1)], 1, 0)


def test_time_format_carry_and_long_hours():
    assert format_time(59.9996) == "00:01:00.000"
    assert format_time(100 * 3600 + 61.08) == "100:01:01.080"


@pytest.mark.parametrize("first,second", [
    ("Hello.", '"Goodbye!"'),
    ("Hello.", "(Goodbye!)"),
    ("Hello.", "[Goodbye!]"),
    ('"Hello."', '"Goodbye!"'),
    ("Hello.", "«Goodbye!»"),
    ("Hello.", "“Goodbye!”"),
    ("Hello.", '("Goodbye!")'),
])
def test_opening_punctuation_stays_with_next_segment(first, second):
    text = f"{first} {second}"
    segments = aligned_segments(
        text, [Word("Hello", 0, 1.2), Word("Goodbye", 2, 3)], 4, 0,
    )
    assert [segment.text for segment in segments] == [first, second]
    assert " ".join(segment.text for segment in segments) == text
