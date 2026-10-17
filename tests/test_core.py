import math

import pytest

from transcriber.core import (
    Segment,
    SpeakerTurn,
    TranscriptionError,
    Word,
    assign_speakers,
    timestamp,
    transcript_text,
)


def test_switch_speakers_within_one_whisper_segment():
    words = [Word(0, 0.5, " Hallo"), Word(0.6, 1, " Welt."), Word(1.1, 1.5, " Hello.")]
    turns = [SpeakerTurn(1.05, 2, "B"), SpeakerTurn(0, 1.05, "A")]
    assert assign_speakers(words, turns) == (
        Segment(0, 1, "Hallo Welt.", "Sprecher 1"),
        Segment(1.1, 1.5, "Hello.", "Sprecher 2"),
    )


def test_unknown_is_not_assigned_to_nearest_speaker():
    assert assign_speakers([Word(3, 4, "Hallo")], [SpeakerTurn(0, 1, "A")]) == (
        Segment(3, 4, "Hallo", "Sprecher unbekannt"),
    )


def test_total_overlap_wins_not_first_matching_turn():
    turns = [SpeakerTurn(0, 0.1, "A"), SpeakerTurn(0.1, 0.8, "B")]
    assert assign_speakers([Word(0, 1, "Hello")], turns)[0].speaker == "Sprecher 2"


def test_without_diarization_no_labels_and_readable_punctuation():
    words = [Word(0, 0.2, "Hello"), Word(0.2, 0.3, ","), Word(0.4, 0.8, " world!")]
    assert assign_speakers(words, None) == (Segment(0, 0.8, "Hello, world!"),)


def test_pause_and_long_passages_are_separate_segments():
    words = [Word(0, 1, "Hi."), Word(5, 6, "Again."), Word(29, 31, "Bye.")]
    assert len(assign_speakers(words, None)) == 3


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2), (math.nan, 1), (0, math.inf)])
def test_invalid_word_timestamps_fail_explicitly(start, end):
    with pytest.raises(TranscriptionError, match="Zeitstempel"):
        Word(start, end, "Hi")


def test_empty_result_and_timestamp_rollover():
    assert transcript_text(assign_speakers([], None)) == ""
    assert timestamp(3599.9996) == "01:00:00.000"


def test_txt_contains_exact_labels_timestamps_and_utf8_text():
    result = transcript_text([Segment(0.25, 2, "Grüße!", "Sprecher 1")])
    assert result == "[00:00:00.250 – 00:00:02.000] Sprecher 1: Grüße!\n"
    assert result.encode("utf-8").decode("utf-8") == result
