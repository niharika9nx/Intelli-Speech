from __future__ import annotations

import pytest

from intelli_speech.telugu_text import (
    akshara_count,
    character_error_rate,
    is_telugu,
    levenshtein,
    normalize_text,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        ("అ", 1),                 # independent vowel
        ("క", 1),                 # consonant with inherent vowel
        ("కా", 1),                # consonant + vowel sign
        ("తెలుగు", 3),            # te-lu-gu
        ("అమ్మ", 2),              # a-mma (conjunct via virama)
        ("స్త్రీ", 1),            # three-consonant conjunct + vowel sign
        ("రంగు", 2),              # anusvara does not add a syllable
        ("ఫోన్", 1),              # word-final dead consonant closes the syllable
        ("తెలుగు భాష", 5),        # two words
        ("hello 123", 0),         # non-Telugu ignored
    ],
)
def test_akshara_count(text: str, expected: int) -> None:
    assert akshara_count(text) == expected


def test_normalize_text_drops_joiners_and_whitespace() -> None:
    assert normalize_text("  తెలు‌గు \n భాష ") == "తెలుగు భాష"


def test_is_telugu() -> None:
    assert is_telugu("abc తెలుగు")
    assert not is_telugu("abc")


def test_levenshtein_and_cer() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert character_error_rate("తెలుగు", "తెలుగు") == 0.0
    assert character_error_rate("", "anything") is None
    assert character_error_rate("abcd", "abed") == pytest.approx(0.25)
