from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intelli_speech.alignment import (
    AlignmentError,
    Emissions,
    accepted_vowels,
    align_transcript,
    align_words,
    clean_transcript,
    ctc_align,
    romanize_word,
    vowel_category_counts,
    write_textgrid,
)

VOCAB = {c: i + 1 for i, c in enumerate("abcdeghijklmnoprstuvy")}   # 0 = blank


def _units(word: str) -> list[tuple[str, str, str | None]]:
    return [(u.kind, u.letters, u.category) for u in romanize_word(word).units]


# --------------------------------------------------------------------------- romanisation


def test_romanisation_keeps_short_and_long_vowels_distinct() -> None:
    assert romanize_word("కాలు").letters == "kaalu"
    assert [c for _, _, c in _units("కాలు") if c] == ["aa", "u"]
    assert romanize_word("కోతి").letters == "kooti"
    assert [c for _, _, c in _units("కోతి") if c] == ["oo", "i"]
    assert romanize_word("ఈగ").letters == "iiga"                    # independent long vowel + inherent a
    assert [c for _, _, c in _units("ఊరు") if c] == ["uu", "u"]
    assert [c for _, _, c in _units("ఏడు") if c] == ["ee", "u"]


def test_romanisation_virama_anusvara_and_non_category_vowels() -> None:
    assert romanize_word("అమ్మ").letters == "amma"                  # virama suppresses inherent a
    assert [c for _, _, c in _units("అమ్మ") if c] == ["a", "a"]
    assert romanize_word("సంత").letters == "samta"                  # anusvara → m
    cats = [c for k, _, c in _units("కై") if k == "vowel"]
    assert cats == [None]                                           # diphthong: no corner category
    assert romanize_word("ఖ").letters == "kha"                      # aspirate digraph
    assert romanize_word("‌").letters == ""                    # ZWNJ ignored


def test_clean_transcript_pre_registered_exclusions() -> None:
    words, why = clean_transcript("అది <noise> ఒక చెట్టు.")
    assert why is None and words == ["అది", "ఒక", "చెట్టు"]
    assert clean_transcript("{చెట్టు} అది")[1] == "bracketed_telugu"
    assert clean_transcript("అది 2 చెట్లు")[1] == "digits"
    assert clean_transcript("అది ౨ చెట్లు")[1] == "digits"         # Telugu digit
    assert clean_transcript("this is a tree అది")[1] == "latin_share"
    assert clean_transcript("")[1] == "empty_transcript"
    assert clean_transcript(None)[1] == "empty_transcript"
    assert clean_transcript("<laugh> [noise]")[1] == "no_letters"


def test_vowel_category_counts_text_only() -> None:
    counts = vowel_category_counts("కాలు కోతి")
    assert counts["aa"] == 1 and counts["u"] == 1 and counts["oo"] == 1 and counts["i"] == 1
    assert sum(vowel_category_counts("2 cats").values()) == 0       # excluded text → zero counts


# --------------------------------------------------------------------------- CTC alignment


def _emissions(path: list[int], vocab_size: int = 22, sharp: float = 6.0) -> np.ndarray:
    """Log-posteriors whose argmax follows `path` (one label per frame)."""
    lp = np.full((len(path), vocab_size), -sharp)
    for t, k in enumerate(path):
        lp[t, k] = 0.0
    lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
    return lp


def test_ctc_align_recovers_token_spans_including_repeats() -> None:
    a, m = VOCAB["a"], VOCAB["m"]
    path = [0, a, a, 0, m, m, 0, m, a, a, a, 0]                   # "amma": blank needed between m m
    spans = ctc_align(_emissions(path), [a, m, m, a], blank=0)
    assert [(s.start, s.end) for s in spans] == [(1, 3), (4, 6), (7, 8), (8, 11)]
    assert all(s.score > 0.9 for s in spans)


def test_ctc_align_failure_modes() -> None:
    a, m = VOCAB["a"], VOCAB["m"]
    with pytest.raises(AlignmentError) as e:
        ctc_align(_emissions([0, m, 0]), [m, m, a], blank=0)      # needs ≥ 4 frames (repeat)
    assert e.value.reason == "audio_too_short"
    with pytest.raises(AlignmentError) as e:
        ctc_align(_emissions([0, 0]), [], blank=0)
    assert e.value.reason == "empty_targets"
    with pytest.raises(AlignmentError) as e:
        ctc_align(_emissions([0, a, 0]), [a, 0], blank=0)
    assert e.value.reason == "blank_in_targets"
    with pytest.raises(AlignmentError):
        ctc_align(np.zeros(5), [a], blank=0)


def _fake_backend(word: str, frames_per_letter: int = 3, conf: float = 6.0):
    letters = romanize_word(word).letters
    path = [0]
    for ch in letters:
        path += [VOCAB[ch]] * frames_per_letter + [0]
    def run(audio):
        return Emissions(_emissions(path, sharp=conf), VOCAB, 0, 0.02)
    return run


def test_align_transcript_returns_vowel_segments_with_categories() -> None:
    res = align_transcript(_fake_backend("కాలు"), np.zeros(16000), "కాలు")
    assert res["status"] == "aligned" and res["utterance_conf"] > 0.9
    cats = [v["category"] for v in res["vowels"]]
    assert cats == ["aa", "u"]
    aa = res["vowels"][0]
    assert aa["start_s"] == pytest.approx(5 * 0.02) and aa["end_s"] == pytest.approx(12 * 0.02)   # "a","a" letters merged: frames 5–11, end exclusive
    assert len(accepted_vowels(res)) == 2


def test_align_transcript_failures_are_returned_not_raised() -> None:
    assert align_transcript(_fake_backend("కాలు"), np.zeros(10), "అది 5")["reason"] == "digits"
    def boom(audio):
        raise RuntimeError("cuda oom")
    res = align_transcript(boom, np.zeros(10), "కాలు")
    assert res["status"] == "failed" and res["reason"].startswith("backend_error")
    def short(audio):
        return Emissions(_emissions([0, 1]), VOCAB, 0, 0.02)
    assert align_transcript(short, np.zeros(10), "కాలు")["reason"] == "audio_too_short"
    def no_k(audio):
        return Emissions(_emissions([0] * 40), {c: i for c, i in VOCAB.items() if c != "k"}, 0, 0.02)
    assert align_transcript(no_k, np.zeros(10), "కాలు")["reason"] == "unknown_token"


def test_low_confidence_utterances_yield_no_accepted_vowels() -> None:
    res = align_transcript(_fake_backend("కాలు", conf=0.05), np.zeros(10), "కాలు")
    assert res["status"] == "low_confidence"
    assert accepted_vowels(res) == []


def test_short_or_low_confidence_vowel_tokens_are_dropped() -> None:
    res = {"status": "aligned", "vowels": [
        {"category": "a", "start_s": 0.0, "end_s": 0.02, "conf": 0.9},     # 20 ms < 40 ms
        {"category": "i", "start_s": 0.1, "end_s": 0.2, "conf": 0.1},      # conf < 0.20
        {"category": "u", "start_s": 0.3, "end_s": 0.4, "conf": 0.8}]}
    assert [v["category"] for v in accepted_vowels(res)] == ["u"]


def test_align_words_word_spans() -> None:
    words = [romanize_word("కాలు"), romanize_word("ఈగ")]
    path = [0] + [VOCAB[c] for c in "kaalu"] + [0] + [VOCAB[c] for c in "iiga"] + [0]
    path = [0, VOCAB["k"], VOCAB["a"], 0, VOCAB["a"], VOCAB["l"], VOCAB["u"], 0, VOCAB["i"], 0, VOCAB["i"], VOCAB["g"], VOCAB["a"], 0]
    res = align_words(Emissions(_emissions(path), VOCAB, 0, 0.02), words)
    assert len(res["words"]) == 2 and res["words"][0]["end_s"] <= res["words"][1]["start_s"]
    assert [v["category"] for v in res["vowels"]] == ["aa", "u", "ii", "a"]


def test_textgrid_writer(tmp_path: Path) -> None:
    path = tmp_path / "x.TextGrid"
    write_textgrid(path, 1.0, {"vowels": [(0.2, 0.3, "aa"), (0.5, 0.6, "u")], "words": []})
    text = path.read_text(encoding="utf-8")
    assert 'class = "IntervalTier"' in text and text.count("intervals [") == 5 + 1
    assert 'text = "aa"' in text and "size = 2" in text


def test_ctc_align_long_transcript_does_not_overflow() -> None:
    """Regression: > 63 targets (> 127 CTC states) overflowed an int8 back-pointer index."""
    letters = "abcdeghijklmnoprstuvy"
    targets = [VOCAB[letters[i % len(letters)]] for i in range(150)]
    path = [0]
    for t in targets:
        path += [t, t, 0]
    spans = ctc_align(_emissions(path), targets, blank=0)
    assert len(spans) == 150 and spans[-1].end <= len(path)
    assert all(a.end <= b.start for a, b in zip(spans[:-1], spans[1:]))
    assert [(s.start, s.end) for s in spans[:2]] == [(1, 3), (4, 6)]
