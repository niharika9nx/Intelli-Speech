"""Telugu script helpers: akshara (orthographic syllable) counting and text normalisation.

Telugu is an abugida. An akshara is an independent vowel, or a consonant cluster
(consonants joined by virama) with its inherent or dependent vowel. Counting aksharas
gives an alignment-free syllable estimate for speaking-rate measures.

Rules implemented (Unicode Telugu block U+0C00–U+0C7F):
- Independent vowels each start one akshara.
- A consonant starts one akshara unless the previous character was a virama
  (then it continues the conjunct cluster).
- A consonant followed by a virama at the end of a word adds no akshara of its own
  when it ends the word (rare; mostly loanwords) — it closes the previous syllable.
- Dependent vowel signs, anusvara, visarga, candrabindu, nukta and ZWJ/ZWNJ never
  start an akshara.
This is an orthographic estimate, not a phonetic syllabification.
"""

from __future__ import annotations

import re
import unicodedata

VIRAMA = "్"
_INDEPENDENT_VOWELS = set(chr(c) for c in range(0x0C05, 0x0C15)) | {"ౠ", "ౡ"}
_CONSONANTS = set(chr(c) for c in range(0x0C15, 0x0C3A)) | {"ౘ", "ౙ", "ౚ"}
_JOINERS = {"‌", "‍"}
_TELUGU_RANGE = re.compile(r"[ఀ-౿]")


def is_telugu(text: str) -> bool:
    """True when the text contains at least one Telugu-script character."""
    return bool(_TELUGU_RANGE.search(text or ""))


def normalize_text(text: str) -> str:
    """NFC-normalise, drop joiners and collapse whitespace (for CER comparisons)."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = "".join(ch for ch in value if ch not in _JOINERS)
    return " ".join(value.split())


def akshara_count(text: str) -> int:
    """Count Telugu aksharas in ``text``; non-Telugu characters are ignored."""
    value = normalize_text(text)
    count = 0
    previous = ""
    for index, ch in enumerate(value):
        if ch in _INDEPENDENT_VOWELS:
            count += 1
        elif ch in _CONSONANTS:
            if previous == VIRAMA:
                pass  # continues a conjunct cluster
            else:
                following = value[index + 1] if index + 1 < len(value) else ""
                after = value[index + 2] if index + 2 < len(value) else " "
                word_final_dead = following == VIRAMA and (
                    after.isspace() or not is_telugu(after)
                )
                if not (word_final_dead and count > 0 and previous not in ("", " ")):
                    count += 1
        previous = ch
    return count


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """CER of ``hypothesis`` against ``reference`` after normalisation (None if no reference)."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)
