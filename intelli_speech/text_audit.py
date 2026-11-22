"""Script-composition statistics for transcription fields (no text is ever returned)."""

from __future__ import annotations

import unicodedata
from typing import Any

from .telugu_text import akshara_count, normalize_text

TELUGU_START, TELUGU_END = 0x0C00, 0x0C7F


def char_class(ch: str) -> str:
    code = ord(ch)
    if TELUGU_START <= code <= TELUGU_END:
        return "telugu"
    category = unicodedata.category(ch)
    if category.startswith("L"):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        return "latin" if "LATIN" in name else "other_letter"
    if category.startswith("N"):
        return "digit"
    if ch.isspace():
        return "space"
    return "symbol"  # punctuation, brackets, marks outside Telugu, etc.


def script_profile(text: Any) -> dict[str, Any]:
    """Proportions over non-space characters, plus lengths and akshara count."""
    value = "" if text is None else normalize_text(str(text))
    if value.lower() in ("", "nan", "none"):
        return {"missing": True}
    counts = {"telugu": 0, "latin": 0, "other_letter": 0, "digit": 0, "symbol": 0}
    symbols: dict[str, int] = {}
    for ch in value:
        cls = char_class(ch)
        if cls == "space":
            continue
        counts[cls] += 1
        if cls == "symbol":
            symbols[ch] = symbols.get(ch, 0) + 1
    total = sum(counts.values())
    profile: dict[str, Any] = {
        "missing": False,
        "chars_non_space": total,
        "words": len(value.split()),
        "aksharas": akshara_count(value),
    }
    for key, count in counts.items():
        profile[f"prop_{key}"] = round(count / total, 4) if total else 0.0
    dominant = max(counts, key=counts.get) if total else "none"
    profile["dominant_class"] = dominant
    # Only symbol characters (never letters) are exposed, for annotation-marker discovery.
    profile["symbols"] = symbols
    return profile
