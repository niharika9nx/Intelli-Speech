"""MMS-style CTC forced alignment for Telugu transcripts (Experiment 2, Stage A).

* `romanize_telugu` — deterministic akshara-aware Telugu→Latin romanisation into the MMS
  forced-alignment alphabet (a–z), keeping a record of which output letters form each
  vowel unit and its category (short/long kept distinct: a/aa, i/ii, u/uu, e/ee, o/oo).
* `clean_transcript` — pre-registered text exclusions (EXP2_PLAN §13.3).
* `ctc_align` — model-agnostic Viterbi CTC forced alignment in numpy (no dependency on
  torchaudio.functional.forced_align).
* Emission backends (Colab): torchaudio `MMS_FA` (primary) or a Hugging Face CTC model
  (fallback, e.g. the converted MMS-300M aligner; CC-BY-NC-4.0, research use only).
* `vowel_segments`, `write_textgrid` — outputs for labelled measures and manual checking.

Transcripts are research data: callers must keep them on Drive and never print them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# §13.3 pre-registered thresholds
UTTERANCE_MIN_CONF = 0.40
TOKEN_MIN_CONF = 0.20
VOWEL_MIN_S = 0.040
LATIN_MAX_SHARE = 0.30

VOWEL_CATEGORIES = ("a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo")
CORNER = {"i": ("i", "ii"), "a": ("a", "aa"), "u": ("u", "uu")}
CORNER_LONG = {"i": ("ii",), "a": ("aa",), "u": ("uu",)}

_INDEPENDENT = {"అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu", "ఋ": "ru", "ౠ": "ruu",
                "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au"}
_MATRA = {"ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "ru", "ౄ": "ruu",
          "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au"}
_CONSONANT = {"క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "n", "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh",
              "ఞ": "n", "ట": "t", "ఠ": "th", "డ": "d", "ఢ": "dh", "ణ": "n", "త": "t", "థ": "th", "ద": "d",
              "ధ": "dh", "న": "n", "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m", "య": "y", "ర": "r",
              "ఱ": "r", "ల": "l", "ళ": "l", "ఴ": "l", "వ": "v", "శ": "sh", "ష": "sh", "స": "s", "హ": "h",
              "ౘ": "ts", "ౙ": "dz"}
_VIRAMA = "్"
_ANUSVARA, _VISARGA, _CANDRABINDU = "ం", "ః", "ఁ"
_IGNORABLE = {"‌", "‍"}   # ZWNJ / ZWJ
_TAG = re.compile(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]")
TELUGU_RE = re.compile(r"[ఀ-౿]")


class AlignmentError(RuntimeError):
    """Raised for a pre-registered alignment failure; `reason` is a short code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Unit:
    kind: str                 # "vowel" | "consonant" | "latin"
    letters: str
    category: str | None = None


@dataclass
class Word:
    source: str
    units: list[Unit] = field(default_factory=list)

    @property
    def letters(self) -> str:
        return "".join(u.letters for u in self.units)


# --------------------------------------------------------------------------- text


def clean_transcript(text: str) -> tuple[list[str] | None, str | None]:
    """Return (words, None) or (None, exclusion_reason) per EXP2_PLAN §13.3."""
    if text is None or not str(text).strip():
        return None, "empty_transcript"
    text = str(text)
    for span in _TAG.findall(text):
        if TELUGU_RE.search(span):
            return None, "bracketed_telugu"
    text = _TAG.sub(" ", text)
    if re.search(r"[0-9౦-౯]", text):
        return None, "digits"
    letters = [c for c in text if c.isalpha() or TELUGU_RE.match(c)]
    if not letters:
        return None, "no_letters"
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
    if latin / len(letters) > LATIN_MAX_SHARE:
        return None, "latin_share"
    words = [w for w in re.split(r"[^\wఀ-౿]+", text) if w and any(c.isalpha() or TELUGU_RE.match(c) for c in w)]
    return (words, None) if words else (None, "no_words")


def romanize_word(word: str) -> Word:
    """Akshara-aware romanisation with vowel units (inherent 'a' made explicit)."""
    out = Word(word)
    chars = [c for c in word if c not in _IGNORABLE]
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in _CONSONANT:
            out.units.append(Unit("consonant", _CONSONANT[c]))
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _MATRA:
                v = _MATRA[nxt]
                out.units.append(Unit("vowel", v, v if v in VOWEL_CATEGORIES else None))
                i += 2
                continue
            out.units.append(Unit("vowel", "a", "a"))
            i += 1
        elif c in _INDEPENDENT:
            v = _INDEPENDENT[c]
            out.units.append(Unit("vowel", v, v if v in VOWEL_CATEGORIES else None))
            i += 1
        elif c == _ANUSVARA:
            out.units.append(Unit("consonant", "m"))
            i += 1
        elif c == _VISARGA:
            out.units.append(Unit("consonant", "h"))
            i += 1
        elif c in (_CANDRABINDU, _VIRAMA) or c in _MATRA:
            i += 1                     # stray marks carry no sound of their own here
        elif c.isascii() and c.isalpha():
            out.units.append(Unit("latin", c.lower()))
            i += 1
        else:
            i += 1
    return out


def romanize_telugu(words: Sequence[str]) -> list[Word]:
    return [w for w in (romanize_word(x) for x in words) if w.letters]


def vowel_category_counts(text: str) -> dict[str, int]:
    """Text-only vowel-category counts (lexical comparability, §13.4)."""
    words, reason = clean_transcript(text)
    counts = {c: 0 for c in VOWEL_CATEGORIES}
    if words is None:
        return counts
    for w in romanize_telugu(words):
        for u in w.units:
            if u.category in counts:
                counts[u.category] += 1
    return counts


# --------------------------------------------------------------------------- CTC alignment


@dataclass
class TokenSpan:
    token: int
    start: int   # frame index (inclusive)
    end: int     # frame index (exclusive)
    score: float


def ctc_align(log_probs: np.ndarray, targets: Sequence[int], blank: int = 0) -> list[TokenSpan]:
    """Viterbi CTC forced alignment. `log_probs`: (T, V) log-posteriors."""
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise AlignmentError("bad_emissions", "expected (T, V)")
    targets = list(targets)
    if not targets:
        raise AlignmentError("empty_targets")
    if any(t == blank for t in targets):
        raise AlignmentError("blank_in_targets")
    T = lp.shape[0]
    repeats = sum(1 for a, b in zip(targets[:-1], targets[1:]) if a == b)
    if T < len(targets) + repeats:
        raise AlignmentError("audio_too_short", f"T={T} < needed {len(targets) + repeats}")
    ext = [blank]
    for t in targets:
        ext += [t, blank]
    ext = np.array(ext)
    S = ext.size
    skip_ok = np.zeros(S, bool)
    for s in range(2, S):
        skip_ok[s] = ext[s] != blank and ext[s] != ext[s - 2]
    neg = -1e30
    dp = np.full(S, neg)
    dp[0] = lp[0, ext[0]]
    dp[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate([[neg], dp[:-1]])
        prev2 = np.where(skip_ok, np.concatenate([[neg, neg], dp[:-2]]), neg)
        cand = np.vstack([stay, prev1, prev2])
        choice = np.argmax(cand, axis=0)
        dp = cand[choice, np.arange(S)] + lp[t, ext]
        back[t] = choice
    end = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[end] <= neg / 2:
        raise AlignmentError("no_path")
    states = np.empty(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= back[t, s]
    spans = []
    for i, tok in enumerate(targets):
        frames = np.flatnonzero(states == 2 * i + 1)
        if frames.size == 0:
            raise AlignmentError("token_without_frames", str(i))
        spans.append(TokenSpan(int(tok), int(frames[0]), int(frames[-1]) + 1,
                               float(np.exp(lp[frames, tok]).mean())))
    return spans


@dataclass
class Emissions:
    log_probs: np.ndarray        # (T, V)
    vocab: dict[str, int]
    blank: int
    frame_shift_s: float


def encode(words: Sequence[Word], vocab: dict[str, int]) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Token ids plus (word_index, unit_index, letter_index) per token."""
    ids, where = [], []
    for wi, w in enumerate(words):
        for ui, u in enumerate(w.units):
            for li, ch in enumerate(u.letters):
                if ch not in vocab:
                    raise AlignmentError("unknown_token", repr(ch))
                ids.append(vocab[ch])
                where.append((wi, ui, li))
    return ids, where


def align_words(em: Emissions, words: Sequence[Word]) -> dict[str, Any]:
    ids, where = encode(words, em.vocab)
    spans = ctc_align(em.log_probs, ids, em.blank)
    units: dict[tuple[int, int], list[TokenSpan]] = {}
    for (wi, ui, _), sp in zip(where, spans):
        units.setdefault((wi, ui), []).append(sp)
    vowels, word_spans = [], []
    for wi, w in enumerate(words):
        wsp = [sp for (a, _), sps in units.items() if a == wi for sp in sps]
        if wsp:
            word_spans.append({"word_index": wi, "start_s": min(s.start for s in wsp) * em.frame_shift_s,
                               "end_s": max(s.end for s in wsp) * em.frame_shift_s,
                               "conf": float(np.mean([s.score for s in wsp]))})
        for ui, u in enumerate(w.units):
            if u.kind != "vowel" or u.category is None or (wi, ui) not in units:
                continue
            sps = units[(wi, ui)]
            vowels.append({"word_index": wi, "unit_index": ui, "category": u.category,
                           "start_s": min(s.start for s in sps) * em.frame_shift_s,
                           "end_s": max(s.end for s in sps) * em.frame_shift_s,
                           "conf": float(np.mean([s.score for s in sps]))})
    conf = float(np.mean([s.score for s in spans]))
    return {"status": "aligned" if conf >= UTTERANCE_MIN_CONF else "low_confidence", "utterance_conf": conf,
            "n_tokens": len(ids), "words": word_spans, "vowels": vowels}


def align_transcript(em_fn: Callable[[np.ndarray], Emissions], audio: np.ndarray, text: str) -> dict[str, Any]:
    """Full pipeline for one utterance; never raises — failures are returned with a reason."""
    words, reason = clean_transcript(text)
    if words is None:
        return {"status": "excluded", "reason": reason}
    rw = romanize_telugu(words)
    if not rw:
        return {"status": "excluded", "reason": "no_alignable_letters"}
    try:
        em = em_fn(audio)
        result = align_words(em, rw)
    except AlignmentError as exc:
        return {"status": "failed", "reason": exc.reason}
    except Exception as exc:   # backend crash: recorded, not fatal
        return {"status": "failed", "reason": f"backend_error:{type(exc).__name__}"}
    return result


def accepted_vowels(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Vowel tokens usable for labelled measures (§13.3)."""
    if result.get("status") != "aligned":
        return []
    return [v for v in result["vowels"]
            if v["conf"] >= TOKEN_MIN_CONF and (v["end_s"] - v["start_s"]) >= VOWEL_MIN_S]


# --------------------------------------------------------------------------- backends (Colab)


def torchaudio_backend(device: str = "cpu") -> Callable[[np.ndarray], Emissions]:
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device).eval()
    labels = bundle.get_dict(star=None)

    def run(audio: np.ndarray) -> Emissions:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None].to(device)
        with torch.inference_mode():
            em, _ = model(wav)
            lp = torch.log_softmax(em[0], dim=-1).cpu().numpy()   # idempotent if already log-probs
        return Emissions(lp, dict(labels), labels.get("-", 0), wav.shape[1] / lp.shape[0] / bundle.sample_rate)

    return run


def hf_backend(model_id: str = "MahmoudAshraf/mms-300m-1130-forced-aligner", device: str = "cpu") -> Callable[[np.ndarray], Emissions]:
    import torch
    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    vocab = {k: v for k, v in tok.get_vocab().items() if len(k) == 1}
    blank = tok.pad_token_id if tok.pad_token_id is not None else 0

    def run(audio: np.ndarray) -> Emissions:
        inputs = fe(np.asarray(audio, dtype=np.float32), sampling_rate=16000, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return Emissions(lp, vocab, int(blank), len(audio) / lp.shape[0] / 16000)

    return run


def load_backend(preference: Sequence[str] = ("torchaudio", "hf"), device: str = "cpu") -> tuple[str, Callable]:
    errors = {}
    for name in preference:
        try:
            return name, (torchaudio_backend(device) if name == "torchaudio" else hf_backend(device=device))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
    raise AlignmentError("no_backend", str(errors))


# --------------------------------------------------------------------------- TextGrid


def write_textgrid(path, duration_s: float, tiers: dict[str, list[tuple[float, float, str]]]) -> None:
    """Minimal Praat long-format TextGrid with interval tiers (gaps filled with empty intervals)."""
    def filled(intervals):
        out, t = [], 0.0
        for a, b, lab in sorted(intervals):
            a, b = max(a, t), min(b, duration_s)
            if b <= a:
                continue
            if a > t:
                out.append((t, a, ""))
            out.append((a, b, lab))
            t = b
        if t < duration_s:
            out.append((t, duration_s, ""))
        return out or [(0.0, duration_s, "")]

    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration_s}",
             "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for k, (name, ivs) in enumerate(tiers.items(), 1):
        f = filled(ivs)
        lines += [f"    item [{k}]:", '        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                  f"        xmax = {duration_s}", f"        intervals: size = {len(f)}"]
        for j, (a, b, lab) in enumerate(f, 1):
            lab = str(lab).replace('"', '""')
            lines += [f"        intervals [{j}]:", f"            xmin = {a:.4f}", f"            xmax = {b:.4f}",
                      f'            text = "{lab}"']
    from pathlib import Path

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
