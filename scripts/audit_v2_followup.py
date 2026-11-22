#!/usr/bin/env python3
"""Experiment 0b — read-only follow-up audit.

A1  Full-scale analysis: clipping evidence vs consistency with peak normalisation
    (atypical Telugu and typical Vaani).
A2  Transcription script composition for `transcription` (intended) and
    `verbatim_transcription`, and the typical `transcript` field.
A3  Difficulty-label investigation: documentation lines quoted verbatim, then
    associations with available quality indicators (pooled AND within-speaker).

Writes only to --output-dir (default <data-root>/analysis_v2/audit_followup_<date>/).
Never writes or prints transcript text. Console output uses pseudonymous speaker
labels (S1..Sn); the mapping to contributor IDs is stored only in the output dir.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from intelli_speech.audio_integrity import full_scale_analysis
from intelli_speech.experiment import ensure_outside_dataset, refuse_overwrite, utc_now, write_json
from intelli_speech.grouped_stats import cliffs_delta, speaker_equal_weight_mean, stratified_permutation_test
from intelli_speech.telugu_text import character_error_rate
from intelli_speech.text_audit import script_profile
from intelli_speech.vaani import ATYPICAL_REPO
from scripts.audit_v2 import audio_quality, describe

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
DOC_KEYWORDS = ("difficulty", "understandab", " qc", "rating", "transcri", "verbatim", "reviewer")
ASSOCIATION_FEATURES = (
    "cer_verbatim_vs_intended",
    "duration_s",
    "rms_dbfs",
    "effective_bandwidth_hz",
    "energy_dynamic_range_db",
    "near_full_scale_fraction",
    "verbatim_aksharas_per_s",
    "intended_aksharas_per_s",
)


def read_pcm(data: bytes) -> tuple[np.ndarray, int, str, bool]:
    """Return (mono samples, sample_rate, subtype, is_pcm16) without float rescaling for PCM16."""
    import soundfile as sf

    info = sf.info(io.BytesIO(data))
    is_pcm16 = info.subtype == "PCM_16"
    audio, rate = sf.read(io.BytesIO(data), dtype="int16" if is_pcm16 else "float32", always_2d=True)
    mono = audio[:, 0] if audio.shape[1] == 1 else (audio.astype(np.float64).mean(axis=1))
    if is_pcm16 and audio.shape[1] > 1:
        mono = np.round(mono).astype(np.int16)
    return mono, int(rate), info.subtype, is_pcm16


def as_float(mono: np.ndarray, is_pcm16: bool) -> np.ndarray:
    return (mono.astype(np.float32) / 32768.0) if is_pcm16 else mono.astype(np.float32)


def pseudonyms(speakers: Iterable[str], prefix: str = "S") -> dict[str, str]:
    return {s: f"{prefix}{i}" for i, s in enumerate(sorted(set(speakers)), start=1)}


def text_fields(prefix: str, value: Any) -> dict[str, Any]:
    profile = script_profile(value)
    out = {f"{prefix}_missing": profile["missing"]}
    if not profile["missing"]:
        for key in ("chars_non_space", "words", "aksharas", "prop_telugu", "prop_latin",
                    "prop_other_letter", "prop_digit", "prop_symbol", "dominant_class"):
            out[f"{prefix}_{key}"] = profile[key]
    return out, profile


def summarize_text(rows: list[dict], prefix: str, symbol_counter: Counter) -> dict[str, Any]:
    present = [r for r in rows if not r.get(f"{prefix}_missing", True)]
    return {
        "missing": sum(1 for r in rows if r.get(f"{prefix}_missing", True)),
        "present": len(present),
        "dominant_class_counts": dict(Counter(r[f"{prefix}_dominant_class"] for r in present)),
        **{
            f"mean_{k}": round(float(np.mean([r[f"{prefix}_{k}"] for r in present])), 4) if present else None
            for k in ("prop_telugu", "prop_latin", "prop_other_letter", "prop_digit", "prop_symbol")
        },
        "chars_non_space": describe(r[f"{prefix}_chars_non_space"] for r in present),
        "words": describe(r[f"{prefix}_words"] for r in present),
        "aksharas": describe(r[f"{prefix}_aksharas"] for r in present),
        "most_common_symbols": symbol_counter.most_common(15),
    }


def paired_comparison(rows: list[dict], a: str, b: str) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank on per-recording values a vs b (pooled) + equal-weight speaker view."""
    from scipy.stats import wilcoxon

    pairs = [(r[a], r[b], r["speaker_id"]) for r in rows if r.get(a) is not None and r.get(b) is not None]
    if len(pairs) < 5:
        return {"n": len(pairs)}
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    diff = x - y
    result: dict[str, Any] = {
        "n": len(pairs),
        "median_difference": float(np.median(diff)),
        "fraction_a_greater": float(np.mean(diff > 0)),
        "speaker_equal_weight_mean_ratio_a_over_b": speaker_equal_weight_mean(
            (x / np.where(y == 0, np.nan, y)).tolist(), [p[2] for p in pairs]
        ),
        "note": "Pooled test treats recordings as independent; read with the speaker-level view.",
    }
    try:
        stat = wilcoxon(x, y, zero_method="wilcox")
        result["wilcoxon_p_pooled"] = float(stat.pvalue)
    except ValueError as exc:
        result["wilcoxon_p_pooled"] = None
        result["wilcoxon_note"] = str(exc)
    return result


def documentation_lines() -> list[str]:
    """Verbatim card lines mentioning the label/transcription vocabulary (public documentation)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ATYPICAL_REPO, "README.md", repo_type="dataset")
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [f"<README unavailable: {type(exc).__name__}>"]
    return [l.strip() for l in lines if any(k in l.lower() for k in DOC_KEYWORDS) and l.strip()][:60]


def difficulty_analysis(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("difficulty_level") in ("Low", "Medium")]
    out: dict[str, Any] = {
        "levels": dict(Counter(str(r.get("difficulty_level")) for r in rows)),
        "analysed_levels": ["Low", "Medium"],
        "excluded_note": "High has n=1 and is excluded from association tests.",
        "features": {},
    }
    for feature in ASSOCIATION_FEATURES:
        values = [r.get(feature) for r in ok]
        if sum(v is not None for v in values) < 10:
            continue
        vals = np.array([np.nan if v is None else float(v) for v in values])
        labels = [r["difficulty_level"] for r in ok]
        groups = [r["speaker_id"] for r in ok]
        medium = vals[np.array(labels) == "Medium"]
        low = vals[np.array(labels) == "Low"]
        within = stratified_permutation_test(vals, labels, groups, positive="Medium", negative="Low", n_permutations=2000)
        out["features"][feature] = {
            "pooled_cliffs_delta_medium_vs_low": cliffs_delta(medium.tolist(), low.tolist()),
            "pooled_note": "Pooled comparison is confounded with speaker (Cramer's V 0.51).",
            "within_speaker": {k: within[k] for k in ("eligible_groups", "statistic", "p_value", "sign_consistency") if k in within},
        }
    out["direction_conclusion"] = (
        "UNKNOWN: the dataset card defines difficulty_level only as 'Low / Medium / High (from QC "
        "understandability rating)' and does not state the scale direction. Associations below can be "
        "consistent with one reading but cannot establish it."
    )
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_atypical(cache_dir: Path | None) -> tuple[list[dict], dict[str, Counter]]:
    from intelli_speech.vaani import load_atypical_telugu

    dataset = load_atypical_telugu(cache_dir)
    rows: list[dict] = []
    symbols = {"intended": Counter(), "verbatim": Counter()}
    for index, item in enumerate(dataset):
        audio_field = item["audio"] or {}
        data = audio_field.get("bytes") or (Path(audio_field["path"]).read_bytes() if audio_field.get("path") else None)
        row: dict[str, Any] = {
            "record_id": f"atyp_te_{index:04d}",
            "speaker_id": str(item["speaker_id"]),
            "difficulty_level": item.get("difficulty_level"),
        }
        try:
            mono, rate, subtype, is_pcm16 = read_pcm(data)
            fs = full_scale_analysis(mono, is_pcm16=is_pcm16)
            row.update({"sample_rate": rate, "subtype": subtype})
            row.update({k: v for k, v in fs.items() if k != "thresholds"})
            row.update(audio_quality(as_float(mono, is_pcm16)[:, None], rate))
        except Exception as exc:
            row["decode_error"] = f"{type(exc).__name__}: {exc}"
        intended, verbatim = item.get("transcription"), item.get("verbatim_transcription")
        t_fields, t_prof = text_fields("intended", intended)
        v_fields, v_prof = text_fields("verbatim", verbatim)
        row.update(t_fields)
        row.update(v_fields)
        symbols["intended"].update(t_prof.get("symbols", {}))
        symbols["verbatim"].update(v_prof.get("symbols", {}))
        if not t_prof["missing"] and not v_prof["missing"]:
            row["cer_verbatim_vs_intended"] = round(character_error_rate(intended, verbatim), 4)
        duration = row.get("duration_s")
        if duration:
            for prefix in ("intended", "verbatim"):
                if row.get(f"{prefix}_aksharas") is not None:
                    row[f"{prefix}_aksharas_per_s"] = round(row[f"{prefix}_aksharas"] / duration, 4)
        rows.append(row)
    return rows, symbols


def audit_typical(directory: Path) -> tuple[list[dict], Counter]:
    from intelli_speech.vaani import load_typical_vaani_metadata

    frame = load_typical_vaani_metadata(directory)
    rows, symbols = [], Counter()
    for _, item in frame.iterrows():
        row: dict[str, Any] = {"record_id": f"typ_vaani_{Path(str(item['filename'])).stem}", "speaker_id": str(item["speakerID"])}
        try:
            mono, rate, subtype, is_pcm16 = read_pcm((directory / str(item["filename"]).strip()).read_bytes())
            fs = full_scale_analysis(mono, is_pcm16=is_pcm16)
            row.update({"sample_rate": rate, "subtype": subtype})
            row.update({k: v for k, v in fs.items() if k != "thresholds"})
        except Exception as exc:
            row["decode_error"] = f"{type(exc).__name__}: {exc}"
        fields, profile = text_fields("transcript", item.get("transcript"))
        row.update(fields)
        symbols.update(profile.get("symbols", {}))
        rows.append(row)
    return rows, symbols


def integrity_summary(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "verdict" in r]
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        by_speaker[r["speaker_id"]][r["verdict"]] += 1
    return {
        "files": len(rows),
        "analysed": len(ok),
        "subtypes": dict(Counter(r.get("subtype") for r in ok)),
        "verdicts": dict(Counter(r["verdict"] for r in ok)),
        "exact_full_scale_samples": describe(r["exact_full_scale_samples"] for r in ok),
        "near_full_scale_samples": describe(r["near_full_scale_samples"] for r in ok),
        "near_full_scale_fraction": describe(r["near_full_scale_fraction"] for r in ok),
        "max_consecutive_near_full_scale": describe(r["max_consecutive_near_full_scale"] for r in ok),
        "files_with_run_ge_3": sum(1 for r in ok if r["max_consecutive_near_full_scale"] >= 3),
        "verdicts_by_speaker": {s: dict(c) for s, c in sorted(by_speaker.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--skip-atypical", action="store_true")
    parser.add_argument("--skip-typical", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def pseudonymize(summary_part: dict, mapping: dict[str, str]) -> dict:
    """Copy of a summary with speaker IDs replaced (for console printing)."""
    def walk(obj):
        if isinstance(obj, dict):
            return {mapping.get(k, k): walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj
    return walk(summary_part)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser()
    typical_dir = data_root / "vaani_telugu_transcribed"
    output = args.output_dir or data_root / "analysis_v2" / f"audit_followup_{date.today().isoformat()}"
    output = ensure_outside_dataset(typical_dir, output)
    summary_path = refuse_overwrite(output / "summary.json", overwrite=args.overwrite)
    summary: dict[str, Any] = {"created_utc": utc_now(), "read_only": True, "output_dir": str(output)}
    mapping: dict[str, str] = {}

    if not args.skip_atypical:
        rows, symbols = audit_atypical(args.hf_cache)
        mapping = pseudonyms(r["speaker_id"] for r in rows)
        summary["atypical_integrity"] = integrity_summary(rows)
        summary["atypical_text"] = {
            "intended": summarize_text(rows, "intended", symbols["intended"]),
            "verbatim": summarize_text(rows, "verbatim", symbols["verbatim"]),
            "chars_verbatim_vs_intended": paired_comparison(rows, "verbatim_chars_non_space", "intended_chars_non_space"),
            "aksharas_verbatim_vs_intended": paired_comparison(rows, "verbatim_aksharas", "intended_aksharas"),
            "cer_verbatim_vs_intended": describe(r.get("cer_verbatim_vs_intended") for r in rows),
        }
        summary["difficulty"] = difficulty_analysis(rows)
        summary["difficulty"]["documentation_lines"] = documentation_lines()
        write_rows(output / "atypical_followup_per_file.csv", rows)

    if not args.skip_typical:
        rows_t, symbols_t = audit_typical(typical_dir)
        mapping.update(pseudonyms((r["speaker_id"] for r in rows_t), prefix="T"))
        summary["typical_integrity"] = integrity_summary(rows_t)
        summary["typical_text"] = {"transcript": summarize_text(rows_t, "transcript", symbols_t)}
        write_rows(output / "typical_followup_per_file.csv", rows_t)

    if mapping:
        write_json(output / "speaker_pseudonyms.json", {"note": "Research-only contributor IDs; do not commit.", "mapping": mapping})
        summary["speaker_pseudonyms_file"] = str(output / "speaker_pseudonyms.json")
    write_json(summary_path, summary)
    summary["_console"] = pseudonymize({k: v for k, v in summary.items()}, mapping)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Follow-up audit failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {summary['output_dir']}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
