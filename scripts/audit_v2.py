#!/usr/bin/env python3
"""Experiment 0 — read-only audit of the v2 Telugu datasets.

Sources (all read-only):
  * Vaani Atypical Speech Corpus, Telugu config (Hugging Face, gated; log in first).
  * Typical Vaani Telugu copy on Drive: <data-root>/vaani_telugu_transcribed/.
  * OpenSLR SLR66 archives on Drive: <data-root>/openslr_slr66/ (listing + small decode sample).

Writes only to --output-dir: summary.json and CSV tables. Transcript text is never
written or printed; only lengths, akshara counts and CER numbers derived from it.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from intelli_speech.experiment import ensure_outside_dataset, refuse_overwrite, utc_now, write_json
from intelli_speech.metrics import clipping_report
from intelli_speech.telugu_text import akshara_count, character_error_rate, normalize_text
from intelli_speech.vaani import AudioRecord, decode_audio_bytes

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
FRAME_SEC = 0.02


# --------------------------------------------------------------------------- signal metrics


def audio_quality(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Recording-condition descriptors for one file (not speech characteristics)."""
    if audio.size == 0 or sample_rate <= 0:
        return {"duration_s": 0.0}
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    clip = clipping_report(mono, full_scale_threshold=0.999)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    frame = max(1, int(FRAME_SEC * sample_rate))
    n = mono.size // frame
    frame_db = None
    if n >= 3:
        frames = mono[: n * frame].reshape(n, frame).astype(np.float64)
        frame_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    spectrum = None
    if mono.size >= 512:
        hop = 256
        count = 1 + (mono.size - 512) // hop
        window = np.hanning(512)
        stack = np.stack([mono[i * hop : i * hop + 512] * window for i in range(count)])
        spectrum = np.mean(np.abs(np.fft.rfft(stack, axis=1)) ** 2, axis=0)
    result = {
        "duration_s": round(mono.size / sample_rate, 4),
        "peak": round(clip["peak_amplitude"], 5),
        "near_full_scale_samples": clip["full_scale_sample_count"],
        "near_full_scale_fraction": round(clip["full_scale_sample_count"] / mono.size, 6),
        "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
        "dc_offset": round(float(mono.mean()), 6),
    }
    if frame_db is not None:
        result["silence_fraction_40db"] = round(float(np.mean(frame_db < frame_db.max() - 40)), 4)
        # Proxy only: spread between loud (speech) and quiet (background) frames.
        result["energy_dynamic_range_db"] = round(
            float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10)), 2
        )
    if spectrum is not None and spectrum.sum() > 0:
        cumulative = np.cumsum(spectrum) / spectrum.sum()
        freqs = np.fft.rfftfreq(512, 1.0 / sample_rate)
        result["effective_bandwidth_hz"] = float(freqs[int(np.searchsorted(cumulative, 0.99))])
    return result


# --------------------------------------------------------------------------- statistics helpers


def cramers_v(table: dict[str, Counter]) -> float | None:
    """Cramér's V for a speaker × label contingency table (None if degenerate)."""
    rows = sorted(table)
    cols = sorted({c for counter in table.values() for c in counter})
    if len(rows) < 2 or len(cols) < 2:
        return None
    observed = np.array([[table[r].get(c, 0) for c in cols] for r in rows], dtype=float)
    total = observed.sum()
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    return float(np.sqrt(chi2 / (total * (min(observed.shape) - 1))))


def describe(values: Iterable[float]) -> dict[str, float] | None:
    array = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or not normalize_text(str(value))


# --------------------------------------------------------------------------- per-dataset audits


def audit_records(records: Iterable[AudioRecord], *, kind: str) -> tuple[list[dict], dict[str, Any]]:
    """Shared per-file audit; returns per-file rows (no transcript text) and a summary."""
    rows: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        meta = record.metadata
        row: dict[str, Any] = {
            "record_id": record.record_id,
            "speaker_id": record.speaker_id,
            "sample_rate": record.sample_rate,
            "channels": record.channels,
            "decode_error": meta.get("decode_error", ""),
        }
        row.update(audio_quality(record.waveform, record.sample_rate))
        if not row["decode_error"]:
            hashes[record.sha1].append(record.record_id)
        if kind == "atypical":
            intended, verbatim = meta.get("transcription"), meta.get("verbatim_transcription")
            row.update(
                {
                    "difficulty_level": meta.get("difficulty_level"),
                    "prompt_file_type": meta.get("prompt_file_type"),
                    "prompt_file_name": meta.get("prompt_file_name"),
                    "metadata_duration_s": meta.get("duration_s"),
                    "intended_empty": _is_empty(intended),
                    "verbatim_empty": _is_empty(verbatim),
                    "intended_aksharas": 0 if _is_empty(intended) else akshara_count(intended),
                    "verbatim_aksharas": 0 if _is_empty(verbatim) else akshara_count(verbatim),
                    "cer_verbatim_vs_intended": None
                    if _is_empty(intended) or _is_empty(verbatim)
                    else round(character_error_rate(intended, verbatim), 4),
                }
            )
        else:
            transcript = meta.get("transcript")
            row.update(
                {
                    "district": meta.get("district"),
                    "metadata_duration": meta.get("duration"),
                    "transcript_empty": _is_empty(transcript),
                    "transcript_aksharas": 0 if _is_empty(transcript) else akshara_count(transcript),
                }
            )
        rows.append(row)

    ok = [r for r in rows if not r["decode_error"]]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_speaker[r["speaker_id"]].append(r)
    summary: dict[str, Any] = {
        "files": len(rows),
        "decoded": len(ok),
        "decode_errors": len(rows) - len(ok),
        "speakers": len(by_speaker),
        "total_minutes": round(sum(r["duration_s"] for r in ok) / 60, 2),
        "duration_s": describe(r["duration_s"] for r in ok),
        "sample_rates": dict(Counter(r["sample_rate"] for r in ok)),
        "channels": dict(Counter(r["channels"] for r in ok)),
        "duplicate_audio_groups": [ids for ids in hashes.values() if len(ids) > 1],
        "files_with_near_full_scale_samples": sum(1 for r in ok if r.get("near_full_scale_samples", 0) > 0),
        "rms_dbfs": describe(r.get("rms_dbfs") for r in ok),
        "energy_dynamic_range_db": describe(r.get("energy_dynamic_range_db") for r in ok),
        "effective_bandwidth_hz": describe(r.get("effective_bandwidth_hz") for r in ok),
        "clips_per_speaker": describe(len(v) for v in by_speaker.values()),
    }
    return rows, summary


def speaker_table(rows: list[dict], *, kind: str) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["decode_error"]:
            by_speaker[r["speaker_id"]].append(r)
    table = []
    for speaker, items in sorted(by_speaker.items()):
        entry = {
            "speaker_id": speaker,
            "clips": len(items),
            "minutes": round(sum(i["duration_s"] for i in items) / 60, 2),
            "mean_duration_s": round(float(np.mean([i["duration_s"] for i in items])), 2),
            "median_rms_dbfs": round(float(np.median([i["rms_dbfs"] for i in items])), 2),
            "median_bandwidth_hz": round(float(np.median([i.get("effective_bandwidth_hz", np.nan) for i in items])), 1),
            "median_dynamic_range_db": round(float(np.median([i.get("energy_dynamic_range_db", np.nan) for i in items])), 2),
        }
        if kind == "atypical":
            levels = Counter(str(i["difficulty_level"]) for i in items)
            entry.update({f"difficulty_{k}": v for k, v in sorted(levels.items())})
            entry["verbatim_empty"] = sum(bool(i["verbatim_empty"]) for i in items)
            cers = [i["cer_verbatim_vs_intended"] for i in items if i["cer_verbatim_vs_intended"] is not None]
            entry["median_cer_verbatim_vs_intended"] = round(float(np.median(cers)), 4) if cers else None
            entry["distinct_prompts"] = len({i["prompt_file_name"] for i in items})
        else:
            entry["districts"] = len({i["district"] for i in items})
        table.append(entry)
    return table


def atypical_extras(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r["decode_error"]]
    crosstab: dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        crosstab[r["speaker_id"]][str(r["difficulty_level"])] += 1
    prompt_speakers: dict[str, set] = defaultdict(set)
    for r in ok:
        prompt_speakers[str(r["prompt_file_name"])].add(r["speaker_id"])
    mismatch = [
        abs(float(r["metadata_duration_s"]) - r["duration_s"])
        for r in ok
        if r.get("metadata_duration_s") not in (None, "")
    ]
    return {
        "difficulty_levels": dict(Counter(str(r["difficulty_level"]) for r in ok)),
        "difficulty_by_speaker": {s: dict(c) for s, c in sorted(crosstab.items())},
        "difficulty_speaker_cramers_v": cramers_v(crosstab),
        "prompt_types": dict(Counter(str(r["prompt_file_type"]) for r in ok)),
        "distinct_prompts": len(prompt_speakers),
        "prompts_shared_by_2plus_speakers": sum(1 for s in prompt_speakers.values() if len(s) >= 2),
        "intended_empty": sum(bool(r["intended_empty"]) for r in ok),
        "verbatim_empty": sum(bool(r["verbatim_empty"]) for r in ok),
        "cer_verbatim_vs_intended": describe(r["cer_verbatim_vs_intended"] for r in ok),
        "max_abs_duration_mismatch_s": round(max(mismatch), 4) if mismatch else None,
        "note": "difficulty_level is a QC understandability rating, not clinical severity.",
    }


def audit_slr66(directory: Path, sample: int, seed: int) -> dict[str, Any]:
    """Listing-level SLR66 audit plus a small deterministic decode sample from the zips."""
    result: dict[str, Any] = {"directory": str(directory), "exists": directory.is_dir()}
    if not directory.is_dir():
        return result
    rng = np.random.default_rng(seed)
    for gender in ("female", "male"):
        index_path = directory / f"line_index_{gender}.tsv"
        zip_path = directory / f"te_in_{gender}.zip"
        entry: dict[str, Any] = {}
        if index_path.is_file():
            lines = [l for l in index_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            ids = [l.split("\t")[0].strip() for l in lines]
            speakers = Counter("_".join(i.split("_")[:2]) for i in ids)
            entry.update(
                {
                    "index_rows": len(ids),
                    "unique_ids": len(set(ids)),
                    "speakers_from_filename_prefix": len(speakers),
                    "clips_per_speaker": describe(speakers.values()),
                    "empty_transcripts": sum(1 for l in lines if len(l.split("\t")) < 2 or not l.split("\t")[1].strip()),
                }
            )
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path) as archive:
                wavs = [n for n in archive.namelist() if n.lower().endswith(".wav")]
                entry["zip_wav_members"] = len(wavs)
                chosen = [wavs[i] for i in sorted(rng.choice(len(wavs), min(sample, len(wavs)), replace=False))] if wavs else []
                durations, rates, channels, errors = [], Counter(), Counter(), 0
                for name in chosen:
                    try:
                        audio, rate, ch = decode_audio_bytes(archive.read(name))
                        durations.append(audio.shape[0] / rate)
                        rates[rate] += 1
                        channels[ch] += 1
                    except Exception:
                        errors += 1
                entry["decode_sample"] = {
                    "files": len(chosen),
                    "errors": errors,
                    "duration_s": describe(durations),
                    "sample_rates": dict(rates),
                    "channels": dict(channels),
                }
        result[gender] = entry
    result["note"] = "Speaker IDs are filename-derived and unverified. SLR66 is read speech (task confound vs Vaani)."
    return result


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=None, help="Hugging Face cache dir (e.g. on Drive).")
    parser.add_argument("--skip-atypical", action="store_true")
    parser.add_argument("--skip-typical", action="store_true")
    parser.add_argument("--skip-slr66", action="store_true")
    parser.add_argument("--slr66-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser()
    typical_dir = data_root / "vaani_telugu_transcribed"
    slr66_dir = data_root / "openslr_slr66"
    output = args.output_dir or data_root / "analysis_v2" / f"audit_{date.today().isoformat()}"
    for protected in (typical_dir, slr66_dir):
        output = ensure_outside_dataset(protected, output)
    summary_path = refuse_overwrite(output / "summary.json", overwrite=args.overwrite)

    summary: dict[str, Any] = {"created_utc": utc_now(), "read_only": True, "output_dir": str(output)}

    if not args.skip_atypical:
        from intelli_speech.vaani import iter_atypical, load_atypical_telugu

        rows, part = audit_records(iter_atypical(load_atypical_telugu(args.hf_cache)), kind="atypical")
        part.update(atypical_extras(rows))
        summary["atypical_telugu"] = part
        write_rows(output / "atypical_per_file.csv", rows)
        write_rows(output / "atypical_per_speaker.csv", speaker_table(rows, kind="atypical"))
        print(f"[atypical] {part['decoded']}/{part['files']} files, {part['speakers']} speakers, {part['total_minutes']} min")

    if not args.skip_typical:
        from intelli_speech.vaani import iter_typical_vaani, load_typical_vaani_metadata

        metadata = load_typical_vaani_metadata(typical_dir)
        rows, part = audit_records(iter_typical_vaani(typical_dir, metadata), kind="typical")
        part["metadata_columns"] = list(metadata.columns)
        part["has_prompt_or_image_column"] = any(
            k in c.lower() for c in metadata.columns for k in ("prompt", "image")
        )
        part["wav_files_on_disk"] = len(list(typical_dir.glob("*.wav")))
        part["districts"] = len({r["district"] for r in rows})
        summary["typical_vaani"] = part
        write_rows(output / "typical_per_file.csv", rows)
        write_rows(output / "typical_per_speaker.csv", speaker_table(rows, kind="typical"))
        print(f"[typical] {part['decoded']}/{part['files']} files, {part['speakers']} speakers, {part['total_minutes']} min")

    if not args.skip_slr66:
        summary["slr66"] = audit_slr66(slr66_dir, args.slr66_sample, args.seed)
        print("[slr66] audited listing + decode sample")

    write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {summary['output_dir']}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
