#!/usr/bin/env python3
"""Experiment 1 — per-recording acoustic features for atypical and typical Telugu speech.

Sources (read-only):
  atypical      Vaani Atypical Speech Corpus, Telugu config (HF, gated)
  controls_v2   <data-root>/typical_controls_v2/manifest.csv (balanced typical controls)
  vaani_copy    <data-root>/vaani_telugu_transcribed/ capped at --cap-per-speaker clips/speaker
                (deterministic salted-hash choice) — secondary control only

Writes <output-dir>/features.csv and run_config.json. No transcript text is written.
Speaker IDs are written (Drive only, research-only data) so speaker-level analysis is
possible; never commit these outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from intelli_speech import acoustic_features as af
from intelli_speech.experiment import ensure_outside_dataset, refuse_overwrite, utc_now, write_json
from intelli_speech.vaani import decode_audio_bytes

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
SOURCES = ("atypical", "controls_v2", "vaani_copy")


def _mono16k(audio: np.ndarray, rate: int) -> np.ndarray:
    from intelli_speech.vaani_controls import to_mono_16k

    return to_mono_16k(audio, rate)


def iter_atypical(cache_dir: Path | None) -> Iterator[dict[str, Any]]:
    from intelli_speech.vaani import load_atypical_telugu

    for index, row in enumerate(load_atypical_telugu(cache_dir)):
        audio_field = row["audio"] or {}
        data = audio_field.get("bytes") or Path(audio_field["path"]).read_bytes()
        audio, rate, _ = decode_audio_bytes(data)
        yield {"dataset": "atypical", "group": "atypical", "record_id": f"atyp_te_{index:04d}",
               "speaker_id": str(row["speaker_id"]), "difficulty_level": row.get("difficulty_level"),
               "prompt_file_type": row.get("prompt_file_type"), "audio": audio, "rate": rate}


def iter_controls_v2(directory: Path) -> Iterator[dict[str, Any]]:
    import soundfile as sf

    manifest = pd.read_csv(directory / "manifest.csv", dtype={"speakerID": str})
    for _, row in manifest[manifest["status"] == "ok"].iterrows():
        audio, rate = sf.read(str(directory / row["file"]), dtype="float32", always_2d=True)
        yield {"dataset": "controls_v2", "group": "typical", "record_id": row["control_id"],
               "speaker_id": str(row["speakerID"]), "gender": row.get("gender"), "district": row.get("district"),
               "audio": audio, "rate": rate}


def iter_vaani_copy(directory: Path, cap: int, salt: str = "exp1-vaani-copy") -> Iterator[dict[str, Any]]:
    from intelli_speech.vaani import load_typical_vaani_metadata

    frame = load_typical_vaani_metadata(directory)
    frame = frame.assign(_rank=[hashlib.sha1(f"{salt}|{s}|{f}".encode()).hexdigest() for s, f in zip(frame["speakerID"], frame["filename"])])
    frame = frame.sort_values("_rank").groupby("speakerID", group_keys=False).head(cap)
    for _, row in frame.iterrows():
        audio, rate, _ = decode_audio_bytes((directory / str(row["filename"]).strip()).read_bytes())
        yield {"dataset": "vaani_copy", "group": "typical", "record_id": f"typ_vaani_{Path(str(row['filename'])).stem}",
               "speaker_id": str(row["speakerID"]), "district": row.get("district"), "audio": audio, "rate": rate}


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--cap-per-speaker", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Per-source cap (smoke tests).")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--no-spectral", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser()
    output = args.output_dir or data_root / "analysis_v2" / f"exp1_features_{date.today().isoformat()}"
    for protected in (data_root / "vaani_telugu_transcribed", data_root / "typical_controls_v2"):
        output = ensure_outside_dataset(protected, output)
    features_path = refuse_overwrite(output / "features.csv", overwrite=args.overwrite)

    iterators = {
        "atypical": lambda: iter_atypical(args.hf_cache),
        "controls_v2": lambda: iter_controls_v2(data_root / "typical_controls_v2"),
        "vaani_copy": lambda: iter_vaani_copy(data_root / "vaani_telugu_transcribed", args.cap_per_speaker),
    }
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    started = time.time()
    for source in args.sources:
        ok = errors = 0
        required = {"controls_v2": data_root / "typical_controls_v2" / "manifest.csv",
                    "vaani_copy": data_root / "vaani_telugu_transcribed" / "metadata.csv"}.get(source)
        if required is not None and not required.is_file():
            counts[source] = {"ok": 0, "errors": 0, "skipped": f"missing {required}"}
            print(f"[{source}] skipped: missing {required.name}")
            continue
        iterator = iterators[source]()
        for k, item in enumerate(iterator):
            if args.limit and k >= args.limit:
                break
            audio, rate = item.pop("audio"), item.pop("rate")
            row = dict(item)
            try:
                mono = _mono16k(audio, rate)
                row.update(af.extract_recording_features(mono, 16000, include_voice=not args.no_voice, include_spectral=not args.no_spectral))
                ok += 1
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
            rows.append(row)
            if (ok + errors) % 25 == 0:
                print(f"[{source}] {ok + errors} recordings ({time.time() - started:.0f}s)", flush=True)
        counts[source] = {"ok": ok, "errors": errors}
        print(f"[{source}] done: {ok} ok, {errors} errors")

    pd.DataFrame(rows).to_csv(features_path, index=False)
    config = {
        "created_utc": utc_now(),
        "feature_version": af.FEATURE_VERSION,
        "git_sha": git_sha(),
        "sources": args.sources,
        "cap_per_speaker_vaani_copy": args.cap_per_speaker,
        "counts": counts,
        "parameters": {k: getattr(af, k) for k in dir(af) if k.isupper()},
        "seconds": round(time.time() - started, 1),
    }
    write_json(output / "run_config.json", config)
    return {"output_dir": str(output), **config}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {result['output_dir']}/features.csv  counts={result['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
