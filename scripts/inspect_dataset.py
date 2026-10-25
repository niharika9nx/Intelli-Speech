#!/usr/bin/env python3
"""Read-only, bounded reconnaissance for a speech dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Intelli-Speech-data/vaani_telugu_transcribed"
)
AUDIO_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg",
    ".opus", ".wav", ".wma",
}
METADATA_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".ndjson", ".tsv", ".txt", ".xml", ".yaml",
    ".yml",
}
TRANSCRIPT_WORDS = ("transcript", "sentence", "utterance", "text", "transcription")
REFERENCE_WORDS = ("audio", "file", "path", "recording", "sound", "wav")
IDENTIFIER_WORDS = ("id", "identifier", "utt", "utterance", "sample", "filename")


def extension(path: Path) -> str:
    return path.suffix.lower() or "[no extension]"


def data_type(value: Any) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (dict, list)):
        return type(value).__name__
    return "string"


def normalized_header(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def read_text_with_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode metadata file: {path.name}")


def inspect_records(
    records: Iterable[dict[str, Any]], root: Path, audio_paths: set[str], row_limit: int
) -> dict[str, Any]:
    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    null_counts: Counter[str] = Counter()
    transcript_counts: Counter[str] = Counter()
    reference_results: dict[str, Counter[str]] = defaultdict(Counter)
    identifier_values: dict[str, set[str]] = defaultdict(set)
    duplicate_ids: Counter[str] = Counter()
    rows = 0

    for record in records:
        if rows >= row_limit:
            break
        rows += 1
        for key, value in record.items():
            name = str(key)
            header = normalized_header(name)
            field_types[name][data_type(value)] += 1
            if value is None or value == "":
                null_counts[name] += 1
            if any(word in header for word in TRANSCRIPT_WORDS) and value not in (None, ""):
                transcript_counts[name] += 1
            if any(word in header for word in IDENTIFIER_WORDS) and value not in (None, ""):
                identifier = str(value).strip()
                if identifier in identifier_values[name]:
                    duplicate_ids[name] += 1
                identifier_values[name].add(identifier)
            if any(word in header for word in REFERENCE_WORDS) and value not in (None, ""):
                ref = str(value).strip()
                if Path(ref).suffix.lower() in AUDIO_EXTENSIONS or "/" in ref or "\\" in ref:
                    resolved = Path(ref)
                    if not resolved.is_absolute():
                        resolved = root / resolved
                    if resolved.is_file():
                        reference_results[name]["existing"] += 1
                    elif Path(ref).name.lower() in audio_paths:
                        reference_results[name]["matched_by_basename"] += 1
                    else:
                        reference_results[name]["missing"] += 1

    fields = {}
    for name, types in sorted(field_types.items()):
        fields[name] = {
            "observed_types": dict(sorted(types.items())),
            "null_or_empty_rows": null_counts[name],
        }

    return {
        "rows_inspected": rows,
        "fields": fields,
        "nonempty_transcript_field_rows": dict(sorted(transcript_counts.items())),
        "audio_reference_checks": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(reference_results.items())
        },
        "duplicate_identifier_values": dict(sorted(duplicate_ids.items())),
    }


def delimited_records(
    path: Path, row_limit: int
) -> tuple[list[dict[str, Any]], str, bool, str]:
    text, encoding = read_text_with_encoding(path)
    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    else:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    records = []
    for row in reader:
        if len(records) >= row_limit:
            return records, delimiter, True, encoding
        records.append(row)
    return records, delimiter, False, encoding


def json_records(path: Path, row_limit: int) -> tuple[list[dict[str, Any]], bool, str]:
    text, encoding = read_text_with_encoding(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if len(records) >= row_limit:
                return records, True, encoding
            if isinstance(value, dict):
                records.append(value)
        return records, False, encoding

    value = json.loads(text)
    if isinstance(value, dict):
        for child in value.values():
            if isinstance(child, list) and (not child or isinstance(child[0], dict)):
                value = child
                break
        else:
            value = [value]
    if not isinstance(value, list):
        return [], False, encoding
    records = [item for item in value[:row_limit] if isinstance(item, dict)]
    return records, len(value) > row_limit, encoding


def inspect_metadata(
    path: Path, root: Path, audio_paths: set[str], byte_limit: int, row_limit: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lower(),
    }
    if result["size_bytes"] > byte_limit:
        result["status"] = "skipped_size_limit"
        return result

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        records, delimiter, truncated, encoding = delimited_records(path, row_limit)
        result["delimiter"] = "tab" if delimiter == "\t" else delimiter
        result["encoding"] = encoding
        result["inspection_truncated_at_row_limit"] = truncated
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        records, truncated, encoding = json_records(path, row_limit)
        result["encoding"] = encoding
        result["inspection_truncated_at_row_limit"] = truncated
    elif suffix == ".txt":
        text, encoding = read_text_with_encoding(path)
        result["line_count"] = len(text.splitlines())
        result["encoding"] = encoding
        result["status"] = "text_file_not_interpreted_as_metadata"
        return result
    else:
        result["status"] = "format_listed_not_parsed"
        return result

    result["status"] = "parsed"
    result.update(inspect_records(records, root, audio_paths, row_limit))
    return result


def evenly_spaced(items: list[Path], limit: int) -> list[Path]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return items[:limit]
    indexes = {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    return [items[index] for index in sorted(indexes)]


def inspect_audio_sample(paths: list[Path], root: Path) -> dict[str, Any]:
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError:
        sf = None

    observations = []
    failures = Counter()
    for path in paths:
        try:
            if sf is not None:
                info = sf.info(str(path))
                with sf.SoundFile(str(path)) as audio:
                    audio.read(frames=min(4096, len(audio)))
                observations.append(
                    {
                        "extension": extension(path),
                        "container_format": info.format,
                        "subtype": info.subtype,
                        "sample_rate_hz": info.samplerate,
                        "channels": info.channels,
                        "duration_seconds": info.duration,
                        "decode_check": "read_prefix",
                    }
                )
            elif extension(path) == ".wav":
                with wave.open(str(path), "rb") as audio:
                    frames = audio.getnframes()
                    rate = audio.getframerate()
                    audio.readframes(min(4096, frames))
                    observations.append(
                        {
                            "extension": extension(path),
                            "container_format": "WAV",
                            "sample_rate_hz": rate,
                            "channels": audio.getnchannels(),
                            "duration_seconds": frames / rate if rate else None,
                            "decode_check": "read_prefix",
                        }
                    )
            else:
                failures["decoder_unavailable"] += 1
        except Exception as error:  # Per-file failures are reported, not hidden.
            failures[type(error).__name__] += 1

    def distribution(field: str) -> dict[str, Any]:
        values = [item[field] for item in observations if item.get(field) is not None]
        counts = Counter(values)
        summary: dict[str, Any] = {"counts": dict(sorted(counts.items(), key=lambda x: str(x[0])))}
        if values and all(isinstance(value, (int, float)) for value in values):
            summary["min"] = min(values)
            summary["median"] = statistics.median(values)
            summary["max"] = max(values)
        return summary

    return {
        "sampling_method": "evenly spaced from lexicographically sorted audio candidates",
        "sampled_files": len(paths),
        "decoded_files": len(observations),
        "decode_failures_by_error": dict(sorted(failures.items())),
        "decoder": "soundfile" if sf is not None else "Python wave fallback (WAV only)",
        "sampled_properties": {
            key: distribution(key)
            for key in ("extension", "container_format", "subtype", "sample_rate_hz", "channels", "duration_seconds")
        },
        "decode_scope": "A prefix was decoded for each supported sampled file; full-file integrity was not verified.",
    }


def collect_dataset(root: Path, audio_sample_size: int, metadata_byte_limit: int, row_limit: int) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist or is not a directory: {root}")

    files: list[Path] = []
    directories: list[Path] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        current_path = Path(current)
        directories.append(current_path)
        files.extend(current_path / name for name in filenames)
    files.sort(key=lambda path: path.as_posix().lower())

    file_counts_by_directory: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    empty_files = 0
    for path in files:
        relative_parent = path.parent.relative_to(root).as_posix() or "."
        file_counts_by_directory[relative_parent] += 1
        ext = extension(path)
        extension_counts[ext] += 1
        try:
            size = path.stat().st_size
            extension_bytes[ext] += size
            if size == 0:
                empty_files += 1
        except OSError:
            extension_bytes[ext] += 0

    audio_files = [path for path in files if extension(path) in AUDIO_EXTENSIONS]
    audio_paths = {path.name.lower() for path in audio_files}
    metadata_files = [path for path in files if extension(path) in METADATA_EXTENSIONS]
    metadata = {}
    for path in metadata_files:
        relative = path.relative_to(root).as_posix()
        try:
            metadata[relative] = inspect_metadata(
                path, root, audio_paths, metadata_byte_limit, row_limit
            )
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as error:
            metadata[relative] = {
                "status": "inspection_error",
                "error_type": type(error).__name__,
            }

    text_stems = {path.stem.lower() for path in files if extension(path) == ".txt"}
    audio_stems = {path.stem.lower() for path in audio_files}
    return {
        "dataset_root": str(root),
        "read_only": True,
        "inspection_limits": {
            "audio_sample_size": audio_sample_size,
            "metadata_file_size_limit_bytes": metadata_byte_limit,
            "metadata_row_limit": row_limit,
        },
        "structure": {
            "file_count": len(files),
            "directory_count_including_root": len(directories),
            "empty_directories": sorted(
                path.relative_to(root).as_posix() or "."
                for path in directories
                if not any(path.iterdir())
            ),
            "files_by_directory": dict(sorted(file_counts_by_directory.items())),
            "file_count_by_extension": dict(sorted(extension_counts.items())),
            "bytes_by_extension": dict(sorted(extension_bytes.items())),
            "zero_byte_file_count": empty_files,
        },
        "audio_candidates": {
            "extension_set": sorted(AUDIO_EXTENSIONS),
            "candidate_count": len(audio_files),
            "empty_candidate_count": sum(path.stat().st_size == 0 for path in audio_files),
            "sample_inspection": inspect_audio_sample(
                evenly_spaced(audio_files, audio_sample_size), root
            ),
        },
        "metadata_files": metadata,
        "text_audio_stem_overlap": {
            "matching_distinct_stems": len(audio_stems & text_stems),
            "distinct_audio_stems": len(audio_stems),
            "distinct_txt_stems": len(text_stems),
            "note": "A matching basename is only a candidate relationship, not proof of transcript alignment.",
        },
        "enhancement_objective_indicators": {
            "clean_noisy_pair_evidence": (
                "Not established automatically; inspect reviewed metadata fields and sample relationships."
            ),
            "limitation": (
                "Filename and metadata heuristics cannot prove clean/noisy pairing, alignment, "
                "or suitability for supervised enhancement."
            ),
        },
        "limitations": [
            "Audio decoding covers only a deterministic sample and reads a prefix, not every full file.",
            "Audio candidates are identified by filename extension; unsupported or mislabeled files may be missed.",
            "Metadata schemas are inspected with bounded, common-format parsers; XML/YAML are listed but not parsed.",
            "TXT files are counted but their contents are never printed or interpreted.",
            "Transcript content and sample-level metadata values are not included in this summary.",
            "Automatic reference and basename checks are heuristic and do not establish semantic alignment.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset directory (read-only).")
    parser.add_argument("--audio-sample-size", type=int, default=100)
    parser.add_argument("--metadata-byte-limit", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--metadata-row-limit", type=int, default=100_000)
    args = parser.parse_args()
    if min(args.audio_sample_size, args.metadata_byte_limit, args.metadata_row_limit) < 1:
        parser.error("inspection limits must all be positive")
    try:
        result = collect_dataset(
            args.root.expanduser(),
            args.audio_sample_size,
            args.metadata_byte_limit,
            args.metadata_row_limit,
        )
    except (OSError, ValueError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
