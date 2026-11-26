#!/usr/bin/env python3
"""Experiment 2 — Stage A (measurement validity) runner. EXP2_PLAN.md v0.2.1 §8, §13.

Typical speech only (plus a sealed atypical feasibility check). No group comparison.

Phases (run in this order; each refuses to overwrite its own outputs unless --overwrite):
  measure               label-free measures on typical clips (controls_v2 + reference sample)
  align                 MMS forced alignment of verbatim typical transcripts; labelled tokens;
                        frame purity (needs torchaudio MMS_FA or transformers; GPU recommended)
  synthetic             synthetic formant-recovery grid (§13.4)
  channel               channel-robustness conditions on controls_v2 (ffmpeg codecs if present)
  detectability         WORLD envelope perturbation test stimuli (needs pyworld) — NOT validity
  manual-sheet          export 30-utterance manual check (WAV + TextGrid + sheets). NEVER overwritten.
  manual-summary        read the filled sheets → acceptance (§13.3)
  lexical               text-only vowel-category comparability flag
  atypical-feasibility  split-half ICC + sufficiency for the 7 atypical speakers; writes ONLY
                        ICCs and counts (no feature values) to keep Stage B blind
  report                combine everything → stage_a_report.json

Outputs: <data-root>/analysis_v2/exp2_stage_a/. Speaker IDs stay on Drive; never commit outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from intelli_speech import alignment as al
from intelli_speech import articulation_features as ar
from intelli_speech import exp2_stage_a as sa
from intelli_speech.experiment import refuse_overwrite, utc_now, write_json

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Intelli-Speech-data")
REFERENCE_MAX_SPEAKERS = 400
REFERENCE_PER_SPEAKER = 3
DETECT_N_CLIPS = 50
DETECT_WARP = (0.1, 0.2, 0.3)          # = retained factors 0.9 / 0.8 / 0.7 (§8.2)
DETECT_SMOOTH_S = (0.03, 0.06, 0.09)
MEASURES = sa.H1 + sa.H23 + sa.H4


# --------------------------------------------------------------------------- data


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    from intelli_speech.vaani_controls import to_mono_16k

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return to_mono_16k(audio, rate)


def controls_clips(data_root: Path) -> pd.DataFrame:
    d = data_root / "typical_controls_v2"
    m = pd.read_csv(d / "manifest.csv", dtype={"speakerID": str})
    m = m[m["status"] == "ok"]
    return pd.DataFrame({"dataset": "controls_v2", "speaker_id": m["speakerID"].astype(str),
                         "record_id": m["control_id"], "path": [str(d / f) for f in m["file"]],
                         "gender": m.get("gender"), "transcript": m.get("transcript")})


def reference_clips(data_root: Path, *, transcribed_only: bool = False, salt: str = "exp2-reference") -> pd.DataFrame:
    d = data_root / "synth_source_v1_full"
    q = pd.read_csv(d / "quality.csv", dtype={"speakerID": str})
    m = pd.read_csv(d / "manifest.csv", dtype={"speakerID": str})
    keep = ["clip_id", "file"] + [c for c in ("transcript", "gender") if c in m]
    q = q[q["usable"] & (q["split"] == "train")].merge(m[keep], on="clip_id", how="left", suffixes=("", "_m"))
    if "gender_m" in q:
        q["gender"] = q["gender"].fillna(q["gender_m"])
    has_text = q.get("transcript", pd.Series(index=q.index, dtype=object)).fillna("").astype(str).str.strip() != ""
    if transcribed_only:
        sel = q[has_text]
    else:
        counts = q.groupby("speakerID").size()
        spk = sorted(counts[counts >= REFERENCE_PER_SPEAKER].index, key=lambda s: sa.stable_hash(salt, s))[:REFERENCE_MAX_SPEAKERS]
        sel = q[q["speakerID"].isin(spk)].copy()
        sel["_h"] = [sa.stable_hash(salt, c) for c in sel["clip_id"]]
        sel = sel.sort_values("_h").groupby("speakerID").head(REFERENCE_PER_SPEAKER)
    return pd.DataFrame({"dataset": "reference", "speaker_id": sel["speakerID"].astype(str), "record_id": sel["clip_id"],
                         "path": [str(d / f) for f in sel["file"]], "gender": sel.get("gender"),
                         "transcript": sel.get("transcript")})


def iter_atypical(hf_cache: Path | None) -> Iterator[dict[str, Any]]:
    from scripts.extract_features_exp1 import iter_atypical as it

    yield from it(hf_cache)


# --------------------------------------------------------------------------- phases


def _extract_rows(clips: pd.DataFrame, limit: int | None, transform=None, label: str = "measure") -> pd.DataFrame:
    rows, t0 = [], time.time()
    for k, (_, c) in enumerate(clips.iterrows()):
        if limit and k >= limit:
            break
        row = {"dataset": c["dataset"], "speaker_id": c["speaker_id"], "record_id": c["record_id"], "gender": c.get("gender")}
        try:
            x = read_audio(Path(c["path"]))
            if transform is not None:
                x = transform(x)
            row.update(ar.extract(x))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"[{label}] {k + 1}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def phase_measure(args, out: Path) -> dict[str, Any]:
    path = refuse_overwrite(out / "features_typical.csv", overwrite=args.overwrite)
    clips = pd.concat([controls_clips(args.data_root), reference_clips(args.data_root)], ignore_index=True)
    feats = _extract_rows(clips, args.limit)
    feats.to_csv(path, index=False)
    suff = sa.speaker_sufficiency(feats.dropna(subset=["gate_pass"]))
    suff.to_csv(out / "speaker_sufficiency_typical.csv", index=False)
    return {"recordings": len(feats), "errors": int(feats.get("error", pd.Series(dtype=object)).notna().sum()),
            "gate_pass_share": float(feats["gate_pass"].mean()) if "gate_pass" in feats else None,
            "speakers": int(feats["speaker_id"].nunique()), "formant_sufficient_speakers": int(suff["formant_sufficient"].sum())}


def phase_align(args, out: Path) -> dict[str, Any]:
    path = refuse_overwrite(out / "alignments.jsonl", overwrite=args.overwrite)
    prefs = ("torchaudio", "hf") if args.backend == "auto" else (args.backend,)
    name, backend = al.load_backend(prefs, device=args.device)
    clips = pd.concat([controls_clips(args.data_root), reference_clips(args.data_root, transcribed_only=True)], ignore_index=True)
    clips = clips[clips["transcript"].fillna("").astype(str).str.strip() != ""]
    token_rows, purity_rows, n = [], [], 0
    t0 = time.time()
    with path.open("w", encoding="utf-8") as fh:
        for k, (_, c) in enumerate(clips.iterrows()):
            if args.limit and k >= args.limit:
                break
            rec = {"utt_id": c["record_id"], "dataset": c["dataset"], "speaker_id": c["speaker_id"], "gender": c.get("gender")}
            try:
                x = read_audio(Path(c["path"]))
                res = al.align_transcript(backend, x, c["transcript"])
            except Exception as exc:
                x, res = None, {"status": "failed", "reason": f"read_error:{type(exc).__name__}"}
            rec.update({k2: v for k2, v in res.items() if k2 != "words"})   # no transcript text stored
            rec["word_spans"] = res.get("words", [])
            if x is not None and res.get("status") in ("aligned", "low_confidence"):
                feats, tr, sel, _ = ar.extract(x, return_tracks=True)
                rec["gate_pass"] = feats["gate_pass"]
                if tr is not None and res["status"] == "aligned" and feats["gate_pass"]:
                    for v in al.accepted_vowels(res):
                        f = sa.token_formants(tr, v["start_s"], v["end_s"])
                        if f:
                            token_rows.append({"utt_id": c["record_id"], "speaker_id": c["speaker_id"], "category": v["category"],
                                               "lf1": f[0], "lf2": f[1], "conf": v["conf"]})
                    inside, total = sa.frame_purity(tr.times[sel], [(v["start_s"], v["end_s"]) for v in res["vowels"]])
                    purity_rows.append({"utt_id": c["record_id"], "inside": inside, "total": total})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"[align] {n}/{len(clips)} ({time.time() - t0:.0f}s)", flush=True)
    tokens = pd.DataFrame(token_rows)
    tokens.to_csv(out / "labelled_tokens.csv", index=False)
    pur = pd.DataFrame(purity_rows)
    pur.to_csv(out / "frame_purity.csv", index=False)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    status = pd.Series([r["status"] for r in recs]).value_counts().to_dict()
    reasons = pd.Series([r.get("reason") for r in recs if r.get("reason")]).value_counts().to_dict()
    summary = {"backend": name, "utterances": n, "status_counts": status, "failure_reasons": reasons,
               "labelled_tokens": len(tokens), "seconds": round(time.time() - t0, 1)}
    write_json(out / "alignment_summary.json", summary)
    return summary


def phase_synthetic(args, out: Path) -> dict[str, Any]:
    path = refuse_overwrite(out / "synthetic_recovery.csv", overwrite=args.overwrite)
    res = sa.synthetic_recovery()
    res["table"].to_csv(path, index=False)
    summary = {k: v for k, v in res.items() if k != "table"}
    write_json(out / "synthetic_recovery.json", summary)
    return summary


def phase_channel(args, out: Path) -> dict[str, Any]:
    refuse_overwrite(out / "channel_summary.json", overwrite=args.overwrite)
    clips = controls_clips(args.data_root)
    conds = sa.channel_conditions()
    ran = {}
    for name, fn in conds.items():
        if fn is None:
            ran[name] = "not_run"
            continue
        _extract_rows(clips, args.limit, transform=fn, label=name).to_csv(out / f"channel_{name}.csv", index=False)
        ran[name] = "run"
    summary = {"conditions": ran, "ffmpeg_encoders_used": sorted(e for e in sa.ffmpeg_encoders() if e in ("libopus", "libopencore_amrnb"))}
    write_json(out / "channel_summary.json", summary)
    return summary


def phase_detectability(args, out: Path) -> dict[str, Any]:
    refuse_overwrite(out / "detectability.csv", overwrite=args.overwrite)
    try:
        import pyworld  # noqa: F401
    except ImportError:
        summary = {"status": "incomplete", "reason": "pyworld not installed"}
        write_json(out / "detectability.json", summary)
        return summary
    feats = pd.read_csv(out / "features_typical.csv", dtype={"speaker_id": str})
    clips = controls_clips(args.data_root).merge(feats[["record_id", "gate_pass"]], on="record_id")
    clips = clips[clips["gate_pass"] == True]  # noqa: E712
    clips = clips.assign(_h=[sa.stable_hash("exp2-detect", r) for r in clips["record_id"]]).sort_values("_h").head(DETECT_N_CLIPS)
    rows = []
    for _, c in clips.iterrows():
        x = read_audio(Path(c["path"]))
        base = ar.extract(x)
        rows.append({"record_id": c["record_id"], "kind": "none", "level": 0.0, **{m: base.get(m) for m in MEASURES}})
        for kind, levels in (("warp", DETECT_WARP), ("smooth", DETECT_SMOOTH_S)):
            for lv in levels:
                f = ar.extract(sa.envelope_perturbation(x, kind, lv))
                rows.append({"record_id": c["record_id"], "kind": kind, "level": lv, **{m: f.get(m) for m in MEASURES}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "detectability.csv", index=False)
    result = {}
    for m, kind in [(m, "warp") for m in sa.H1] + [(m, "smooth") for m in sa.H23]:
        vals = [table[table.kind == "none"][m].median()] + [table[(table.kind == kind) & (table.level == lv)][m].median()
                                                             for lv in (DETECT_WARP if kind == "warp" else DETECT_SMOOTH_S)]
        vals = [float(v) if pd.notna(v) else None for v in vals]
        result[m] = {"status": "pass" if sa.monotonic_decrease(vals) else "fail", "kind": kind, "medians": vals}
    summary = {"status": "run", "n_clips": int(len(clips)), "measures": result,
               "note": "Transform detectability only — not evidence of validity (§8.6)."}
    write_json(out / "detectability.json", summary)
    return summary


def phase_manual_sheet(args, out: Path) -> dict[str, Any]:
    folder = out / "manual_check"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already exists — manual sheets are never overwritten.")
    folder.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in (out / "alignments.jsonl").read_text(encoding="utf-8").splitlines()]
    al_df = pd.DataFrame([{"utt_id": r["utt_id"], "speaker_id": r["speaker_id"], "utterance_conf": r.get("utterance_conf", 0.0),
                           "status": r["status"], "categories": [v["category"] for v in r.get("vowels", [])]} for r in recs])
    sample = sa.select_manual_sample(al_df)
    by_utt = {r["utt_id"]: r for r in recs}
    alias = {s: f"T{i + 1}" for i, s in enumerate(sorted(sample["speaker_id"].unique()))}
    tokens = sa.select_manual_tokens({u: by_utt[u].get("vowels", []) for u in sample["utt_id"]}, list(sample["utt_id"]))
    clips = pd.concat([controls_clips(args.data_root), reference_clips(args.data_root, transcribed_only=True)]).set_index("record_id")
    import soundfile as sf

    for u in sample["utt_id"]:
        x = read_audio(Path(clips.at[u, "path"]))
        sf.write(str(folder / f"{u}.wav"), x, ar.SR, subtype="PCM_16")
        words, _ = al.clean_transcript(clips.at[u, "transcript"])
        rw = al.romanize_telugu(words or [])
        spans = by_utt[u].get("word_spans", [])
        word_tier = [(w["start_s"], w["end_s"], words[w["word_index"]] if words and w["word_index"] < len(words) else "")
                     for w in spans]
        vowel_tier = [(v["start_s"], v["end_s"], v["category"]) for v in by_utt[u].get("vowels", [])]
        al.write_textgrid(folder / f"{u}.TextGrid", x.size / ar.SR, {"words": word_tier, "vowels": vowel_tier})
    tok = pd.DataFrame([{"check_id": f"c{i:03d}", "utt_id": t["utt_id"], "speaker": alias[by_utt[t["utt_id"]]["speaker_id"]],
                         "token_index": t["token_index"], "category": t["category"], "auto_start_s": round(t["start_s"], 3),
                         "auto_end_s": round(t["end_s"], 3), "auto_conf": round(t["conf"], 3), "human_start_s": "",
                         "human_end_s": "", "vowel_present": "", "category_correct": "", "notes": ""} for i, t in enumerate(tokens)])
    tok.to_csv(folder / "tokens_sheet.csv", index=False)
    utt = pd.DataFrame({"utt_id": sample["utt_id"], "speaker": [alias[s] for s in sample["speaker_id"]],
                        "auto_utt_conf": sample["utterance_conf"].round(3), "low_conf_quartile": sample["low_conf_quartile"],
                        "selection_reason": sample["selection_reason"], "gross_misalignment": "", "notes": ""})
    utt.to_csv(folder / "utterances_sheet.csv", index=False)
    cover = sample.attrs.get("category_coverage", {})
    summary = {"utterances": len(sample), "speakers": int(sample["speaker_id"].nunique()), "tokens": len(tok),
               "category_coverage_utterances": cover, "low_confidence_utterances": int(sample["low_conf_quartile"].sum())}
    write_json(folder / "sample_summary.json", summary)
    return summary


def phase_manual_summary(args, out: Path) -> dict[str, Any]:
    folder = out / "manual_check"
    path = refuse_overwrite(out / "manual_summary.json", overwrite=args.overwrite)
    tok = pd.read_csv(folder / "tokens_sheet.csv", dtype=str, keep_default_na=False)
    for c in ("auto_start_s", "auto_end_s", "auto_conf"):
        tok[c] = pd.to_numeric(tok[c], errors="coerce")
    utt = pd.read_csv(folder / "utterances_sheet.csv", dtype=str, keep_default_na=False)
    summary = sa.summarise_manual(tok, utt)
    write_json(path, summary)
    return summary


def phase_lexical(args, out: Path) -> dict[str, Any]:
    path = refuse_overwrite(out / "lexical.json", overwrite=args.overwrite)
    recs = [json.loads(l) for l in (out / "alignments.jsonl").read_text(encoding="utf-8").splitlines()]
    aligned = {r["utt_id"] for r in recs if r["status"] == "aligned"}
    clips = pd.concat([controls_clips(args.data_root), reference_clips(args.data_root, transcribed_only=True)])
    ctrl = {c: 0 for c in al.VOWEL_CATEGORIES}
    for t in clips[clips["record_id"].isin(aligned)]["transcript"]:
        for k, v in al.vowel_category_counts(t).items():
            ctrl[k] += v
    from intelli_speech.vaani import load_atypical_telugu

    ds = load_atypical_telugu(args.hf_cache)
    atyp = {c: 0 for c in al.VOWEL_CATEGORIES}
    for t in ds["transcription"]:
        for k, v in al.vowel_category_counts(t).items():
            atyp[k] += v
    res = sa.lexical_flag(ctrl, atyp)
    res.update({"control_counts": ctrl, "atypical_counts": atyp, "note": "Text only; atypical transcripts are intended meaning."})
    write_json(path, res)
    return res


def phase_atypical_feasibility(args, out: Path) -> dict[str, Any]:
    """Sealed: only ICCs and counts are written — no per-recording or per-speaker values."""
    path = refuse_overwrite(out / "atypical_feasibility.json", overwrite=args.overwrite)
    from intelli_speech.vaani_controls import to_mono_16k

    rows = []
    for k, item in enumerate(iter_atypical(args.hf_cache)):
        if args.limit and k >= args.limit:
            break
        try:
            f = ar.extract(to_mono_16k(item["audio"], item["rate"]))
        except Exception:
            f = {}
        rows.append({"speaker_id": item["speaker_id"], "record_id": item["record_id"], "gate_pass": f.get("gate_pass", False),
                     "sel_n_selected": f.get("sel_n_selected", 0), **{m: f.get(m) for m in MEASURES}})
    df = pd.DataFrame(rows)
    suff = sa.speaker_sufficiency(df)
    result = {
        "note": "Feasibility only (§13.1). Sealed: no atypical feature values are stored.",
        "recordings": int(len(df)), "speakers": int(df["speaker_id"].nunique()),
        "gate_pass_recordings": int(df["gate_pass"].sum()),
        "speakers_formant_sufficient": int(suff["formant_sufficient"].sum()),
        "per_speaker_counts": [{"speaker": f"S{i + 1}", "n_clips": int(r.n_clips), "n_gate_clips": int(r.n_gate_clips),
                                "n_selected_frames": int(r.n_selected_frames)}
                               for i, r in enumerate(suff.sort_values("speaker_id").itertuples())],
        "split_half": {m: sa.split_half(df, m) for m in MEASURES},
    }
    del df
    write_json(path, result)
    return result


def phase_report(args, out: Path) -> dict[str, Any]:
    path = refuse_overwrite(out / "stage_a_report.json", overwrite=args.overwrite)
    load = lambda name: json.loads((out / name).read_text(encoding="utf-8")) if (out / name).is_file() else None  # noqa: E731
    feats = pd.read_csv(out / "features_typical.csv", dtype={"speaker_id": str})
    results: dict[str, dict[str, Any]] = {t: {} for t in ("synthetic", "validity", "purity", "reliability", "channel", "detectability")}
    syn = load("synthetic_recovery.json")
    for m in sa.H1 + sa.H23:
        results["synthetic"][m] = {"status": ("pass" if syn["pass"] else "fail") if syn else "incomplete",
                                   **({k: syn[k] for k in ("f1_median_rel_err", "f2_median_rel_err")} if syn else {})}
    manual = load("manual_summary.json")
    tokens_path = out / "labelled_tokens.csv"
    if tokens_path.is_file() and tokens_path.stat().st_size > 10:
        labelled = sa.labelled_speaker_measures(pd.read_csv(tokens_path, dtype={"speaker_id": str}))
        labelled_long = sa.labelled_speaker_measures(pd.read_csv(tokens_path, dtype={"speaker_id": str}), corner=al.CORNER_LONG)
    else:
        labelled = labelled_long = None
    gate_feats = feats[feats["gate_pass"] == True]  # noqa: E712
    for m in sa.H1:
        if labelled is None:
            results["validity"][m] = {"status": "incomplete", "reason": "no alignment"}
            continue
        v = sa.validity_gate(gate_feats, labelled, m)
        v["long_vowel_sensitivity"] = sa.validity_gate(gate_feats, labelled_long, m)
        if manual is None or manual.get("acceptance") == "incomplete":
            v = {**v, "status": "incomplete", "reason": "manual alignment check not completed"}
        elif manual.get("acceptance") == "fail":
            v = {**v, "status": "fail", "reason": "manual alignment acceptance failed (§13.3)"}
        results["validity"][m] = v
    pur_path = out / "frame_purity.csv"
    if pur_path.is_file() and pur_path.stat().st_size > 10:
        pur = pd.read_csv(pur_path)
        share = float(pur["inside"].sum() / pur["total"].sum()) if pur["total"].sum() else None
        st = "pass" if share is not None and share >= sa.PURITY_MIN else ("fail" if share is not None else "incomplete")
    else:
        share, st = None, "incomplete"
    for m in sa.H1 + sa.H23:
        results["purity"][m] = {"status": st, "share": share}
    for m in MEASURES:
        ref = sa.split_half(feats[feats["dataset"] == "reference"], m)
        ctl = sa.split_half(feats[feats["dataset"] == "controls_v2"], m)
        icc = ref["icc21"]
        results["reliability"][m] = {"status": "incomplete" if icc is None else ("pass" if icc >= sa.ICC_MIN else "fail"),
                                     "reference": ref, "controls_v2_reported": ctl}
    ch = load("channel_summary.json")
    orig = feats[feats["dataset"] == "controls_v2"]
    for m in MEASURES:
        if ch is None:
            results["channel"][m] = {"status": "incomplete"}
            continue
        frames = {name: (pd.read_csv(out / f"channel_{name}.csv", dtype={"speaker_id": str}) if state == "run" else None)
                  for name, state in ch["conditions"].items()}
        results["channel"][m] = sa.channel_gate(orig, frames, m)
    det = load("detectability.json")
    for m in sa.H1 + sa.H23:
        results["detectability"][m] = (det["measures"][m] if det and det.get("status") == "run"
                                       else {"status": "incomplete", "reason": (det or {}).get("reason", "not run")})
    report = {
        "created_utc": utc_now(), "plan": "EXP2_PLAN.md v0.2.1", "feature_version": ar.FEATURE_VERSION,
        "scope": "Measurement validity on typical Telugu speech only. No group comparison; no diagnosis.",
        "measures": {m: sa.measure_status(results, m) for m in MEASURES},
        "manual_alignment": manual, "alignment": load("alignment_summary.json"),
        "lexical_flag": load("lexical.json"), "atypical_feasibility": load("atypical_feasibility.json"),
        "selection_diagnostics_by_dataset": feats.groupby("dataset")[[c for c in ("nuclei_per_s", "selected_frames_per_s", "gate_pass")
                                                                      if c in feats]].mean().round(4).to_dict(),
    }
    write_json(path, report)
    return {m: report["measures"][m]["overall"] for m in MEASURES}


PHASES = {"measure": phase_measure, "align": phase_align, "synthetic": phase_synthetic, "channel": phase_channel,
          "detectability": phase_detectability, "manual-sheet": phase_manual_sheet, "manual-summary": phase_manual_summary,
          "lexical": phase_lexical, "atypical-feasibility": phase_atypical_feasibility, "report": phase_report}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=tuple(PHASES), required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--hf-cache", type=Path, default=None)
    p.add_argument("--backend", choices=("auto", "torchaudio", "hf"), default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: first N clips per phase.")
    p.add_argument("--overwrite", action="store_true", help="Derived outputs only; manual sheets are never overwritten.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output_dir or args.data_root / "analysis_v2" / "exp2_stage_a"
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = PHASES[args.phase](args, out)
    except (FileExistsError, FileNotFoundError, al.AlignmentError) as exc:
        print(f"Stage A [{args.phase}] stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
