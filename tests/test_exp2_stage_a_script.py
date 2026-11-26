from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

pytest.importorskip("parselmouth")

import scripts.exp2_stage_a as runner
from intelli_speech import alignment as al
from intelli_speech import exp2_stage_a as sa
from tests.test_articulation_features import _vowel_sentence

SR = 16000
TEXTS = ["కాలు ఈగ ఊరు కాకి", "ఇల్లు ఉడుత అది కోతి", "చెట్టు <noise> ఏడు నీరు", "అది 5 కాయలు"]


def _fake_align(backend, audio, text):
    """Deterministic stand-in for MMS: vowels spread evenly over the clip."""
    words, reason = al.clean_transcript(text)
    if words is None:
        return {"status": "excluded", "reason": reason}
    cats = [u.category for w in al.romanize_telugu(words) for u in w.units if u.kind == "vowel" and u.category]
    dur = audio.size / SR
    step = dur / (len(cats) + 1)
    vowels = [{"word_index": 0, "unit_index": i, "category": c, "start_s": (i + 0.6) * step, "end_s": (i + 1.2) * step,
               "conf": 0.9} for i, c in enumerate(cats)]
    return {"status": "aligned", "utterance_conf": 0.8, "n_tokens": 10, "vowels": vowels,
            "words": [{"word_index": 0, "start_s": 0.1, "end_s": dur - 0.1, "conf": 0.9}]}


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    ctl = root / "typical_controls_v2"
    rows = []
    for s in range(3):
        for c in range(3):
            f = f"audio/spk{s}/ctl_{len(rows):04d}.wav"
            (ctl / f).parent.mkdir(parents=True, exist_ok=True)
            sf.write(ctl / f, _vowel_sentence(seed=10 * s + c), SR, subtype="PCM_16")
            rows.append({"speakerID": f"C{s}", "gender": ["Female", "Male"][s % 2], "status": "ok",
                         "control_id": f"ctl_{len(rows):04d}", "file": f, "transcript": TEXTS[(s + c) % len(TEXTS)]})
    pd.DataFrame(rows).to_csv(ctl / "manifest.csv", index=False)
    src = root / "synth_source_v1_full"
    q, m = [], []
    for s in range(6):
        for c in range(4):
            cid = f"src_{len(q):04d}"
            f = f"audio/R{s}/{cid}.wav"
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            sf.write(src / f, _vowel_sentence(seed=100 + 10 * s + c), SR, subtype="PCM_16")
            q.append({"clip_id": cid, "speakerID": f"R{s}", "gender": "Female", "usable": True, "split": "train"})
            m.append({"clip_id": cid, "speakerID": f"R{s}", "file": f, "transcript": TEXTS[c % 3] if c < 2 else ""})
    q.append({"clip_id": "src_val", "speakerID": "RV", "gender": "Male", "usable": True, "split": "val"})
    m.append({"clip_id": "src_val", "speakerID": "RV", "file": "missing.wav", "transcript": TEXTS[0]})
    pd.DataFrame(q).to_csv(src / "quality.csv", index=False)
    pd.DataFrame(m).to_csv(src / "manifest.csv", index=False)
    return root


def _run(root: Path, phase: str, *extra: str) -> int:
    return runner.main(["--phase", phase, "--data-root", str(root), *extra])


def test_stage_a_runner_end_to_end(data_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(sa, "synthetic_grid", lambda: [(300.0, 2300.0, 120.0), (700.0, 1200.0, 150.0)])
    monkeypatch.setattr(sa, "ffmpeg_encoders", lambda: set())
    monkeypatch.setattr(al, "load_backend", lambda prefs, device="cpu": ("fake", None))
    monkeypatch.setattr(al, "align_transcript", _fake_align)

    def fake_atypical(cache):
        for k in range(6):
            yield {"speaker_id": f"A{k % 2}", "record_id": f"a{k}", "audio": _vowel_sentence(seed=500 + k)[:, None], "rate": SR}
    monkeypatch.setattr(runner, "iter_atypical", fake_atypical)
    import intelli_speech.vaani as vaani
    monkeypatch.setattr(vaani, "load_atypical_telugu", lambda cache: {"transcription": ["కాలు కాకి అది", "ఉడుత"]})

    out = data_root / "analysis_v2" / "exp2_stage_a"
    assert _run(data_root, "measure") == 0
    feats = pd.read_csv(out / "features_typical.csv")
    assert len(feats) == 9 + 18 and set(feats["dataset"]) == {"controls_v2", "reference"}   # val split excluded
    assert _run(data_root, "measure") == 1                                   # refuses to overwrite
    assert _run(data_root, "align") == 0
    recs = [json.loads(l) for l in (out / "alignments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["status"] for r in recs} >= {"aligned", "excluded"}
    assert all("transcript" not in r for r in recs)                         # no transcript text stored
    assert _run(data_root, "synthetic") == 0
    assert _run(data_root, "channel") == 0
    ch = json.loads((out / "channel_summary.json").read_text())
    assert ch["conditions"]["opus_8k"] == "not_run" and ch["conditions"]["lp3400"] == "run"
    assert _run(data_root, "detectability") == 0
    assert _run(data_root, "manual-sheet") == 0
    sheet = pd.read_csv(out / "manual_check" / "tokens_sheet.csv")
    assert len(sheet) > 0 and set(sheet["speaker"]).issubset({f"T{i}" for i in range(1, 20)})   # pseudonyms only
    assert (out / "manual_check" / f"{sheet['utt_id'].iloc[0]}.TextGrid").is_file()
    assert _run(data_root, "manual-sheet") == 1                              # never overwritten
    sheet["human_start_s"], sheet["human_end_s"] = "=", "="
    sheet["vowel_present"], sheet["category_correct"] = "y", "y"
    sheet.to_csv(out / "manual_check" / "tokens_sheet.csv", index=False)
    utt = pd.read_csv(out / "manual_check" / "utterances_sheet.csv")
    utt["gross_misalignment"] = "n"
    utt.to_csv(out / "manual_check" / "utterances_sheet.csv", index=False)
    assert _run(data_root, "manual-summary") == 0
    assert json.loads((out / "manual_summary.json").read_text())["acceptance"] == "pass"
    assert _run(data_root, "lexical") == 0
    assert _run(data_root, "atypical-feasibility") == 0
    feas = json.loads((out / "atypical_feasibility.json").read_text())
    text = json.dumps(feas)
    assert "A0" not in text and "aavs_log\": 0" not in text                # sealed: pseudonyms, ICCs/counts only
    assert set(feas["split_half"]["aavs_log"]) == {"n_speakers", "icc21"}
    assert _run(data_root, "report") == 0
    report = json.loads((out / "stage_a_report.json").read_text())
    for m, entry in report["measures"].items():
        assert entry["overall"] in {"pass", "fail", "incomplete"}
        assert set(entry["tests"]) == set(sa.APPLICABLE[m])
    assert report["measures"]["aavs_log"]["tests"]["channel"]["status"] in {"incomplete", "fail"}   # codecs not run
    assert report["measures"]["aavs_log"]["overall"] != "pass"


def test_runner_reports_missing_inputs(tmp_path: Path) -> None:
    assert runner.main(["--phase", "manual-summary", "--data-root", str(tmp_path)]) == 1   # sheets missing → clean stop
    with pytest.raises(FileNotFoundError):
        runner.phase_lexical(runner.build_parser().parse_args(["--phase", "lexical", "--data-root", str(tmp_path)]), tmp_path)
