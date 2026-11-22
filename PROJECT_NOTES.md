# Intelli-Speech — Project Notes

**Last updated:** 2026-11-22
**Purpose:** Single source of truth for verified state, blockers, decisions, and the next
task. Update it only with observed evidence.

## 0. v2 master progress tracker (atypical Telugu speech)

**Research direction:** see `project_brief_spt.md`.
- Telugu-only Vaani Atypical corpus.
- Evidence-based synthetic data.
- Modular pipeline: a DSP temporal module, a prosody module, and HDNZF-Net (GMM + ZF-Net + DNN) as the frame-aligned spectral-clarity core. ASR→TTS is a comparison only.
- Environmental-noise suppression (v1) stays a separate model.
- Leave-one-speaker-out throughout, with transform parameters fitted per fold.
- Status labels: **implemented** (code exists), **tested** (synthetic unit tests), **run** (executed on real data), **validated** (scientifically evaluated).

| Phase / item | Status | Evidence / location |
|---|---|---|
| **1 Dataset readiness** | | |
| Exp 0 audit (`scripts/audit_v2.py`, `AUDIT_V2.ipynb`) | **run** 2026-11-22 | Drive `analysis_v2/audit_2026-11-22/` |
| Exp 0b follow-up (clipping, transcription scripts, difficulty) | **run** 2026-11-22 | Drive `analysis_v2/audit_followup_2026-11-22/` |
| Balanced typical controls (≤40 spk × ≤10 clips, ≤400) | **run** 2026-11-22: 40 spk (20 F / 20 M), 126 clips, 12.4 min | Drive `typical_controls_v2/` (manifest + provenance, rev `9e6ed4f`) |
| Speaker-independent folds | not started | — |
| **2 Characterization (Exp 1–2)** | **run** 2026-11-22 (400 recordings, 0 errors): 17 candidate_class1 (all temporal + energy-modulation rate), 17 speaker_specific, 14 not_supported, 2 channel_confounded; robustness **run** 2026-11-22: **14 robust class-1 features** (see below) | features: `intelli_speech/acoustic_features.py` (`exp1-features-0.1`); `scripts/extract_features_exp1.py`; `scripts/analyze_exp1.py` (speaker-level catalogue); `EXP1_CHARACTERIZATION.ipynb` |
| **3 Synthetic + models** | not started | — |
| **4 Integration + evaluation** | not started | — |
| **5 Prototype + delivery** | v1 app plumbing only (noise model) | `server/`, `app/` |

**Exp 0 verified facts (2026-11-22, real data):**
- **Atypical Telugu:** 189 clips, 7 speakers, 109.05 min, 16 kHz mono, 0 decode errors, 0 duplicates.
  - Clips per speaker: 45 / 45 / 40 / 40 / 9 / 5 / 5.
  - All 189 files reach ≥0.999 of full scale (whether this is clipping or normalisation is unresolved).
  - Effective bandwidth per speaker is 1.2–5.0 kHz, which points to channel differences between speakers.
  - Difficulty level: Low 55 / Medium 133 / High 1. Cramér's V with speaker = 0.51. **Scale direction unknown**: the card only says "from QC understandability rating".
  - The CER between verbatim and intended transcriptions averages 1.13, so the two fields' semantics and scripts are unresolved.
  - All 189 prompts are distinct; none is shared between speakers.
- **Typical Vaani Drive copy:** 246 clips, 41 speakers, 25.9 min. Median 1 clip per speaker; one speaker has 161. No prompt column.
- **SLR66:** 4,448 recordings, 47 speaker IDs derived from filenames, 48 kHz, read speech.
- **Regular Vaani (HF card):** `ARTPARK-IISc/Vaani`, gated, CC-BY-4.0, parquet, Telugu ≈ 2,326 h (≈184 h transcribed), image-prompted spontaneous speech, sample rate not stated.

**Exp 0b follow-up facts (2026-11-22, real data):**
- **A1 full scale:** atypical 172/189 consistent with peak normalisation, 17 flagged clipping evidence (S4 8/9, S1 4, S7 3, S5 2); max run of near-full-scale samples 5, near-full-scale fraction ≤ 1e-4 → clipping is sparse. Typical copy: 155/246 below full scale, 91 peak-normalised, none clipped. → Level (RMS/peak) differs by *processing*, so level features are channel descriptors only; voice-quality features get a sensitivity check excluding flagged files.
- **A2 scripts:** intended transcriptions are Telugu script (189/189; 3 stray Kannada signs). Verbatim transcriptions are mixed: 106 Telugu-dominant, 83 Latin-dominant (romanised), with annotation tags (`<>`, `{}`, `[]`). Typical copy: 197 Telugu / 49 Latin, same tags. → The verbatim-vs-intended CER (mean 1.13) mostly measures script/markup mismatch and is **not** an intelligibility measure; transcript-based akshara rates are not usable for verbatim text. Speaking rate stays acoustic.
- **A3 difficulty:** card says only "from QC understandability rating"; reviewers are "not specialists in atypical speech"; intended text is written by people familiar with the speaker. Direction remains **UNKNOWN**. The only within-speaker association (Medium < Low on CER, p = 0.012, 5 speakers) is confounded by script mixing. → difficulty_level is a descriptive covariate only, never a severity label or target.
- **Controls v2:** 30/872 shards scanned; 3,334 transcribed rows, 47 eligible speakers, 40 selected; 3–5 clips per speaker (cap 10 not reached: few transcribed clips per speaker); 16 kHz native, 0 decode errors, 0 duplicates. Districts are skewed (Manyam 14, Krishna 7 speakers). Control clips are short (~6 s) vs atypical (~35 s) → features are rate/proportion based; duration is reported but excluded.

**Exp 1 confirmed pattern catalogue (2026-11-22):** class 1 in the main analysis, against the second control set (`vaani_copy`) and with atypical audio cut into 5.8 s windows (1,126 windows):
- **Rate:** speech rate and articulation rate (acoustic syllable nuclei) lower, 7/7 speakers in all three.
- **Pauses:** more pauses (≥250 ms, ≥500 ms per minute), longer pauses (P90 at all thresholds, median ≥500 ms), more variable pauses (IQR ≥150/250 ms), larger pause-time fraction (all thresholds).
- **Activity:** lower speech-activity ratio.
- **Not robust:** energy-modulation peak (6/7 main, 2/7 vs vaani_copy); median pause ≥150/250 ms (fails vs vaani_copy).
- **Speaker-specific / not shared:** F0 SD and range (1/7), voiced fraction, rhythm CV/nPVI, voice quality (HNR, CPPS, jitter, shimmer), MFCC/spectral shape.
- **Limits:** both control sets are regular Vaani (same collection pipeline); atypical clips peak-normalised; 7 speakers (two with 5 clips); syllable counts are acoustic estimates. Not clinical markers.
- **Implication for Phase 3:** synthetic transforms are primarily temporal (articulation-rate slowing, pause insertion/lengthening, distribution matched per fold); pitch changes speaker-specific (random per synthetic speaker); no shared spectral target yet.

**Decision 2026-11-22: component roles (approved by the user).** The two-stage system is kept: Stage 1 is the v1 noise model (SLR66 + DEMAND + `paired_dataset_v3`), and Stage 2 is atypical-speech processing. Exp 1 showed the shared atypical pattern is temporal, while spectral, pitch and voice-quality differences are speaker-specific. A frame-aligned network cannot change timing. So, inside HDNZF-Net:
- **GMM (identification):** typical-vs-atypical scoring and degree, from the timing features plus frame features. Its score also sets the correction strength.
- **ZF-Net (identification and reconstruction):** frame-level localisation of atypical events (abnormal pauses, slowed stretches) and residual spectral clean-up for speakers who need it.
- **DNN:** fuses the GMM and ZF-Net outputs and predicts the correction parameters (pause shortening, rate change).
- **DSP temporal module:** executes the timing edits.

Each component's contribution is shown by ablation under leave-one-speaker-out evaluation, and the detector results are reported as preliminary (7 real speakers). Vaani source clips are **not** pre-cleaned with the v1 model, to avoid baking in artefacts; noisy clips are rejected by a quality filter instead.

**Synthetic source (2026-11-22):** regular Vaani Telugu, spontaneous speech.
- **Speaker overlap:** speakers in `typical_controls_v2` and `vaani_telugu_transcribed` are excluded.
- **Selection:** at most 20 clips per speaker, 2–30 s, gender-balanced, transcripts not required (transcribed clips are flagged), about 30% oversampling for the quality filter.
- **Split:** by speaker, 80/10/10.
- **Pilot:** about 60 speakers, about 1,000 clips that pass the filter, in `synth_source_v1_pilot/`.
- **Full run (after the pilot is reviewed):** about 300 speakers, about 6,000 clips, about 10 h.

**Synthetic-source pilot, run 1 (2026-11-22): failed as a selection, kept as a record.**
- **What happened:**
  - 30 shards were scanned (49,297 rows, 1,865 speaker IDs; 2,207 rows removed as control speakers), but only 1 "speaker" was eligible.
  - That one was a placeholder ID covering 5 districts (it is read back as NaN, which is why the download reported 0 speakers). Only 10 clips were downloaded, to `synth_source_v1_pilot/`; ignore that folder.
- **Quality on those 10 clips:**
  - The 99 %-energy bandwidth median is 1.6 kHz, so the 3 kHz floor was wrong for Vaani. The floor is now 1.2 kHz, the lowest atypical speaker on the same measure.
  - The SNR-estimate median is 15.6 dB; the threshold moved from 15 to 12 dB, to be reviewed on run 2's quantiles.
- **Fixes:**
  - Placeholder speaker and utterance IDs are filtered.
  - Rows without an utterance ID are deduplicated by parquet position, not collapsed together.
  - Shards are scanned in runs of 3 consecutive shards.
  - The speaker count is now a ceiling (200), the minimum is 5 clips per speaker, and the clip total binds.
  - The notebook's cell-5 syntax error is fixed.
- **Run 2** writes to `synth_source_v1_pilot_r2/`.

**Technical decisions (2026-11-22):**
- **VAD:** an adaptive energy VAD (frames more than 35 dB below the 95th percentile, or within 6 dB of the floor, count as non-speech). It is deterministic and needs no download. It is less robust in noisy recordings, so a neural VAD (Silero) is a possible later upgrade.
- **Speaking rate without transcripts:** acoustic syllable nuclei, de Jong & Wempe style (voiced intensity peaks with a dip of at least 2 dB). Articulation rate = nuclei / (active time − pauses ≥ 250 ms). The transcript-based akshara rate stays blocked until A2 resolves which script each field uses.
- **F0:** two passes. A first pass at 60–600 Hz, then a second pass bounded by 0.75 × the 15th percentile and 1.5 × the 85th percentile. Variability is reported in semitones relative to the recording's median.
- **CPPS:** Praat with 10 ms frames and a straight-trend "Robust" fit. It is about 100× faster than the "Robust slow" default (223 s → 0.3 s per 10 s clip) and within about 0.3 dB on test signals.
- **Spectral features:** band-limited to 80–4000 Hz, with per-recording MFCC mean removal. A feature is flagged `channel_confounded` when |Spearman ρ| ≥ 0.5 with bandwidth or RMS.
- **Catalogue criterion (class 1):** all of the following:
  - at least 5 of 7 speakers outside the control 10th–90th percentile range, in the same direction;
  - the sign holds when each atypical speaker is left out;
  - Benjamini–Hochberg q < 0.10;
  - |Cliff's δ| ≥ 0.474;
  - not channel-confounded.
- **opencode:** the deepseek, nemotron-super and mimo free models hit their usage limits on 2026-11-22. Only `nemotron-3.5-lightning-free` is both available and listed in `opencode.json`. It was used for one code review: 9 of 11 findings were false positives, and one robustness fix was applied.

**Open blockers:**
1. (resolved) Follow-up code pushed in `1563dae`; Vaani access accepted; follow-up run.
2. Optional: controls with more clips per speaker would need un-transcribed rows (transcripts are not used by Exp 1); not needed for speaker-level medians.
3. Notebooks with run outputs (`AUDIT_V2*.ipynb`, and the `TRAIN_V1`, `MAINFILE` and `RECOVERY` output changes) must not be committed with outputs: they contain research-only contributor IDs.

## Notebook and archive map

| File | Role | Writes |
|---|---|---|
| `RECOVERY.ipynb` | Runtime, Drive/repo, cache, Hugging Face, checkpoint, ClearerVoice recovery | cache directories only |
| `TRAIN_V1.ipynb` | v1 runs: baselines, GMM, ZF-Net training, evaluation, results table | `experiments/` only |
| `MAINFILE.ipynb` | Read-only workflow over existing Drive artifacts | none |
| `docs/dataset_structure.md` | Dataset structure and audit record | — |
| `docs/progress.md` | Dataset-generation progress narrative (moved from root) | — |
| `docs/project_plan.md` | Research plan and operating rules (moved from root) | — |
| `prev_files/` | Untracked archive of pre-cleanup notebooks; never imported by code | none |

Run `RECOVERY.ipynb` first in a fresh Colab runtime, then `MAINFILE.ipynb`.

## 1. Verified progress

### 1.1 Paired dataset v3 — verified

- 500 clean targets + 1,000 noisy mixtures (5 dB and 10 dB), 16 kHz mono, 1,500 WAV files.
- 80/10/10 source split: 400 / 50 / 50, no candidate speaker overlap across splits.
- 1,000 metadata rows; 200 mixtures per DEMAND category (5 categories).
- Independent read-only audit: **21/21 checks passed**; maximum absolute SNR error
  0.0003905 dB; no full-scale clipping.
- Source manifest SHA-256 recorded as
  `2af46ef5c9a4545c0775f08a07b734bc0696a888339a18d5e40cdd1103941eb0`.
- Evidence: `docs/dataset_structure.md`, `docs/progress.md`. The audit itself was **not**
  re-run in this session; the numbers above are the recorded audit results.

### 1.2 STFT analysis state — recorded, currently blocked in Colab

- `librosa.stft` with `n_fft=512`, `hop_length=128` at 16 kHz.
- A saved analysis checkpoint exists at
  `Intelli-Speech-data/analysis_cache/stft_example.npz` (clean/noisy waveforms, their
  STFTs, sample rate, FFT and hop settings), written and re-loaded in the historical
  notebook session.
- Recorded checkpoint contents: `clean_stft=(257, 510)`, `noisy_stft=(257, 510)`, all
  values finite, `sample_rate=16000`, `n_fft=512`, `hop_length=128`.
- The latest recorded `MAINFILE.ipynb` run fails at step 4 with
  `ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.` The checkpoint itself is
  present; the `librosa` import path is what fails. See blocker 1.

### 1.3 Baselines — ported into active code; metrics pending

- Oracle IRM computed from clean and estimated noise magnitudes:
  `mask = |clean| / (|clean| + |noise| + 1e-10)`.
- Recorded oracle IRM result on one sample: input SNR 5.000 dB, output 28.595 dB,
  SNR change 23.595 dB. The result file was saved to
  `Intelli-Speech-data/analysis_cache/irm_baseline_results.json`. This is oracle-only and
  explicitly not deployable.
- Non-oracle Wiener-style baseline implemented using only the noisy input, with noise
  power estimated from the first 0.25 s. Recorded gain range `[0, ~1]`, mean `0.1727`;
  no objective metric was computed.
- A 12-sample listening comparison (clean reference / noisy input / Wiener-style output)
  was executed and its audio players are stored in the archived notebook output.
- The baseline logic is now ported into `intelli_speech/baselines.py`,
  `intelli_speech/evaluation.py`, and `scripts/evaluate_baselines.py`. The full test suite
  passes in an isolated venv with librosa 0.10.2.post1; system Python skips only the
  librosa-dependent cases because librosa is not installed there.
- **No objective metrics have been recorded yet.** The first metrics table still requires
  a working Colab/NumPy runtime. PESQ and STOI are optional wrappers and are not run by
  default.

### 1.4 FRCRN — model initialized once, no inference result

- Recorded preflight in the archive: Python 3.13.18, NumPy 2.4.6, PyTorch 2.14.1+cpu,
  torchaudio 2.11.0+cpu, CUDA unavailable, ClearerVoice not installed at that time.
- In a later recorded session, `clearvoice` 0.1.2 was importable and
  `ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])` succeeded:
  `downloading checkpoint for FRCRN_SE_16K` followed by
  `FRCRN model initialized successfully!`.
- **No enhanced audio was produced.** The single-sample enhancement cell has no output.
- The latest `RECOVERY.ipynb` run found **no FRCRN checkpoint** in any inspected location
  (`checkpoints/FRCRN_SE_16K`, `persistent_cache/clearvoice_runtime`, `~/.cache/clearvoice`,
  Hugging Face caches). The downloaded checkpoint did not persist across the runtime reset.
- Conclusion: FRCRN is not blocked at import time in the recorded runtime, but checkpoint
  persistence and inference remain unverified.

### 1.5 Repository utilities — verified in this session

- `intelli_speech/audio_io.py`, `intelli_speech/metrics.py`,
  `intelli_speech/dataset.py`, `intelli_speech/preprocessing.py`,
  `intelli_speech/stft.py`, `intelli_speech/baselines.py`,
  `intelli_speech/evaluation.py`, `intelli_speech/frcrn.py`,
  `scripts/evaluate_baselines.py`, and `scripts/run_frcrn.py` exist and import cleanly.
- System Python: `python -m pytest -p no:cacheprovider -ra` → **73 passed, 6 skipped**;
  the skips are librosa-dependent tests, and the system environment has no librosa.
- Temporary isolated venv with `librosa==0.10.2.post1` and NumPy 2.3.1:
  `python -m pytest -p no:cacheprovider -ra` → **79 passed**. This includes the STFT
  round-trip, Wiener/IRM, silence-trimming, FRCRN adapter, and CLI smoke tests.
- `npx tsc --noEmit -p tsconfig.app.json` in `app/` → **clean** (frontend is out of scope
  for the current research work).

### 1.6 v1 model (ZF-Net + GMM) and app wiring — implemented, not yet trained on real data

Decision (2026-11-22): the GMM's role is a **noise-type classifier**. Its 5-class
posterior conditions the ZF-Net mask estimator through FiLM and is shown in the UI.

- Code: `intelli_speech/{features,gmm_noise,zfnet,enhancer,experiment}.py`,
  `scripts/train_gmm.py`, `scripts/train_zfnet.py`, `--method zfnet-gmm` in
  `scripts/evaluate_baselines.py`, the FastAPI server in `server/`, and the frontend in
  `app/src/enhance/` (16 kHz WAV upload when `VITE_ENHANCE_API_URL` is set; demo mode is
  kept otherwise).
- Local venv (torch 2.15.0+cpu, librosa 0.10.2.post1, NumPy 2.4.6, sklearn 1.9.1):
  `pytest` → **108 passed**. `npm run build` passes with and without the API URL.
- A synthetic 36-row smoke dataset ran end to end: train_gmm → train_zfnet (3 epochs) →
  evaluate zfnet-gmm → `POST /enhance` returned audio plus a noise label. **These are smoke
  numbers only, not research results.**
- PESQ cannot be built on this Windows machine (no MSVC). Compute it in Colab.
- Not yet done: training on `paired_dataset_v3` and the real metrics table.

## 2. Archive map and cleanup result

The cleanup did **not** delete unique research content. It moved the working notebook out
of the active path and reduced the active repo to documentation, orchestration, and
generic I/O. The research logic is preserved in the archives below.

| Archive | Cells | Code lines | Functions | Unique content | SHA-256 |
|---|---:|---:|---:|---|---|
| `prev_files/Copy_of_Copy_of_code (1).ipynb` | 152 | 4,982 | 28 | IRM apply/reconstruct, SNR, Wiener baseline, 12-sample listening, FRCRN install/init success | `76d30e1c643c323d984acbc3c2f7b1dd721e92c2f393c70c50b4ddc933350ad3` |
| `prev_files/Copy_of_code.ipynb` | 121 | 4,129 | 23 | Dataset v1/v2/v3 generation, manifest, audit, STFT, early IRM | `d8f7ede7619cb763e6298cdd620b9ac56611f3ff28303c2f893cc3c5e53eb718` |
| `prev_files/MAINFILE_monolithic_2026-11-21.ipynb` | 155 | 5,369 | 33 | FRCRN preflight, API/checkpoint inspection, single-sample FRCRN drafts, manifest path updates | `fafff8829b2452bd7369d6a8f2f51b3b0b03e7ba34c065a6439b2404d104fc58` |
| `prev_files/MAINFILE_before_cleanup.ipynb` | 161 | 5,633 | 33 | Adds no unique cells beyond the two snapshots above; chronological archive only | `23ceeae8712a69161b14642cb6089a47f9841d4ad0ee43b69c89d4c8395b8e9d` |
| `prev_files/code.ipynb` | — | — | — | Older 7 MB code snapshot | `4ee78624f1011f0bed7d14c79d978f290016a6ffc12bc127eb466228febbeef3` |

**Verified comparison:** all 152 cell sources in `Copy_of_Copy_of_code (1).ipynb` match the
HEAD version, and all 121 cell sources in `Copy_of_code.ipynb` match HEAD. Only stored
outputs/execution counts differ. `before_cleanup` has **0 cells** that are not present in
the other two snapshots, so it is optional for preservation.

**Canonical archive set:** keep `Copy_of_Copy_of_code (1).ipynb` and
`MAINFILE_monolithic_2026-11-21.ipynb`; they are complementary. Keep
`MAINFILE_before_cleanup.ipynb` only if the chronological snapshot is useful.

**What is missing from the active repo:** dataset generation, manifest optimization,
the 21-check audit, and the 12-sample listening comparison. STFT/iSTFT, IRM, Wiener,
SNR evaluation, and the guarded FRCRN adapter are now ported into `intelli_speech/` and
`scripts/`; FRCRN inference has not been run. The remaining archive content is preserved
and not deleted.

**Sensitive content in the archives:** the legacy notebooks contain a personal email in a
`git config` cell and an HF login widget output with a token-looking string
(`oauth-nix-999`). The string appears to be a placeholder, but if it is real it should be
revoked. Never re-commit these notebooks; keep `prev_files/` untracked.

## 3. Blockers

1. **NumPy/Numba incompatibility in the current Colab runtime.** Recorded:
   `ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.` This blocks `librosa`
   and therefore `MAINFILE` step 4. The baseline/STFT environment must pin NumPy
   `>=2.0,<2.5` (or use a NumPy-compatible Numba build).
2. **ClearerVoice declares `numpy<2.0`.** Import succeeded with NumPy 2.5.5 in the
   recorded runtime, but model use is not guaranteed. Use a separate environment pinned
   to `numpy<2.0` for FRCRN.
3. **No persisted FRCRN checkpoint.** The successful init downloaded a checkpoint, but
   it was not found after the runtime reset. A download is a separate, explicit decision.
4. **CPU-only runtime.** `torch 2.14.1+cpu`, `CUDA available: False`. Any FRCRN run will
   be slow; keep batch size at one sample until a GPU runtime is confirmed.
5. **First baseline run is still pending.** The metric and test-set script now exist and
   pass synthetic smoke tests, but no baseline table has been produced on the real
   `paired_dataset_v3` test split. This is the immediate P2 milestone.
6. **Colab checkout is behind `main`, and `main` is not pushed.** The recorded
   `RECOVERY.ipynb` run reported `HEAD: e1b05f9`. `origin/main` is at `7faec80`; local
   `main` is ahead (from `18114b5` "Reorganize repo and add baseline research pipeline"
   onward) and **not pushed**. A Colab `git pull` gets nothing new until local `main` is
   pushed.
7. **Active notebooks contain uncommitted outputs.** `MAINFILE.ipynb` and
   `RECOVERY.ipynb` have execution counts and outputs from a recorded run; the cell
   sources are unchanged vs HEAD. Review before committing.

## 4. Decisions

1. **Three active notebooks only.** `RECOVERY.ipynb` owns environment and cache recovery;
   `MAINFILE.ipynb` owns read-only artifact checks; `TRAIN_V1.ipynb` (added 2026-11-22)
   orchestrates the v1 baseline/GMM/ZF-Net runs and writes only to `experiments/`.
   Research logic belongs in `intelli_speech/` and `scripts/`, not in notebooks.
2. **`prev_files/` is an untracked archive.** Keep it git-ignored. Never import from it or
   use it as runtime input.
3. **`MAINFILE.ipynb` stays read-only.** Scripts may write experiment outputs; the
   notebook only reports what exists.
4. **No automatic dependency changes.** P1 records explicit pins before any install or
   downgrade.
5. **No dataset regeneration, downloads, or inference** are performed or scripted as
   automatic actions. Dataset v3 remains the reference dataset.
6. **Read-only means read-only.** Checks report a `REVIEW` result instead of "fixing"
   artifacts; nothing writes to `Intelli-Speech-data/experiments/` without an explicit
   experiment command.

## 5. Next tasks

### P0 — Repository reorganization — done

- Canonical archives preserved under `prev_files/` and git-ignored; cell sources verified
  against HEAD.
- Root `progress.md` and `plan (1).md` moved into `docs/` with historical notes.
- Active repo reduced to two notebooks, `intelli_speech/`, `scripts/`, `tests/`, and three
  `docs/` files.
- Experiment/checkpoint/audio artifacts added to `.gitignore`.
- Committed locally as the P0 reorganization commit; not pushed.

### P1 — Unblock the runtime

1. Pin the baseline/STFT environment to NumPy `>=2.0,<2.5` in
   `requirements-colab.txt` and verify `librosa.stft`/`librosa.istft` + `soundfile`.
2. Keep FRCRN in a separate environment pinned to `numpy<2.0` in
   `requirements-colab-frcrn.txt`; record the decision before installing.
3. Pull `main` in Colab, run `RECOVERY.ipynb` steps 1–6, then `MAINFILE.ipynb` top to
   bottom. Record: versions, checkpoint load, iSTFT round-trip error, experiment listing,
   and FRCRN checkpoint status.

### P2 — Port the research into code

1. `intelli_speech/dataset.py` — metadata loading, split filtering, pair iteration. **Done.**
2. `intelli_speech/preprocessing.py` — mono/16 kHz conversion, trimming, noise segments,
   SNR mixing. **Done.**
3. `intelli_speech/stft.py` — STFT/iSTFT configuration and checkpoint loading. **Done.**
4. `intelli_speech/baselines.py` — oracle IRM and non-oracle Wiener. **Done.**
5. `intelli_speech/evaluation.py` — SNR improvement and optional PESQ/STOI. **Done.**
6. `scripts/evaluate_baselines.py` — first real metrics table on the 50-source test split.
   **Code done; run pending.**
7. `intelli_speech/frcrn.py` + `scripts/run_frcrn.py` — guarded adapter and fake-model
   tests are done; real inference is still pending a ClearerVoice runtime and checkpoint.
8. `scripts/build_dataset.py` + `scripts/audit_dataset.py` — reproducibility, lower
   priority because dataset v3 already exists.

## 6. Validation performed

| Check | Result |
|---|---|
| Archive cell-source comparison vs HEAD | 152/152 and 121/121 identical |
| `before_cleanup` unique cells vs other archives | 0 |
| System Python pytest | 73 passed, 6 skipped (librosa absent) |
| Temporary venv pytest with librosa 0.10.2.post1 | 79 passed |
| `npx tsc --noEmit -p tsconfig.app.json` | clean |
| `prev_files/` tracked by Git | no (git-ignored) |
| New module/script import check | passed |

## 7. Still unverified

- That both notebooks run end to end in Colab (no runtime available in this session).
- That `analysis_cache/stft_example.npz` and the `experiments/` tree still exist in Drive.
- The librosa-dependent STFT, baseline, and CLI paths on the actual Colab runtime. They
  passed locally in an isolated venv with librosa 0.10.2.post1 and NumPy 2.3.1, but the
  Colab NumPy/Numba combination is still unverified.
- Any FRCRN inference or persisted checkpoint.
- Objective enhancement metrics for any method.
- Perceptual quality of the Wiener baseline beyond the recorded 12-sample listening pass.
