# Intelli-Speech — Project Notes

**Last updated:** 2026-11-22
**Purpose:** Single source of truth for verified state, blockers, decisions, and the next
task. Update it only with observed evidence.

## Notebook and archive map

| File | Role | Writes |
|---|---|---|
| `RECOVERY.ipynb` | Runtime, Drive/repo, cache, Hugging Face, checkpoint, ClearerVoice recovery | cache directories only |
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
6. **Colab checkout is behind `main`.** The recorded `RECOVERY.ipynb` run reported
   `HEAD: e1b05f9`; local `main` is at `7faec80`. The `intelli_speech/` package is not
   present in that checkout. Pull before the next run.
7. **Active notebooks contain uncommitted outputs.** `MAINFILE.ipynb` and
   `RECOVERY.ipynb` have execution counts and outputs from a recorded run; the cell
   sources are unchanged vs HEAD. Review before committing.

## 4. Decisions

1. **Two active notebooks only.** `RECOVERY.ipynb` owns environment and cache recovery;
   `MAINFILE.ipynb` owns read-only artifact checks. Research logic belongs in
   `intelli_speech/` and `scripts/`, not in notebooks.
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
