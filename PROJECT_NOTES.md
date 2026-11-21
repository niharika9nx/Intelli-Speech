# Intelli-Speech — Project Notes

**Last updated:** 2026-11-21
**Purpose:** Single source of truth for verified state, blockers, decisions, and the next
task. Update it only with observed evidence.

**Notebook map**

| File | Role | Writes |
|---|---|---|
| `RECOVERY.ipynb` | Runtime, Drive/repo, cache, Hugging Face, checkpoint, ClearerVoice recovery | cache directories only |
| `MAINFILE.ipynb` | Read-only workflow over existing Drive artifacts | none |
| `prev_files/MAINFILE_monolithic_2026-11-21.ipynb` | Archived 155-cell history (dataset v1/v2/v3 generation, IRM, Wiener, FRCRN drafts); local-only, git-ignored | none |
| `docs/dataset_structure.md` | Dataset structure and audit record | — |
| `progress.md` | Dataset-generation progress narrative | — |

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
- Evidence: `docs/dataset_structure.md`, `progress.md`. The audit itself was **not**
  re-run in this session; the numbers above are the recorded audit results.

### 1.2 STFT analysis state — recorded

- `librosa.stft` with `n_fft=512`, `hop_length=128` at 16 kHz.
- A saved analysis checkpoint exists at
  `Intelli-Speech-data/analysis_cache/stft_example.npz` (clean/noisy waveforms, their
  STFTs, sample rate, FFT and hop settings), written and re-loaded in the historical
  notebook session.
- Evidence: archived notebook cells for checkpoint save, restore, and the loader helper.
  Presence in Drive is re-checked by `MAINFILE.ipynb` step 4 and `RECOVERY.ipynb` step 5.

### 1.3 Baselines — implemented, partially evaluated

- Oracle IRM computed from clean and estimated noise magnitudes:
  `mask = |clean| / (|clean| + |noise| + 1e-10)`.
- Non-oracle Wiener-style baseline implemented using only the noisy input.
- A 12-sample listening comparison (clean reference / noisy input / Wiener-style output)
  was executed and its audio players are stored in the archived notebook output.
- **No objective metrics exist yet.** PESQ, STOI, or SNR-improvement numbers have not
  been computed for any baseline.

### 1.4 FRCRN — never executed

- Recorded preflight: Python 3.13.18, NumPy 2.4.6, PyTorch 2.14.1+cpu,
  torchaudio 2.11.0+cpu, CUDA unavailable, ClearerVoice not installed.
- No FRCRN checkpoint exists in any inspected location
  (`checkpoints/FRCRN_SE_16K`, `persistent_cache/clearvoice_runtime`,
  `~/.cache/clearvoice`, Hugging Face caches — all 0 checkpoint files).
- The single-sample FRCRN cell in the archived notebook has no output, so no enhanced
  audio has ever been produced.

### 1.5 Repository utilities — verified in this session

- `intelli_speech/audio_io.py`, `intelli_speech/metrics.py`, and
  `scripts/setup_colab_runtime.py` exist and behave as documented.
- `python -m pytest -q` → **15 passed**.

## 2. Blockers

1. **ClearerVoice cannot be installed in the current runtime.** Recorded evidence:
   installing `clearvoice` resolved to NumPy 1.26.4 and reported conflicts with packages
   requiring NumPy ≥ 2, and the following `from clearvoice import ClearVoice` raised
   `ModuleNotFoundError`. `clearvoice==0.1.2` declares `numpy<2.0`, while the runtime
   carries NumPy 2.x. A plain install is not a safe recovery path.
2. **No FRCRN checkpoint.** Even with a working import, FRCRN would need to download a
   checkpoint. That download has never been approved or performed.
3. **CPU-only runtime.** `torch 2.14.1+cpu`, `CUDA available: False`. Any FRCRN run will
   be slow; batch sizes must stay at one sample until a GPU runtime is confirmed.
4. **No evaluation protocol.** No objective metric, test-set script, or baseline table
   exists, so enhancement quality cannot currently be claimed or compared.
5. **Historical path ambiguity.** Some archived diagnostics ran with `DATASET_DIR`
   pointing at `vaani_telugu_transcribed` (246 rows) instead of `paired_dataset_v3`
   (1,000 rows). The new notebooks pin `paired_dataset_v3` explicitly.
6. **Repository sync — resolved 2026-11-21.** The restructure (`MAINFILE.ipynb`,
   `RECOVERY.ipynb`, `intelli_speech/`, `tests/`, `scripts/`, `opencode.json`,
   `AGENTS.md`, `pyproject.toml`, `requirements-colab.txt`, docs) was committed and
   pushed to `origin/main` (`16db1c4`, `e9af1a8`). `prev_files/` is untracked and
   git-ignored by decision; the archives remain on disk locally only.
7. **Pre-existing working-tree changes not made by this task:** the `readme.md`
   modification (the "Reusable Python utilities" section) was reviewed and committed
   with the sync. The deletion of `Copy_of_Copy_of_code.ipynb` remains uncommitted and
   untouched.

## 3. Decisions

1. **One active working notebook.** `MAINFILE.ipynb` is now a compact, read-only
   artifact workflow (8 cells). Dataset generation, IRM/Wiener experiments, and FRCRN
   drafts no longer live there.
2. **Recovery has exactly one home.** `RECOVERY.ipynb` owns runtime diagnostics, Drive
   and repository sync, persistent caches, Hugging Face authentication state, checkpoint
   verification, and ClearerVoice inspection. `MAINFILE.ipynb` deliberately contains no
   mount, install, or login cell, and fails with a pointer to `RECOVERY.ipynb` instead.
3. **History is archived, not deleted.** The 155-cell, 57.9 MB monolithic notebook was
   copied byte-for-byte to `prev_files/MAINFILE_monolithic_2026-11-21.ipynb`
   (SHA-256 `FAFFF8829B2452BD7369D6A8F2F51B3B0B03E7BA34C065A6439B2404D104FC58`).
   `prev_files/MAINFILE_before_cleanup.ipynb` was left untouched. As of 2026-11-21,
   `prev_files/` is untracked and git-ignored (kept locally only).
4. **No automatic dependency changes.** No cell installs, upgrades, or downgrades
   anything. Resolving ClearerVoice/NumPy requires an explicit, recorded decision.
5. **No dataset regeneration, downloads, or inference** were performed or scripted as
   automatic actions.
6. **Read-only means read-only.** Checks report a `REVIEW` result instead of "fixing"
   artifacts; nothing writes to `Intelli-Speech-data/experiments/`.

## 4. Next task

**Primary next task — verify the restructure in a fresh Colab runtime (read-only).**

1. Open `RECOVERY.ipynb` in a fresh Colab runtime and run steps 1–6.
2. Run `MAINFILE.ipynb` top to bottom.
3. Paste the observed output into this file under a new "Runtime verification" section,
   including: Python/NumPy/PyTorch versions, whether the STFT checkpoint loaded, the
   iSTFT round-trip error, the experiment-artifact listing, and whether any FRCRN
   checkpoint was found.

This confirms that the split works in the real environment and captures the true artifact
state. It requires no installs, no downloads, and no dataset changes.

**Follow-ups, in order**

1. Decide the ClearerVoice environment (isolated venv/runtime pinned to `numpy<2.0`
   versus a dedicated pinned Colab session). Record the decision here before acting.
2. Define and run the baseline evaluation: SNR improvement plus PESQ/STOI where the
   metric licensing and 16 kHz inputs permit, on the 50-source test split, for the
   Wiener baseline.
3. **Done 2026-11-21** — committed `intelli_speech/`, `tests/`, `scripts/`, both
   notebooks, and this file so the Colab clone matches the local state (`16db1c4`,
   pushed to `origin/main`).
4. Only after (1)–(3): one-sample FRCRN enhancement, listened to and compared against
   the baseline, saved under `experiments/clearvoice_frcrn/` without overwriting.

## 5. Validation performed for this change

| Check | Result |
|---|---|
| Archive integrity (`Get-FileHash` on source and archive) | identical SHA-256 |
| `nbformat.validate` on `MAINFILE.ipynb` | valid, nbformat 4.5 |
| `nbformat.validate` on `RECOVERY.ipynb` | valid, nbformat 4.5 |
| Python syntax compile of every code cell (both notebooks) | no syntax errors |
| Stored outputs / execution counts in both notebooks | none |
| Duplicate recovery concerns (mount/install/login/git in `MAINFILE`) | none |
| `python -m pytest -q` | 15 passed |

## 6. Still unverified

- That both notebooks run end to end in Colab (no runtime available in this session).
- That `analysis_cache/stft_example.npz` and the `experiments/` tree still exist in Drive.
- Any ClearerVoice import or FRCRN inference in the current runtime.
- Objective enhancement metrics for any method.
- Perceptual quality of the Wiener baseline beyond the recorded 12-sample listening pass.

## 7. Runtime verification — pending

To be completed in a fresh Colab runtime: run `RECOVERY.ipynb` steps 1–6, then
`MAINFILE.ipynb` top to bottom. Capture:

- [ ] Python / NumPy / PyTorch versions
- [ ] Whether `analysis_cache/stft_example.npz` loaded
- [ ] iSTFT round-trip error
- [ ] `experiments/` artifact listing
- [ ] Whether any FRCRN checkpoint was found
- [ ] Observed output pasted below
