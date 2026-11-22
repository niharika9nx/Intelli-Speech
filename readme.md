# Intelli-Speech

### A Speech Enhancement and Voice-Aid Research Project

Intelli-Speech is a research project focused on developing a speech-enhancement pipeline for noisy Telugu speech, with the broader goal of supporting a voice-aid system for people with speech impairments and elderly individuals.

The project explores deep-learning-based speech processing to improve speech clarity in noisy environments. The proposed research direction includes a ZF-Net-based neural network, with the potential integration of Gaussian Mixture Models (GMMs) and time-frequency signal processing.

> **Project status:** Research and development. Dataset v3 is generated and audited; baseline evaluation and model implementation are ongoing.

## Objectives

- Investigate techniques for enhancing speech affected by background noise.
- Explore the use of ZF-Net and convolutional neural networks for speech representation and enhancement.
- Evaluate suitable combinations of deep learning and statistical signal-processing techniques.
- Develop a reproducible training and evaluation pipeline.
- Work toward a software prototype for an assistive voice-aid application.

## Proposed Methodology

The following pipeline represents the current research direction. Individual stages and their interfaces will be validated before implementation is finalized.

<pre>
Input Audio
    │
    ▼
Audio Analysis and Preprocessing
    │
    ▼
Short-Time Fourier Transform (STFT)
    │
    ▼
ZF-Net / CNN-Based Processing
    │
    ▼
Gaussian Mixture Model (GMM) Integration
    │
    ▼
Inverse Short-Time Fourier Transform (iSTFT)
    │
    ▼
Enhanced Audio
    │
    ▼
Evaluation and Analysis
</pre>

### Pipeline components

| Component | Purpose |
|---|---|
| Audio preprocessing | Inspect audio files and prepare suitable inputs for experimentation. |
| STFT | Represent audio in the time-frequency domain for analysis and processing. |
| ZF-Net / CNN | Investigate neural-network-based feature learning and speech enhancement. |
| GMM | Explore statistical modeling of speech or noise features, depending on the validated design. |
| iSTFT | Reconstruct a time-domain waveform when the processed representation supports reconstruction. |
| Evaluation | Compare input and output audio using appropriate objective metrics and listening tests. |

The final architecture will depend on the dataset characteristics, availability of suitable training targets, and experimental findings. The proposed components should not be interpreted as an already implemented or validated model.

## Dataset

The project uses a paired Telugu speech dataset (`paired_dataset_v3`) prepared for speech-enhancement research. It was built from OpenSLR SLR66 clean speech and DEMAND environmental noise, and it has been generated, independently audited (21/21 checks passed), and reviewed through listening tests.

**Dataset location in Google Colab:**

`/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3`

**Verified composition:**

- 500 clean speech targets (16 kHz mono) derived from OpenSLR SLR66.
- 1,000 noisy mixtures at target SNR levels of 5 dB and 10 dB (two noisy variants per source).
- 80/10/10 train/validation/test split with no candidate speaker overlap across splits.
- 200 mixtures per DEMAND noise category (DKITCHEN, NPARK, OOFFICE, STRAFFIC, TBUS).

The dataset remains in Google Drive and is accessed from Colab after mounting the drive with the user's authorization.

The dataset has not yet been used to train or evaluate a model. Baseline code now exists, but no objective metrics have been recorded yet. The enhancement objective, evaluation protocol, and model architecture are still being defined.

**Important:** The dataset uses controlled synthetic noise mixtures; performance on naturally noisy speech remains untested.

The VAANI Telugu recordings (`vaani_telugu_transcribed`) are retained in Drive as supplementary real-world noisy-speech evaluation candidates, subject to suitability checks.

The original source recordings must remain unchanged. Dataset files are not stored in this GitHub repository.

## Technology Stack

The planned tools and technologies include:

- **Python** — experimentation and implementation.
- **TensorFlow** — proposed deep-learning framework.
- **NumPy and SciPy** — numerical computing and signal processing.
- **Audio-processing libraries** — reading and analyzing audio files, selected according to dataset requirements.
- **Google Colab** — execution, experiments, training, and evaluation.
- **Google Drive** — persistent storage for the dataset, checkpoints, and experiment outputs.
- **React, Vite, and TypeScript** — browser demonstration for recording, playback, and the proposed enhancement experience.
- **Git and GitHub** — version control and collaboration.

Dependencies will be finalized as the implementation develops.

## Frontend Demonstration

The repository includes a small browser interface in [`app/`](app/) for exploring the intended voice-aid experience: record speech, listen back, and preview the enhancement flow.

**Important:** this frontend is a demonstration, not the enhancement model.

- Recording and playback are real and run locally in the browser using the microphone and `MediaRecorder` APIs. Recordings are not uploaded.
- The enhancement step currently plays a bundled sample clip and is labelled as a demo in the interface. It does not process the user's recording.
- The written interface includes a welcome page, the recording workflow, and a profile page for browser-local display and accessibility preferences. No account or backend is connected.
- The real enhancement engine is not connected yet. Its interface will be defined alongside the research implementation.

To run the frontend locally:

```bash
cd app
npm install
npm run dev
```

Open the local address printed by Vite. Microphone access requires `localhost` or an HTTPS connection.

## Development Environment

The project uses separate environments for code development, execution, and data storage.

| Environment | Responsibility |
|---|---|
| Local development environment | Writing, reviewing, and maintaining project code. |
| GitHub | Version-controlled code, notebooks, and documentation. |
| Google Colab | Dataset inspection, preprocessing, training, and evaluation. |
| Google Drive | Original dataset, model checkpoints, and generated research outputs. |

### Running the project in Google Colab

1. Open the project notebook in Google Colab.
2. Mount Google Drive and authorize access when prompted.
3. Verify that the dataset directory exists at the configured path.
4. Run the dataset inspection workflow.
5. Review the resulting dataset characteristics and limitations before proceeding to preprocessing or training.

Example code to mount Drive and verify the directory:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

DATASET_PATH = Path(
    "/content/drive/MyDrive/Intelli-Speech-data/"
    "paired_dataset_v3"
)

if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_PATH}"
    )

print("Dataset directory found:", DATASET_PATH)
```

This only verifies that the directory exists; it does not validate the dataset contents or training suitability.

Colab runtime storage is temporary. Save checkpoints and experiment outputs to an appropriate location in Google Drive if they need to persist across sessions.

## Reusable Python utilities

The repository now includes a small, notebook-independent Python foundation:

- `intelli_speech/audio_io.py` resolves metadata-relative paths, loads and writes
  validated mono audio, and rejects empty, non-finite, or unexpected-format samples.
- `intelli_speech/metrics.py` provides SNR, RMS, peak, and clipping calculations.
- `intelli_speech/dataset.py` loads and validates `metadata.csv`, filters splits, and
  yields resolved clean/noisy pairs without modifying the dataset.
- `intelli_speech/preprocessing.py` provides mono/16 kHz conversion, optional silence
  trimming, noise-segment selection, and SNR-controlled mixing.
- `intelli_speech/stft.py` defines the STFT/iSTFT configuration and checkpoint loader.
- `intelli_speech/baselines.py` implements the oracle IRM and non-oracle Wiener-style
  baselines. The oracle method uses the clean reference and is not deployable.
- `intelli_speech/evaluation.py` computes error-to-reference SNR, SNR improvement, and
  optional PESQ/STOI metrics when those packages are installed.
- `intelli_speech/frcrn.py` is a guarded ClearerVoice/FRCRN adapter; it does not import
  ClearerVoice at module import time and reports a clear dependency error when absent.
- `scripts/evaluate_baselines.py` runs a baseline over one split and writes
  `metrics.csv`, `config.json`, and `summary.json` under an experiment directory.
- `scripts/run_frcrn.py` runs the guarded FRCRN adapter on a bounded number of pairs and
  writes enhanced WAVs plus metrics under `experiments/clearvoice_frcrn/`.
- `scripts/setup_colab_runtime.py` reports the runtime and dependency state without
  installing or downgrading packages.
- `requirements-colab.txt` is the baseline/STFT environment (NumPy `<2.5`).
- `requirements-colab-frcrn.txt` is the separate ClearerVoice/FRCRN environment
  (NumPy `<2.0`). Do not install both files into one environment.

Run focused utility tests from the repository root:

```bash
python -m pytest
```

Run local runtime diagnostics:

```bash
python scripts/setup_colab_runtime.py
```

Run the first baseline on the test split after mounting Drive:

```bash
python scripts/evaluate_baselines.py \
  --dataset-dir /content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3 \
  --split test \
  --method wiener
```

Use `--method oracle-irm` only for the reference upper-bound experiment; it uses the
clean target to construct the mask and must not be reported as a deployable method.
Outputs default to `Intelli-Speech-data/experiments/<method>_<split>/` and never modify
the dataset.

Run FRCRN only in the separate ClearerVoice environment after a checkpoint is available:

```bash
python scripts/run_frcrn.py \
  --dataset-dir /content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3 \
  --split test \
  --limit 1
```

The FRCRN script processes one sample by default, refuses to write inside the dataset,
and writes results to `Intelli-Speech-data/experiments/clearvoice_frcrn/`.

## Repository Layout

The active repository is intentionally small:

- `intelli_speech/` — reusable Python modules for dataset loading, preprocessing, STFT,
  baselines, evaluation, and the guarded FRCRN adapter.
- `scripts/` — runnable entry points for dataset inspection, runtime diagnostics, baseline
  evaluation, and FRCRN inference.
- `tests/` — focused tests using synthetic arrays and temporary files; no Drive access.
- `docs/` — dataset report, historical progress, and historical project plan.
- `app/` — the clearly labelled frontend demonstration (not the enhancement model).
- `MAINFILE.ipynb` and `RECOVERY.ipynb` — orchestration and runtime recovery only.
- `prev_files/` — git-ignored archive of pre-cleanup notebooks; never imported by code.

Experiment outputs, checkpoints, audio artifacts, and `analysis_cache/` are ignored and
should live in Google Drive or another persistent location.

## Research Workflow

The development process is organized into the following stages:

1. **Dataset reconnaissance** — inspect the existing data and document verified findings.
2. **Research and design** — review relevant literature and establish a technically justified enhancement objective.
3. **Preprocessing** — define audio representations, data preparation, and training targets.
4. **Model implementation** — develop and test the selected architecture in small, reproducible steps.
5. **Experimental evaluation** — compare against suitable baselines and evaluate enhanced audio.
6. **Prototype development** — investigate integration into a software-based voice-aid application after the model demonstrates adequate performance.

Training and evaluation data should be separated appropriately to reduce data leakage. Baselines, configuration details, and experimental results should be documented for reproducibility.

## Evaluation

Evaluation criteria will be selected after the enhancement objective and available reference data have been established.

Potential measures include:

- **Signal-to-Noise Ratio (SNR):** assess signal quality relative to noise when the required signals are available.
- **PESQ:** evaluate perceptual speech quality where the metric's input requirements and licensing permit.
- **STOI:** assess speech intelligibility under its applicable evaluation conditions.
- **Listening tests:** assess perceived clarity, residual noise, and speech artifacts.

No performance claims are made until experiments have been conducted and results have been recorded.

## Repository Guidelines

- Commit source code, notebooks, configuration examples, and documentation.
- Do not commit datasets, credentials, model checkpoints, or large generated artifacts.
- Keep original dataset files unchanged.
- Use configurable paths rather than relying on machine-specific locations.
- Document experiment settings, evaluation results, and known limitations.
- Validate each pipeline stage before launching expensive training runs.

## Current Priorities

- [x] Inspect and document the existing Telugu dataset.
- [x] Verify audio metadata and transcript mappings.
- [x] Build and audit paired clean/noisy dataset v3.
- [ ] Review relevant speech-enhancement research and validate the proposed architecture.
- [ ] Establish baseline experiments and evaluation criteria. **In progress:** Wiener
  and oracle IRM code exists; the first metrics table is pending a working Colab runtime.
- [ ] Implement and evaluate the selected model.
- [ ] Explore integration into a voice-aid software prototype.

## Project Repository

**GitHub:** https://github.com/niharika9nx/Intelli-Speech

## Acknowledgment

Intelli-Speech is being developed as a research-oriented project toward assistive speech technology, with an emphasis on speech clarity, reproducible experimentation, and the eventual development of a voice-aid prototype.
