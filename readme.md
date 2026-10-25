# Intelli-Speech

### A Speech Enhancement and Voice-Aid Research Project

Intelli-Speech is a research project focused on developing a speech-enhancement pipeline for noisy Telugu speech, with the broader goal of supporting a voice-aid system for people with speech impairments and elderly individuals.

The project explores deep-learning-based speech processing to improve speech clarity in noisy environments. The proposed research direction includes a ZF-Net-based neural network, with the potential integration of Gaussian Mixture Models (GMMs) and time-frequency signal processing.

> **Project status:** Research and development. Dataset inspection, architecture validation, implementation, and experimental evaluation are ongoing.

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

The project uses an existing Telugu subset of the Vaani transcribed speech dataset stored in Google Drive.

**Dataset location in Google Colab:**

`/content/drive/MyDrive/Intelli-Speech-data/vaani_telugu_transcribed`

The dataset remains in Google Drive and is accessed from Colab after mounting the drive with the user's authorization.

Before model development, the dataset will be examined to establish:

- Directory structure and file inventory.
- Audio formats, sample rates, channels, and durations.
- Metadata structure and transcript availability.
- Mapping between audio files and transcripts.
- Missing files, unreadable samples, and other data-quality issues.
- Whether the available data supports the intended speech-enhancement objective.

**Important:** Transcribed speech does not automatically provide paired noisy and clean audio. The suitability of the dataset for supervised speech enhancement must be verified before choosing a training strategy.

The original dataset must remain unchanged. Dataset files are not stored in this GitHub repository.

## Technology Stack

The planned tools and technologies include:

- **Python** — experimentation and implementation.
- **TensorFlow** — proposed deep-learning framework.
- **NumPy and SciPy** — numerical computing and signal processing.
- **Audio-processing libraries** — reading and analyzing audio files, selected according to dataset requirements.
- **Google Colab** — execution, experiments, training, and evaluation.
- **Google Drive** — persistent storage for the dataset, checkpoints, and experiment outputs.
- **Git and GitHub** — version control and collaboration.

Dependencies will be finalized as the implementation develops.

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
    "vaani_telugu_transcribed"
)

if not DATASET_PATH.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_PATH}"
    )

print("Dataset directory found:", DATASET_PATH)
```

This only verifies that the directory exists; it does not validate the dataset contents or training suitability.

Colab runtime storage is temporary. Save checkpoints and experiment outputs to an appropriate location in Google Drive if they need to persist across sessions.

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

- [ ] Inspect and document the existing Telugu dataset.
- [ ] Verify audio metadata and transcript mappings.
- [ ] Determine whether suitable clean/noisy training pairs are available or how appropriate training targets could be obtained.
- [ ] Review relevant speech-enhancement research and validate the proposed architecture.
- [ ] Establish baseline experiments and evaluation criteria.
- [ ] Implement and evaluate the selected model.
- [ ] Explore integration into a voice-aid software prototype.

## Project Repository

**GitHub:** https://github.com/niharika9nx/Intelli-Speech

## Acknowledgment

Intelli-Speech is being developed as a research-oriented project toward assistive speech technology, with an emphasis on speech clarity, reproducible experimentation, and the eventual development of a voice-aid prototype.
