# Intelli-Speech — Project Progress

## 1. Project Overview

**Project:** Intelli-Speech  
**Objective:** Develop a Telugu speech-enhancement pipeline for a voice-aid prototype, aimed at improving speech intelligibility in noisy environments.

The project explores deep-learning-based speech enhancement, with an initial research direction involving CNN/ZF-Net feature extraction and Gaussian Mixture Models (GMM).

**Current stage:** Dataset preparation and validation.

## 2. Environment and Constraints

- Development environment: Google Colab.
- Dataset storage: Google Drive.
- Available local hardware: Laptop with 8 GB RAM and integrated graphics; no NVIDIA GPU.
- Compute strategy: Use free cloud resources where possible.
- Dataset preparation should be memory-efficient and avoid downloading unnecessary large datasets.
- Preserve original recordings and maintain reproducible preprocessing steps.

## 3. Dataset Preparation

### 3.1 OpenSLR SLR66 — Telugu Speech

**Drive location:**  
`/content/drive/MyDrive/Intelli-Speech-data/openslr_slr66`

Files available:

- `te_in_female.zip` — 2,294 WAV recordings.
- `te_in_male.zip` — 2,154 WAV recordings.
- `line_index_female.tsv` — female recording names and Telugu transcripts.
- `line_index_male.tsv` — male recording names and Telugu transcripts.

**Total available recordings:** 4,448.

#### Validation completed

- Confirmed that both ZIP archives can be read.
- Inspected multiple sample recordings from both archives.
- Verified that sample transcript lookups matched the corresponding recordings.
- Sample audio was mono, with a sampling rate of 48 kHz.
- Sample durations varied, including short clips and longer recordings.
- Listening tests indicated that the inspected recordings were generally clear and suitable as candidate clean-speech references.

**Current assessment:** OpenSLR SLR66 is the primary candidate source for clean Telugu speech targets. The complete dataset has not yet undergone comprehensive quality validation.

### 3.2 DEMAND — Environmental Noise

**Drive location:**  
`/content/drive/MyDrive/Intelli-Speech-data/DEMAND/extracted`

Five environmental noise categories have been extracted:

- `DKITCHEN` — kitchen noise.
- `NPARK` — park noise.
- `OOFFICE` — office noise.
- `STRAFFIC` — street traffic noise.
- `TBUS` — bus interior noise.

Each category contains 16 channel recordings.

**Available noise recordings:** 5 categories × 16 recordings = 80 recordings.

- Sampling rate observed: 16 kHz.
- Channels observed: Mono.
- Duration observed: 300 seconds per recording.
- Total observed duration: approximately 400 minutes (6 hours and 40 minutes).

These recordings contain environmental noise rather than paired speech-and-noise samples.

**Current assessment:** DEMAND will provide noise segments for controlled mixing with clean speech.

### 3.3 VAANI Telugu Speech

**Drive location:**  
`/content/drive/MyDrive/Intelli-Speech-data/vaani_telugu_transcribed`

Approximately 246 audio recordings and a metadata file are available.

Initial listening indicates that these recordings contain speech with background noise.

**Potential use:** Real-world evaluation or supplementary noisy-speech data.

**Important:** These recordings have not been established as clean targets or as perfectly aligned noisy-clean pairs.

## 4. Silence-Trimming Validation

The OpenSLR recordings contain noticeable leading and trailing silence.

A trimming test was implemented using `librosa.effects.split` with:

- `top_db = 30`
- `frame_length = 2048`
- `hop_length = 512`
- Padding: 150 ms before the first detected active region and after the last detected active region.

The trimmed audio is created by retaining one continuous segment from the first detected active region to the last. This preserves internal pauses between those regions.

### Validation completed

- Tested 10 recordings initially: five female and five male.
- Reviewed the original and trimmed durations.
- Some recordings showed substantial duration reductions because of silence.
- Expanded the listening test to 30 recordings: 15 female and 15 male.
- Manually listened to the original and trimmed versions.
- Confirmed that the trimming worked well on the reviewed samples.

**Current decision:** Use this trimming approach for the initial dataset-generation pipeline.

**Safeguard:** Keep the original source recordings unchanged. Continue checking for clipped words or quiet speech in future samples, particularly when scaling up.

## 5. Planned Data-Generation Pipeline

The next implementation stage is to generate paired clean and noisy speech examples.

### Step 1 — Select clean speech

- Sample a small set of OpenSLR recordings.
- Read the corresponding Telugu transcript metadata where available.
- Preserve the original recordings.

### Step 2 — Preprocess speech

- Convert audio to mono if necessary.
- Trim leading and trailing silence using the validated method.
- Resample speech from 48 kHz to 16 kHz.
- Validate duration and audio integrity.

### Step 3 — Prepare environmental noise

- Load suitable DEMAND noise recordings.
- Extract noise segments matching each speech recording's duration.
- Use reproducible random sampling for experiments.

### Step 4 — Mix speech and noise

Generate noisy mixtures at controlled signal-to-noise ratios (SNRs).

Initial target SNR levels:

- 5 dB
- 10 dB

The SNR calculation must use the active clean-speech signal and the selected noise segment. Apply safeguards against clipping while preserving the intended SNR as closely as possible.

### Step 5 — Save paired examples

Each generated example should contain:

- Clean speech target.
- Corresponding noisy speech input.
- Source recording identifier.
- Noise category and source identifier.
- Target SNR.
- Sampling rate and duration.
- Transcript, where available.

Store the files separately from the original datasets and create a metadata CSV linking each noisy input to its clean target.

### Step 6 — Small-batch verification

Start with 20 clean recordings.

- Generate noisy examples at 5 dB and 10 dB.
- Listen to clean and noisy versions.
- Check SNR calculations, duration alignment, clipping, and file integrity.
- Confirm that every noisy example has the correct clean target.
- Scale up only after this validation passes.

## 6. Model Development — Planned

The research direction currently includes:

- CNN/ZF-Net-based feature extraction.
- Gaussian Mixture Model (GMM) components.
- Noise suppression and speech reconstruction.
- Short-time Fourier transform (STFT) and inverse STFT as potential signal-processing stages.

The exact architecture, feature representation, training objective, and integration of ZF-Net and GMM still need to be specified and validated experimentally.

No model-training results or enhancement-quality claims have been established yet.

## 7. Current Status

### Completed

- [x] Established the initial project data directory in Google Drive.
- [x] Extracted the five selected DEMAND noise categories.
- [x] Inspected OpenSLR SLR66 archives and transcript indexes.
- [x] Listened to representative OpenSLR recordings.
- [x] Tested silence trimming on sample recordings.
- [x] Manually reviewed 30 original-versus-trimmed recordings.
- [x] Approved the trimming approach for the initial pipeline.

### In progress / next

- [ ] Implement the preprocessing pipeline.
- [ ] Resample clean speech to 16 kHz.
- [ ] Implement and test SNR-controlled noise mixing.
- [ ] Generate a first batch of 20 clean recordings at 5 dB and 10 dB.
- [ ] Save paired WAV files and metadata.
- [ ] Listen to generated mixtures and validate alignment, SNR, and clipping.
- [ ] Decide on train, validation, and test splits.
- [ ] Finalize the model architecture and training procedure.
- [ ] Train and evaluate the first baseline.
- [ ] Compare enhanced speech against noisy input using objective metrics and listening tests.

## 8. Important Data-Management Rules

1. Never overwrite original dataset files.
2. Keep clean targets and noisy inputs clearly separated.
3. Record the source identifiers and mixing parameters for reproducibility.
4. Avoid mixing related versions of the same source recording across training and evaluation splits.
5. Do not treat naturally noisy VAANI recordings as clean targets without validation.
6. Validate the small-batch pipeline before generating large datasets.
7. Document actual experimental results separately from proposed methods.

---

**Last updated:** 2026-10-25  
**Current milestone:** Dataset inspection and silence-trimming validation completed; paired noisy-clean sample generation is the next task.
