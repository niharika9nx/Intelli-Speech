# Intelli-Speech — Project Progress

**Last updated:** 2026-10-25  
**Current milestone:** Paired Telugu speech dataset v3 generated and independently validated.

## 1. Project Overview

**Project:** Intelli-Speech

**Objective:** Develop a Telugu speech-enhancement pipeline for a voice-aid prototype, aimed at improving speech intelligibility in noisy environments.

The research explores deep-learning-based speech enhancement, with an initial research direction involving CNN/ZF-Net feature extraction and Gaussian Mixture Models (GMM).

**Current stage:** Dataset preparation completed for the initial paired dataset; qualitative review and dataset documentation remain before model development.

## 2. Environment and Constraints

- Development environment: Google Colab.
- Dataset storage: Google Drive.
- Local hardware: Laptop with 8 GB RAM and integrated graphics; no NVIDIA GPU.
- Compute strategy: Use free cloud resources where possible.
- Dataset preparation should be memory-efficient and avoid downloading unnecessary large datasets.
- Preserve original recordings and maintain reproducible preprocessing steps.
- Model training and final architecture decisions have not started.

## 3. Dataset Inventory

### 3.1 OpenSLR SLR66 — Telugu Speech

**Drive location:**

`/content/drive/MyDrive/Intelli-Speech-data/openslr_slr66`

Available files:

- `te_in_female.zip` — 2,294 WAV recordings.
- `te_in_male.zip` — 2,154 WAV recordings.
- `line_index_female.tsv` — female recording names and Telugu transcripts.
- `line_index_male.tsv` — male recording names and Telugu transcripts.

**Total available recordings:** 4,448.

#### Validation completed

- Confirmed that both ZIP archives are readable.
- Inspected representative audio recordings and transcript lookups.
- Verified the filename-to-transcript mapping for the selected source manifest.
- Inspected sample recordings at 48 kHz, mono.
- Tested the silence-trimming procedure on 30 recordings: 15 female and 15 male.
- Created a manifest containing 500 selected source recordings, their transcripts, candidate speaker IDs, and train/validation/test assignments.
- Verified that the selected manifest contains 250 female and 250 male recordings.
- Checked that the candidate speaker IDs do not overlap across the assigned splits.

**Current assessment:** OpenSLR SLR66 is the primary source for clean-speech targets. The 500 selected sources have been processed for v3, but the entire 4,448-recording corpus has not undergone comprehensive quality validation.

### 3.2 DEMAND — Environmental Noise

**Drive location:**

`/content/drive/MyDrive/Intelli-Speech-data/DEMAND/extracted`

Five environmental categories are available:

- `DKITCHEN` — kitchen noise.
- `NPARK` — park noise.
- `OOFFICE` — office noise.
- `STRAFFIC` — street traffic noise.
- `TBUS` — bus interior noise.

Each category contains 16 channel recordings.

**Available noise recordings:** 80.

Observed properties:

- Sampling rate: 16 kHz.
- Channels: mono.
- Duration: 300 seconds per recording.
- Total duration: approximately 400 minutes, or 6 hours and 40 minutes.

**Current assessment:** Original DEMAND recordings are used to create controlled noisy mixtures. For v3, 200 mixtures were generated from each of the five environmental categories.

### 3.3 VAANI Telugu Speech

**Drive location:**

`/content/drive/MyDrive/Intelli-Speech-data/vaani_telugu_transcribed`

Approximately 246 audio recordings and a metadata file are available.

Initial listening indicates that these recordings contain speech with background noise.

**Potential use:** Supplementary real-world noisy-speech evaluation, subject to suitability checks.

**Important:** These recordings have not been established as clean targets or as perfectly aligned noisy-clean pairs. They are not used as clean targets in v3.

## 4. Silence-Trimming Validation

The OpenSLR recordings contain leading and trailing silence. The following trimming procedure was tested:

- `librosa.effects.split`
- `top_db = 30`
- `frame_length = 2048`
- `hop_length = 512`
- Padding: 150 ms before the first detected active region and after the last detected active region.

The output retains one continuous segment between the first and last detected active regions, preserving internal pauses.

### Validation completed

- Tested 10 recordings initially.
- Expanded the review to 30 recordings, split equally between female and male speakers.
- Compared original and trimmed durations.
- Manually listened to the original and trimmed versions.
- Confirmed that the procedure worked well on the reviewed samples.
- Applied the procedure to the 500 selected sources during v3 generation.

**Current decision:** Retain this procedure for the initial dataset pipeline.

**Safeguard:** The sample review does not establish that every recording is free from clipped words or incorrectly removed quiet speech. Additional quality checks remain useful.

## 5. Paired Dataset Generation — v3

**Output directory:**

`/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3`

**Source manifest:**

`/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3_manifest_speaker_balanced.csv`

### 5.1 Generation procedure

For each of the 500 selected OpenSLR sources:

1. Read the original audio directly from its ZIP archive.
2. Convert it to mono and resample it to 16 kHz.
3. Trim leading and trailing silence using the validated procedure.
4. Extract noise segments from original DEMAND recordings.
5. Generate noisy mixtures at 5 dB and 10 dB SNR.
6. Save one clean target and two corresponding noisy variants.
7. Record source identifiers, transcripts, split assignments, noise sources, SNR values, durations, sample rate, and peak amplitudes in metadata.

A common gain is applied to the clean target and both noisy mixtures when necessary to prevent clipping while preserving the intended SNR.

### 5.2 Final dataset composition

| Property | Verified result |
|---|---:|
| Unique clean-speech sources | 500 |
| Clean target WAV files | 500 |
| Noisy mixture WAV files | 1,000 |
| Total WAV files | 1,500 |
| Sampling rate | 16 kHz |
| Target SNR levels | 5 dB and 10 dB |
| Mixtures at 5 dB | 500 |
| Mixtures at 10 dB | 500 |
| Training sources | 400 |
| Validation sources | 50 |
| Test sources | 50 |
| Training mixtures | 800 |
| Validation mixtures | 100 |
| Test mixtures | 100 |

Each source has one clean target and two noisy mixtures. Both variants inherit the source's split assignment.

### 5.3 Noise-environment distribution

| DEMAND environment | Noisy mixtures |
|---|---:|
| DKITCHEN | 200 |
| NPARK | 200 |
| OOFFICE | 200 |
| STRAFFIC | 200 |
| TBUS | 200 |
| **Total** | **1,000** |

This provides balanced representation by environment at the dataset level. It does not guarantee that every speaker, gender, or split has an identical distribution of noise categories.

### 5.4 Generation validation

The generation script completed successfully.

Verified results included:

- 500 clean targets and 1,000 noisy mixtures generated.
- 1,000 metadata rows written.
- All five noise environments represented.
- No detected full-scale clipping in the saved noisy WAV files.
- Maximum reported SNR error during generation: approximately 0.000391 dB.
- Maximum noisy-signal peak: approximately 0.9901.
- Manifest fingerprint recorded in the build configuration.

**Manifest SHA-256:**

`2af46ef5c9a4545c0775f08a07b734bc0696a888339a18d5e40cdd1103941eb0`

### 5.5 Independent audit

An independent read-only audit inspected all 1,000 saved clean/noisy pairs.

**Result: 21/21 checks passed.**

The audit verified:

- Expected clean and noisy file counts.
- Unique source IDs and correct pair counts.
- Source-to-clean-target mappings.
- Matching sample rates and audio lengths.
- Finite waveform samples.
- SNR accuracy for all 1,000 mixtures.
- Peak values below full scale.
- Agreement between generated source IDs and the original manifest.
- Agreement of split, speaker ID, and gender fields with the manifest.
- No candidate speaker overlap between training, validation, and test.
- Expected split and noise-environment counts.
- Metadata paths and manifest fingerprint.

Independent SNR results:

| Statistic | Absolute SNR error |
|---|---:|
| Mean | 0.0000251 dB |
| Median | 0.0000162 dB |
| Maximum | 0.0003905 dB |

Duration statistics across the 1,000 pairs:

- Mean: approximately 3.076 seconds.
- Minimum: 0.908 seconds.
- Maximum: approximately 8.673 seconds.

**Current assessment:** The generated dataset passes the implemented structural, mapping, waveform-integrity, SNR, and split checks. These checks do not establish perceptual quality for every example or demonstrate speech-enhancement performance.

## 6. Data Splitting and Leakage Prevention

The 500-source manifest uses an 80/10/10 split:

- Training: 400 sources.
- Validation: 50 sources.
- Test: 50 sources.

The manifest was constructed to keep candidate speaker IDs separate across splits. The independent v3 audit confirmed that no candidate speaker ID appears in more than one split.

All noisy variants of a source inherit the same split as that source.

**Important:** The speaker separation is based on the candidate speaker IDs parsed from the dataset's filename convention. It should not be interpreted as independently verified real-world speaker identity.

The manifest is retained separately from the generated dataset so the split assignments can be reused reproducibly.

## 7. Data Management

Existing original recordings and earlier paired datasets have been preserved.

Relevant paths:

- Original OpenSLR speech: `openslr_slr66/`
- Original DEMAND noise: `DEMAND/extracted/`
- Original VAANI recordings: `vaani_telugu_transcribed/`
- Previous pilot dataset: `paired_dataset_v3_pilot/`
- Final generated dataset: `paired_dataset_v3/`
- Source manifest: `paired_dataset_v3_manifest_speaker_balanced.csv`

The v3 generation and audit procedures did not intentionally modify the original source recordings or the existing v1/v2 datasets.

**Reproducibility artifacts in v3:**

- `metadata.csv`
- `build_config.json`
- `progress.jsonl`
- `dataset_report.json`

These files record source mappings, split assignments, mixing parameters, the manifest fingerprint, and generation status.

## 8. Model Development — Planned

The research direction currently includes:

- CNN/ZF-Net-based feature extraction.
- Gaussian Mixture Model (GMM) components.
- Noise suppression and speech reconstruction.
- Short-time Fourier transform (STFT) and inverse STFT as potential signal-processing stages.

The exact architecture, feature representation, training objective, and integration of ZF-Net and GMM still require a clear technical specification and experimental justification.

No model training has been performed, and no enhancement-quality claims have been established.

## 9. Current Status

### Completed

- [x] Established the initial project data directory in Google Drive.
- [x] Extracted and inspected the selected DEMAND noise categories.
- [x] Inspected OpenSLR SLR66 archives and transcript indexes.
- [x] Reviewed 30 original-versus-trimmed recordings.
- [x] Selected 500 clean-speech sources and prepared the source manifest.
- [x] Created a speaker-separated 80/10/10 source split.
- [x] Generated 500 clean targets and 1,000 noisy mixtures.
- [x] Generated mixtures at 5 dB and 10 dB SNR.
- [x] Distributed mixtures evenly across five DEMAND environments.
- [x] Validated the pilot dataset.
