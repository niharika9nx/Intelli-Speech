# Intelli-Speech Dataset Structure and Validation Report

**Last updated:** 2026-10-25  
**Dataset version:** v3  
**Status:** Generated, independently audited, and reviewed through listening tests  
**Purpose:** Document the paired Telugu speech dataset prepared for speech-enhancement research.

## 1. Dataset Overview

The Intelli-Speech v3 dataset contains Telugu speech recordings derived from OpenSLR SLR66 and controlled noisy mixtures created using environmental noise from DEMAND.

Each selected speech source has:
- One clean speech target.
- One noisy mixture at a target SNR of 5 dB.
- One noisy mixture at a target SNR of 10 dB.

The dataset is intended for developing and evaluating a Telugu speech-enhancement system. It has not yet been used to train a model.

### Storage locations

**Base directory**

`/content/drive/MyDrive/Intelli-Speech-data`

**Generated dataset**

`/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3`

**Source manifest**

`/content/drive/MyDrive/Intelli-Speech-data/paired_dataset_v3_manifest_speaker_balanced.csv`

## 2. Dataset Composition

| Property | Verified value |
|---|---:|
| Unique speech sources | 500 |
| Clean WAV files | 500 |
| Noisy WAV files | 1,000 |
| Total WAV files | 1,500 |
| Sample rate | 16,000 Hz |
| Audio channels | Mono |
| Noisy variants per source | 2 |
| Target SNR levels | 5 dB and 10 dB |

### Source distribution

| Attribute | Count |
|---|---:|
| Female sources | 250 |
| Male sources | 250 |
| Total sources | 500 |

The gender labels originate from the source archives. They are metadata labels and should not be interpreted as independently verified demographic information.

## 3. Source Datasets

### 3.1 OpenSLR SLR66

**Location:** `openslr_slr66/`

The source corpus contains 4,448 recordings:
- Female archive: 2,294 recordings.
- Male archive: 2,154 recordings.

The associated transcript index files are:
- `line_index_female.tsv`
- `line_index_male.tsv`

A selected subset of 500 recordings was used to build v3. The source manifest records the archive, audio member, transcript, candidate speaker ID, and split assignment.

The complete 4,448-recording corpus has not undergone comprehensive quality validation.

### 3.2 DEMAND

**Location:** `DEMAND/extracted/`

Five environmental noise categories were used:
- `DKITCHEN`
- `NPARK`
- `OOFFICE`
- `STRAFFIC`
- `TBUS`

The available source collection contains 80 recordings in total, with 16 recordings per category. The inspected recordings are mono, sampled at 16 kHz, and 300 seconds long.

Original DEMAND recordings were used for v3 noise mixing.

### 3.3 VAANI Telugu

**Location:** `vaani_telugu_transcribed/`

Approximately 246 recordings and a metadata file are available.

Initial listening indicates that these recordings contain speech with background noise. They were not used as clean targets in v3 because they have not been established as clean speech or perfectly aligned noisy-clean pairs.

They may be considered for a separate real-world noisy-speech evaluation after suitability checks.

## 4. Preprocessing

### 4.1 Silence trimming

Leading and trailing silence was trimmed using `librosa.effects.split` with these parameters:

- `top_db = 30`
- `frame_length = 2048`
- `hop_length = 512`
- Padding: 150 ms before the first detected active region and after the last detected active region.

The resulting audio retains one continuous segment between the first and last detected active regions. Internal pauses are preserved.

The procedure was manually reviewed on 30 recordings: 15 female and 15 male.

The reviewed samples sounded satisfactory. However, this sample-based review does not prove that every recording is free of clipped words or incorrectly removed quiet speech.

### 4.2 Sample rate and channels

Selected source recordings were converted to mono where necessary and resampled to 16 kHz to match the DEMAND noise recordings.

The generated clean and noisy files use a 16 kHz sample rate.

### 4.3 Noise mixing

Each clean speech target was mixed with a segment of environmental noise to create two noisy variants.

The target SNR values were:
- 5 dB.
- 10 dB.

A common gain was applied to the clean target and both mixtures when necessary to keep peaks below the configured clipping threshold while preserving their relative SNR.

## 5. Directory Structure

The generated dataset contains the following verified components:

```text
paired_dataset_v3/
├── clean/
│   └── 500 clean target WAV files
├── noisy/
│   └── 1,000 noisy mixture WAV files
├── metadata.csv
├── build_config.json
├── progress.jsonl
└── dataset_report.json
```

The directory descriptions above indicate the contents and roles of each component; the exact filenames within the audio directories are recorded by the dataset metadata.

### Metadata

`metadata.csv` contains 1,000 rows, one per noisy mixture. It links each noisy input to its clean target and records the source and mixing information needed for reproducibility.

The independent audit checked the metadata paths, source mappings, split assignments, SNR values, and consistency with the source manifest.

The source manifest contains these columns:

```text
source_id
gender
archive
audio_member
transcript
sample_rate
trimmed_duration_sec
split
speaker_id
```

The manifest is stored separately from the generated audio dataset.

## 6. Train, Validation, and Test Splits

The 500 selected sources were divided into an 80/10/10 split.

| Split | Clean sources | Noisy mixtures |
|---|---:|---:|
| Training | 400 | 800 |
| Validation | 50 | 100 |
| Test | 50 | 100 |
| **Total** | **500** | **1,000** |

Both noisy variants of a source inherit that source's split assignment. This prevents the same source recording from appearing in multiple splits through its different SNR variants.

The independent audit confirmed that the candidate speaker IDs in the manifest do not overlap across the three splits.

**Limitation:** Candidate speaker IDs were parsed from the source filename convention. They have not been independently verified as real-world speaker identities.

## 7. Noise Distribution

The dataset contains 200 noisy mixtures per environmental category.

| Noise category | Mixtures |
|---|---:|
| DKITCHEN | 200 |
| NPARK | 200 |
| OOFFICE | 200 |
| STRAFFIC | 200 |
| TBUS | 200 |
| **Total** | **1,000** |

The dataset is balanced by noise category overall. This does not imply that every split or speaker group has identical representation across all noise categories.

## 8. Validation Results

### 8.1 Independent audit

An independent read-only audit inspected all 1,000 saved clean/noisy pairs.

**Result: 21 of 21 checks passed.**

The checks covered:
- Clean and noisy file counts.
- Metadata row counts and unique source IDs.
- Clean-target and noisy-mixture mappings.
- Exactly two noisy variants per source.
- Sample rate, waveform length, and finite sample values.
- SNR accuracy and peak levels.
- Source manifest consistency.
- Split assignments and candidate speaker separation.
- Noise category counts.
- Metadata paths and manifest fingerprint.

### 8.2 SNR accuracy

| Statistic | Absolute SNR error |
|---|---:|
| Mean | 0.0000251 dB |
| Median | 0.0000162 dB |
| Maximum | 0.0003905 dB |

The maximum error is substantially below the audit tolerance of 0.2 dB.

### 8.3 Duration statistics

Across the 1,000 clean/noisy pairs:

| Statistic | Duration |
|---|---:|
| Mean | 3.076 seconds |
| Minimum | 0.908 seconds |
| Maximum | 8.673 seconds |

### 8.4 Peak levels

- Maximum clean-target peak: approximately 0.9092.
- Maximum noisy-mixture peak: approximately 0.9901.
- No full-scale clipping was detected in the audited WAV files.

### 8.5 Listening review

Representative audio files were manually reviewed after generation. The generated mixtures sounded satisfactory during this review.

This qualitative check complements the numerical audit. It does not establish that all 1,000 mixtures are perceptually ideal or that a future model will improve speech intelligibility.

## 9. Reproducibility

The dataset includes:
- `metadata.csv`
- `build_config.json`
- `progress.jsonl`
- `dataset_report.json`

The source manifest SHA-256 recorded for this dataset is:

`2af46ef5c9a4545c0775f08a07b734bc0696a888339a18d5e40cdd1103941eb0`

These artifacts preserve the recorded source mappings, split assignments, mixing configuration, generation progress, and report information.

## 10. Known Limitations

1. Only 500 of the 4,448 available OpenSLR recordings were selected for v3.
2. The full source corpus has not undergone comprehensive quality validation.
3. The trimming method was manually reviewed on a sample, not every source recording.
4. Candidate speaker separation depends on filename-derived identifiers.
5. The dataset uses controlled synthetic noise mixtures; performance on naturally noisy speech remains untested.
6. VAANI recordings have not been established as aligned noisy-clean pairs.
7. SNR accuracy and listening checks do not establish speech-enhancement effectiveness.
8. No model has been trained or evaluated on this dataset yet.

## 11. Current Status and Next Steps

### Completed

- Generated 500 clean speech targets.
- Generated 1,000 noisy mixtures at 5 dB and 10 dB.
- Included five DEMAND noise categories.
- Created source-level train, validation, and test splits.
- Passed all 21 independent audit checks.
- Completed a representative listening review.

### Remaining

- Commit this report to the project repository.
- Define the speech-enhancement objective and evaluation protocol.
- Select an appropriate baseline and objective metrics.
- Finalize the model architecture only after the objective is specified.
- Train and evaluate a baseline after the data and research protocol have been reviewed.

**Current conclusion:** The v3 dataset has passed the implemented structural and numerical checks, and representative audio has passed the listening review. It is ready for the next stage of research planning, but model effectiveness has not yet been established.

