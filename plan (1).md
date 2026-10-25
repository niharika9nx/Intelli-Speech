# Intelli-Speech — Project Plan

## 1. Project Objective

Develop a speech-enhancement research pipeline for noisy Telugu speech, intended to support a voice-aid prototype.

The proposed research direction involves:

- Short-Time Fourier Transform (STFT) for time-frequency representation.
- ZF-Net-based deep learning architecture / CNN for feature learning or speech enhancement.
- Gaussian Mixture Model (GMM) for a role that must be established from the research design.
- Inverse Short-Time Fourier Transform (iSTFT) for waveform reconstruction.
- Evaluation of enhancement quality and intelligibility.

The implementation must follow the research requirements and the actual properties of the available dataset. Do not assume that the proposed components can be connected directly without investigating their interfaces and purposes.

## 2. Development Environment and Storage

### Antigravity — Local Development

- Primary environment for writing, reviewing, and organizing code.
- Use GitHub Copilot and OpenCode/DeepSeek agents where appropriate.
- Agents can inspect repository files and documentation locally.
- Agents must not assume they can directly access the Google Drive dataset from the local Windows environment.
- Agents should use verified dataset documentation when designing code that depends on the dataset.

### GitHub — Version Control

Repository: `https://github.com/niharika9nx/Intelli-Speech`

Branch: `main`

GitHub stores:
- Source code and configuration.
- Jupyter notebooks.
- Research and implementation documentation.
- Dataset structure reports and sanitized metadata summaries.
- Tests and small, reproducible examples where appropriate.

Do not commit:
- The full audio dataset.
- Private or sensitive transcript data unnecessarily.
- Credentials, access tokens, or secrets.
- Large generated audio files, checkpoints, or experiment outputs.

### Google Colab — Execution Environment

Colab is the primary environment for:
- Inspecting the real dataset.
- Executing notebook code.
- Preprocessing and feature extraction.
- Training and evaluating models.
- Running experiments and smoke tests.

The notebook must be synchronized with the latest GitHub version before execution.

### Google Drive — Persistent Data Storage

The existing dataset is stored at:

`/content/drive/MyDrive/Intelli-Speech-data/vaani_telugu_transcribed`

This path is available in Colab after Google Drive is mounted. It is not a path on the local Windows machine.

Google Drive is also the preferred persistent location for generated features, checkpoints, enhanced audio, and experiment outputs.

Do not redownload the full dataset, duplicate it unnecessarily, or modify or delete the original dataset files.

## 3. Mandatory Phase 1 — Dataset Reconnaissance

**No major preprocessing or model implementation should begin until the available dataset has been inspected and documented.**

The agents may write inspection utilities, but the actual inspection must run in Colab against the existing Google Drive dataset.

### 3.1 Inspect the Dataset Structure

Establish the actual directory hierarchy, including:
- Folder and subfolder names.
- File counts by directory and extension.
- Audio files and metadata files.
- Nested or unexpected files.
- Empty directories, if any.

Distinguish audio samples from metadata, documentation, and unrelated files. Do not assume every file is an audio sample.

### 3.2 Inspect Metadata

Determine:
- Metadata file format and encoding.
- Column names and data types.
- Sample identifiers and filename conventions.
- Audio paths or references.
- Transcript fields and language information, if present.
- Speaker identifiers, splits, labels, or other attributes, if present.
- Missing values, duplicate identifiers, and invalid references.

Do not assume a particular schema before inspecting the files.

### 3.3 Inspect Audio Properties

For a representative sample and, where practical, the complete dataset, establish:
- Audio format and encoding.
- Sample rates.
- Number of channels.
- Duration distribution.
- Empty, unreadable, or corrupted audio files.
- Whether audio can be decoded consistently.

Report the inspection method and distinguish complete-dataset statistics from sample-based observations.

### 3.4 Verify Audio–Transcript Relationships

Establish whether transcripts are available and how they map to audio samples.

Check:
- Whether referenced audio files exist.
- Whether audio files have corresponding transcripts.
- Whether identifiers or filenames can be matched reliably.
- Whether duplicate or conflicting mappings exist.
- Whether the data supports the intended training objective.

Do not invent missing transcripts, infer target labels, or assume that the dataset contains clean/noisy audio pairs.

### 3.5 Produce Dataset Documentation

Create:

`docs/dataset_structure.md`

The report must contain:
1. Dataset location and inspection date.
2. Verified directory structure.
3. File counts and file types.
4. Metadata schema and representative field descriptions.
5. Audio characteristics and summary statistics.
6. Transcript availability and mapping method.
7. Missing files, duplicates, and data-quality issues.
8. Unknowns and limitations.
9. Implications for preprocessing and model design.

Where useful, create a sanitized machine-readable summary such as `docs/dataset_manifest.json` or a metadata schema document.

Do not publish the full transcripts or sensitive sample-level data unnecessarily.

### Phase 1 completion criteria

Phase 1 is complete only when:
- The real dataset has been inspected in Colab.
- The report reflects observed facts rather than assumptions.
- Audio and metadata relationships have been checked.
- Important unknowns and data-quality issues are documented.
- The report has been committed to GitHub.

If the dataset does not support a required training objective, report the gap before proceeding.

## 4. Phase 2 — Research and Architecture Design

After the dataset report is available, inspect the research requirements and decide how the pipeline should work.

### 4.1 Define the Enhancement Objective

Determine what the model is expected to receive and produce.

Investigate:
- Whether the available data contains noisy and clean speech pairs.
- Whether noisy samples must be synthetically generated.
- Whether clean reference recordings are available.
- Whether training requires aligned waveform pairs, spectral targets, masks, or other representations.
- How training, validation, and testing will be separated.

If the available dataset cannot directly support supervised speech enhancement, document viable alternatives and their limitations before implementation.

### 4.2 Define the Pipeline Interfaces

Establish the input and output expected at each stage.

**STFT**
- Define waveform preprocessing, sample-rate handling, window size, hop length, and representation.
- Determine whether the model operates on magnitude, complex spectra, masks, or another representation.

**ZF-Net / CNN**
- Establish the intended architecture from the research requirements and relevant references.
- Define input tensor shape, output tensor shape, target representation, loss function, and training strategy.
- Do not assume a conventional CNN implementation is equivalent to the proposed ZF-Net architecture.

**GMM**
- Establish the specific role of the GMM in the proposed system.
- Determine which features it consumes, how it is trained, and how its output interacts with the neural network.
- Do not add a GMM merely because it appears in the proposed pipeline.

**iSTFT**
- Define how the processed representation will be converted back to a waveform.
- Ensure compatibility with the chosen STFT configuration and output representation.

Document unresolved architectural decisions and request clarification before making consequential assumptions.

## 5. Phase 3 — Implementation

Implement the pipeline incrementally, with each stage independently testable.

Recommended order:
1. Repository and environment checks.
2. Dataset loader and metadata validation.
3. Audio decoding and preprocessing.
4. STFT and inverse-STFT round-trip test.
5. Baseline processing and evaluation.
6. ZF-Net/CNN implementation.
7. GMM integration, once its role is established.
8. End-to-end enhancement pipeline.
9. Evaluation and reproducibility checks.

Use modular functions and clear configuration variables. Avoid hard-coded machine-specific paths where practical.

Keep the original dataset unchanged. Save derived data and outputs to designated locations in Drive.

## 6. Phase 4 — Evaluation

Before expensive training, run a small end-to-end smoke test.

Evaluation should consider:
- Correctness of audio loading and reconstruction.
- Numerical stability and tensor shapes.
- Training and validation behavior.
- Appropriate objective speech-enhancement metrics, selected according to the task and available references.
- Listening tests comparing input and enhanced audio.
- Baseline comparisons.
- Generalization to held-out samples.

Keep training, validation, and test data separate. Prevent speaker or sample leakage where relevant.

Do not report improvement unless supported by actual evaluation results.

## 7. Agent Operating Rules

All agents must follow these rules:

1. **Inspect before implementing.** Read the repository and relevant documentation before proposing changes.
2. **Use verified dataset facts.** Read `docs/dataset_structure.md` before writing dataset-dependent preprocessing or model code.
3. **Do not assume local dataset access.** The actual dataset is in Google Drive and is inspected and executed against in Colab.
4. **Separate coding from execution.** Agents may develop code locally, but dataset-dependent behavior must be validated in Colab.
5. **Do not invent dataset properties.** Mark unverified assumptions and request inspection or clarification.
6. **Avoid premature architecture decisions.** Establish the research objective and component interfaces before implementing the full model.
7. **Work incrementally.** Prefer small, testable changes over broad refactors.
8. **Preserve existing work.** Do not delete or overwrite project files, dataset files, or notebook content without explicit authorization.
9. **Explain significant changes.** State the intended modifications and affected files before broad changes.
10. **Validate notebook structure.** Run notebook-format validation before committing notebook changes.
11. **Avoid simultaneous notebook edits.** Treat GitHub as the synchronization point. Avoid editing the same notebook in Antigravity and Colab at the same time.
12. **Control compute usage.** Do not start expensive training until the data pipeline and a small smoke test succeed.
13. **Record reproducibility details.** Document configuration, random seeds where applicable, dependencies, and experiment outputs.
14. **Never fabricate results.** Clearly distinguish implemented, tested, and proposed functionality.

## 8. Git and Experiment Workflow

Use this workflow consistently:

1. Inspect or edit code in Antigravity.
2. Review the diff and validate modified notebooks.
3. Commit and push intended code and documentation changes to GitHub.
4. In Colab, enter the cloned repository and pull the latest changes.
5. Mount Google Drive and verify required paths.
6. Execute the notebook or scripts in Colab.
7. Save checkpoints and experiment outputs to Drive.
8. Record meaningful findings in version-controlled documentation.
9. If code changes are made in Colab, synchronize them deliberately and resolve conflicts by inspecting both versions.

Never assume that `/content` itself is a Git repository. Run Git commands from the cloned repository directory.

## 9. Immediate Next Task

**The first task is dataset reconnaissance, not model training.**

The assigned agent should:
1. Inspect the existing repository and notebook without modifying unrelated files.
2. Prepare a Colab-compatible dataset inspection script or notebook section.
3. Run the inspection against the existing Google Drive dataset.
4. Summarize the real directory structure, metadata, audio characteristics, and data-quality issues.
5. Create `docs/dataset_structure.md` from the observed results.
6. Identify the information still needed to determine the training objective and model interfaces.
7. Stop before major model implementation until the findings have been reviewed.

The output of this task is a verified dataset report and an evidence-based implementation plan—not a prematurely constructed model.
