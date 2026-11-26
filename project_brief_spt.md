# Intelli-Speech — Complete Project Context & Technical Direction

**Project:** Intelli-Speech: Voice Aid System for Speech-Impaired People  
**Institution:** Shri Vishnu Engineering College for Women, Bhimavaram, Andhra Pradesh  
**Project duration in official proposal:** 36 months  
**Primary technical direction:** Hybrid Deep Neural ZF-Network (HDNZF-Net) for speech enhancement  
**Current research focus:** Telugu atypical speech analysis, controlled synthetic data generation, intelligibility improvement and speech reconstruction.

---

# 1. Project Background and Official Scope

## 1.1 Official project proposal

The official administration proposal is titled:

**Intelli-Speech — Voice Aid System for Speech-Impaired People.**

It was submitted under the TIDE proposal by Shri Vishnu Engineering College for Women.

The proposal identifies speech impairment as a significant barrier to effective communication and social participation.

The project targets two broad groups:

1. Elderly individuals.
2. Persons with Disabilities (PwD).

The proposal particularly discusses communication difficulties associated with ageing, hearing loss, weakened vocal cords and speech difficulties.

## 1.2 Official objectives

The administration's proposal identifies the following objectives:

### A. Compensating for age-related hearing loss

Develop assistive solutions that help elderly individuals hear and understand speech more clearly.

### B. Improving communication in noisy environments

Reduce background noise and improve speech audibility and intelligibility.

### C. Enhancing voice clarity

Improve the clarity and audibility of speech for individuals experiencing weakened vocal cords or speech difficulties.

### D. Facilitating social interaction

Enable elderly individuals and people with speech difficulties to communicate more effectively with family, friends and caregivers.

## 1.3 Identified gaps in existing technology

The official proposal identifies several gaps:

- Personalization and adaptability to individual speech patterns.
- Real-time adaptation to changing speech difficulties.
- Noise robustness.
- Affordability and accessibility.

The proposal emphasizes inclusivity, independence, dignity, social interaction and quality of life.

## 1.4 Existing state of the art

The proposal discusses:

- Augmentative and Alternative Communication (AAC) devices.
- Communication applications.
- Note-taking applications.
- AAC boards and picture exchange systems.
- Text-to-speech software.

It identifies the novelty of the proposed project as a hybrid deep-learning-based assistive speech enhancement solution, with an application as the intended output.

## 1.5 Official implementation timeline

The original proposal outlines a 36-month project.

| Phase | Official planned activities |
|---|---|
| Q1–Q2 | Literature collection, study of conventional and modern speech enhancement techniques |
| Q3–Q4 | Development of conventional speech enhancement techniques |
| Q5–Q6 | Development using the proposed HDNZF-Net |
| Q7–Q8 | Research paper publication |
| Q11 | Prototype development and testing |
| Q12 | Review, modifications and documentation |

The expected deliverables include model development, a software-based prototype, testing and research publications.

**Important:** This official proposal establishes the project's original mandate. The research direction described in this document is a technical refinement of that mandate, not a replacement for the administration's original commitments.

---

# 2. My Proposed Technical Architecture

I designed a separate three-page technical architecture specification based on the project's requirements.

This is my proposed implementation design. It should be treated as an architecture to investigate and validate, not as an already scientifically established or validated model.

The proposed architecture is called:

**Hybrid Deep Neural ZF-Network (HDNZF-Net).**

The initial conceptual pipeline is:

```text
Input Audio
    |
    v
STFT
    |
    v
ZF-Net
(CNN-based feature extraction)
    |
    v
DNN
(Acoustic relationship modelling)
    |
    v
GMM
(Probabilistic modelling and mask generation)
    |
    v
Spectrogram Masking
    |
    v
iSTFT
    |
    v
Enhanced Speech
```

## 2.1 Audio preprocessing and STFT

The system receives a one-dimensional audio waveform.

The waveform is converted into a time-frequency representation using the Short-Time Fourier Transform (STFT).

The technical specification initially proposes:

- Frame duration: approximately 20–30 ms.
- Window: Hamming window.
- Overlap: 50%.
- Representation: magnitude spectrogram, with phase retained for reconstruction.

The spectrogram provides a time-frequency representation that allows the model to examine speech characteristics.

## 2.2 ZF-Net — Acoustic feature extraction

The proposed ZF-Net component is based on convolutional feature extraction.

Its purpose is to learn hierarchical representations from the spectrogram.

The original technical specification proposes:

- Initial convolutional layer with 7 × 7 filters.
- Stride of 2.
- Convolutional feature extraction.
- Pooling.
- Deep feature representation.

The intention is to learn useful acoustic patterns at different levels of representation.

However, the proposed convolutional structure and its suitability for speech enhancement must be experimentally evaluated.

## 2.3 DNN — Acoustic relationship modelling

The proposed DNN receives features extracted by ZF-Net.

Its intended role is to learn relationships among acoustic representations and produce a representation useful for the subsequent probabilistic modelling stage.

The exact structure, temporal context, output dimensions and training objective need to be established through research.

## 2.4 GMM — Probabilistic modelling

The original technical specification proposes a Gaussian Mixture Model to model statistical distributions in the learned acoustic representation.

The original idea is to distinguish target speech from unwanted components and generate a probabilistic mask.

The mask is intended to avoid excessively aggressive binary suppression.

The original specification describes an Ideal Ratio Mask (IRM) bounded between 0 and 1.

However, the original GMM formulation was written in the context of distinguishing clean speech from background noise.

**Our current research direction requires investigating how GMM should be adapted to atypical speech characteristics instead of simply reusing the environmental noise formulation.**

This distinction is fundamental.

## 2.5 Reconstruction

The original technical design proposes applying the generated mask to the magnitude spectrogram and using the retained phase for inverse STFT reconstruction.

The output is a reconstructed time-domain waveform.

The reconstruction approach must be evaluated against alternatives, including magnitude prediction, complex-domain processing and waveform-based approaches.

---

# 3. Current Research Direction

The current research direction is more specific than the initial general speech enhancement concept.

We are investigating:

**How can we identify acoustic characteristics present in atypical Telugu speech, reproduce selected characteristics in a controlled synthetic dataset, and develop a speech processing model that improves intelligibility while preserving linguistic content?**

This is the central research question guiding the present technical work.

The project should not assume that all atypical speech is caused by one condition or exhibits one common pattern.

We are interested in generalized atypical speech characteristics, not diagnosing individuals with specific medical conditions.

## 3.1 Why Telugu?

We have decided to focus on Telugu for the current research and prototype.

This decision is based on the availability of Telugu atypical speech recordings and the need to develop and validate a language-focused pipeline.

We should not combine Hindi, Marathi and Telugu recordings indiscriminately to generate Telugu atypical speech patterns.

Phonetics, articulation, pronunciation and language-specific acoustic characteristics differ.

Cross-language transfer may be investigated separately in future work, but it must not be assumed valid for our present Telugu synthetic dataset.

---

# 4. Real Atypical Telugu Dataset

## 4.1 Primary dataset

Dataset: **ARTPARK-IISc Vaani Atypical Speech Corpus**

Official Hugging Face page:

https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Atypical-Speech-Corpus

Official project page:

https://vaani.iisc.ac.in/atypical-speech

The dataset contains:

| Language | Recordings | Speakers |
|---|---:|---:|
| Hindi | 688 | 25 |
| Marathi | 290 | 8 |
| Telugu | 189 | 7 |
| **Total** | **1,167** | **40** |

Our current research uses only the Telugu subset.

### Current Telugu dataset facts

- 189 real atypical speech recordings.
- 7 Telugu speakers.
- Approximately 109.1 minutes of audio.
- Approximately 34.6 seconds average duration per recording.
- WAV audio, 16 kHz, mono.
- Includes transcription and verbatim transcription fields.
- Includes a difficulty level field.

The difficulty field contains Low, Medium and High ratings.

It represents an understandability or quality-control assessment. It must not be treated as a clinical severity label.

The dataset contains speech from individuals with diverse characteristics and conditions. However, we do not have reliable speaker-level clinical diagnoses for every recording.

Therefore:

- Do not assign specific clinical disorders to speakers without evidence.
- Do not assume every speaker exhibits the same speech impairment.
- Do not treat the seven speakers as seven representatives of all atypical Telugu speech.
- Distinguish intended transcription from verbatim transcription.
- Do not interpret the difficulty rating as a medical diagnosis or clinical severity measurement.

## 4.2 Why the 189 recordings matter

The 189 recordings are our primary real-world reference for discovering atypical Telugu speech characteristics.

We should not immediately train a large model on them or generate arbitrary distorted audio.

First, we must understand what is actually present in these recordings.

Our research must identify:

1. Which acoustic characteristics are observable.
2. Which characteristics recur across multiple speakers.
3. Which characteristics are specific to individual speakers.
4. Which characteristics may be explained by recording conditions, background noise, speaking style or linguistic content.
5. Which characteristics can be reproduced through controlled synthetic transformations.

This analysis must precede synthetic data generation.

---

# 5. Typical Telugu Speech Dataset

We also have typical Telugu speech recordings that may be used as source material for controlled synthetic data generation.

### Existing data sources

**A. Vaani Telugu typical speech**

A typical Telugu subset from the regular Vaani dataset has been downloaded.

Current Drive location:

```text
Intelli-Speech-data/
    vaani_telugu_transcribed/
```

The latest known verification showed approximately 247 audio files and one metadata file.

This count must be rechecked before experimentation.

**B. SLR66 Telugu**

A Telugu speech dataset containing approximately 4,448 recordings has previously been used in the project.

Its actual contents, speaker information, recording quality, transcripts, licensing and suitability must be audited before use.

### Important requirement

Do not automatically classify every typical-dataset recording as clean speech.

Audit:

- Speaker distribution.
- Recording conditions.
- Audio quality.
- Sample rate and channel count.
- Transcription availability and accuracy.
- Duration distribution.
- Duplicate recordings.
- Licensing and permitted usage.
- Potential background noise and existing speech irregularities.

Typical speech is intended to provide source material and a comparison baseline, not an unquestioned ground truth.

---

# 6. Acoustic Analysis of Real Atypical Telugu Speech

This is one of the most important stages of the project.

Before generating synthetic atypical speech, analyse the 189 real Telugu recordings.

The purpose is to discover patterns supported by actual recordings.

Potential measurements include the following.

## 6.1 Temporal characteristics

Investigate:

- Speaking rate.
- Pause frequency and duration.
- Duration variability.
- Syllable duration.
- Phoneme duration, where reliable alignment is possible.
- Repetitions and prolongations, where they are genuinely present.
- Timing irregularities.
- Changes in speech rhythm.

## 6.2 Prosodic characteristics

Investigate:

- Fundamental frequency (F0).
- Pitch range.
- Pitch variability.
- Energy and intensity contours.
- Rhythm.
- Stress patterns.
- Prosodic irregularities.
- Intonation characteristics.

## 6.3 Spectral and voice characteristics

Potential measurements:

- MFCCs.
- Spectral centroid.
- Spectral rolloff.
- Spectral flux.
- Harmonicity.
- Voicing characteristics.
- Formant frequencies, where reliable.
- Other suitable voice-quality measurements.

## 6.4 Articulatory and phonetic characteristics

Where the data supports it, investigate:

- Consonant and vowel duration.
- Phoneme-level variations.
- Pronunciation consistency.
- Repeated or extended speech segments.
- Vowel formant patterns.
- Consonant articulation characteristics.

Do not claim that a particular acoustic measurement directly proves a specific medical condition.

## 6.5 Cross-speaker analysis

There are only seven Telugu speakers.

Consequently, we must distinguish:

- Patterns appearing across multiple speakers.
- Patterns specific to one speaker.
- Patterns appearing only in particular utterances.
- Patterns potentially caused by recording artefacts.

A characteristic observed in one recording must not automatically become a general atypical Telugu speech characteristic.

We need speaker-level statistical analysis, visualization and careful interpretation.

---

# 7. Synthetic Atypical Telugu Dataset Generation

## 7.1 Why synthetic data?

The current real atypical Telugu dataset is small.

We have 189 recordings from seven speakers.

Training a sufficiently generalizable model directly from this limited dataset is difficult.

Synthetic data may help us investigate and train models using more controlled examples.

However, synthetic data must be grounded in actual acoustic characteristics observed in the real Telugu recordings.

**We must not introduce arbitrary distortions simply to increase the dataset size.**

## 7.2 Intended generation strategy

The proposed approach is:

```text
Real Atypical Telugu Speech
             |
             v
Acoustic Analysis
             |
             v
Identify Supported Characteristics
             |
             v
Determine Reproducible Characteristics
             |
             v
Select Typical Telugu Speech
             |
             v
Apply Controlled Transformations
             |
             v
Synthetic Atypical Telugu Speech
             |
             v
Validate Against Held-Out Real Speech
```

## 7.3 Potential transformations to investigate

The following are candidates, not predetermined requirements:

- Controlled speaking-rate modification.
- Pause insertion or modification.
- Duration modification.
- Pitch and energy variation.
- Prosodic modification.
- Segment-level transformations.
- Phoneme-level transformations, if reliable alignment is available.
- TTS-based atypical speech synthesis.
- Healthy-speaker emulation.
- Neural speech generation or voice conversion, if suitable Telugu support exists.

For every transformation, establish:

1. Which real acoustic characteristic justifies it?
2. How was that characteristic measured?
3. What transformation method can reproduce it?
4. What parameters and intensity levels are appropriate?
5. Does it preserve the original linguistic content?
6. Does the resulting speech resemble the real recordings?
7. Does it introduce unrealistic artefacts?

## 7.4 Synthetic data traceability

Every generated recording must have metadata containing:

- Source audio ID.
- Source speaker ID, where available.
- Transformation type.
- Transformation parameters.
- Rationale for applying the transformation.
- Transformation strength.
- Generated audio ID.
- Generation method and software version.
- Dataset split assignment.

Synthetic descendants of a source recording must remain in the same train, validation or test partition as that source.

Never generate training examples from held-out evaluation speakers.

## 7.5 Critical limitation

Synthetic atypical speech is not genuine clinical speech.

Increasing the number of generated audio files does not increase the number of independent human speakers.

We must report real and synthetic dataset sizes separately.

Synthetic difficulty or intelligibility labels must not be inherited automatically from the original recording. They require a justified labelling procedure or human assessment.

---

# 8. GMM and ZF-Net: Intended New Research Roles

This section is especially important.

The existing implementation and the intended future architecture must not be confused.

## 8.1 GMM — Atypical speech acoustic modelling

Our intended direction is to investigate the use of GMM to model statistical distributions of atypical speech characteristics.

Possible roles include:

- Modelling distributions of acoustic features.
- Representing groups of recurring acoustic patterns.
- Learning statistical relationships in acoustic embeddings.
- Providing probabilistic information to a downstream enhancement model.
- Potentially conditioning ZF-Net using the learned pattern representation.

One possibility is using GMM-derived information with Feature-wise Linear Modulation (FiLM).

However, this is a research hypothesis, not a finalized architecture.

We must investigate whether GMM is useful for this task, what features it should model, how its components should be interpreted and whether it improves the enhancement model.

Do not automatically describe GMM as an atypical speech classifier.

There are no reliable clinical diagnosis labels for supervised clinical classification.

## 8.2 ZF-Net — Atypical speech enhancement

The intended role of ZF-Net is to learn useful acoustic representations and support speech enhancement or reconstruction.

The model should be investigated for its ability to process atypical Telugu speech and improve intelligibility.

The target is not simply environmental noise suppression.

We need to determine:

- What representation should ZF-Net receive?
- Should it process magnitude spectrograms, complex spectrograms or another representation?
- What should it predict?
- Should it estimate a mask, enhanced magnitude, residual or reconstructed waveform?
- What training objective is suitable?
- How should temporal context be represented?
- How can GMM information condition the network?
- What safeguards are required to preserve speech content and speaker characteristics?

The original ZF-Net specification provides a starting architecture, but its design must be validated for the current task.

## 8.3 Potential paired synthetic training

A possible experimental design is:

```text
Typical Telugu Source Audio
             |
             +---------------------> Reference Target
             |
             v
Controlled Atypical Transformation
             |
             v
Synthetic Atypical Audio
             |
             v
GMM + ZF-Net
             |
             v
Reconstructed / Enhanced Audio
```

In this experiment:

- Input: synthetically transformed atypical Telugu speech.
- Target: original typical Telugu source recording.

This creates a paired training example.

The model would learn a mapping from simulated atypical speech towards a more intelligible version of the original utterance.

However, this is only a proxy training setup.

It does not provide genuine paired clean references for real atypical speakers.

It may also remove legitimate speaker-specific characteristics or produce speech that is more typical but not necessarily more appropriate for the original speaker.

These risks must be investigated.

---

# 9. Existing Implementation — What Has Already Been Built

We previously developed Version 1 of the Intelli-Speech pipeline.

It is important to understand that this implementation was initially designed for environmental noise suppression.

It has not yet been validated as an atypical speech enhancement model.

## 9.1 Existing components

- Audio preprocessing and I/O.
- STFT and inverse STFT.
- GMM environmental noise classifier.
- ZF-Net-based time-frequency mask prediction.
- GMM-conditioned feature modulation.
- Wiener filtering baseline.
- Oracle Ideal Ratio Mask baseline.
- FastAPI backend.
- React frontend with audio upload flow.
- Experiment logging and testing infrastructure.

## 9.2 Original Version 1 pipeline

```text
Noisy Speech
     |
     v
STFT
     |
     +-------------------------+
     |                         |
     v                         v
Log-Mel Features          Log-Magnitude
     |                         |
     v                         v
GMM Noise Classifier       ZF-Net
     |                         |
     +------> Conditioning <---+
                 |
                 v
            Predicted Mask
                 |
                 v
      Mask × Noisy Spectrogram
                 |
                 v
              iSTFT
                 |
                 v
         Enhanced Audio
```

The previous implementation used:

- Five DEMAND environmental noise classes.
- STFT with n_fft = 512 and hop_length = 128.
- Log-mel features for GMM.
- Log-magnitude features for ZF-Net.
- FiLM-style conditioning.
- Noisy phase retained for reconstruction.

## 9.3 Existing dataset for Version 1

The previous `paired_dataset_v3` contains:

- 500 SLR66 clean Telugu clips.
- 1,000 DEMAND environmental noise mixtures.

This dataset was constructed for environmental noise enhancement.

It does not contain atypical speech labels or paired real atypical/clean Telugu speech.

Therefore, the old model cannot be claimed to understand or enhance atypical speech.

## 9.4 What can be reused?

Potentially reusable components:

- Audio loading.
- Resampling.
- Preprocessing.
- STFT/iSTFT.
- Feature extraction utilities.
- FastAPI infrastructure.
- React frontend.
- Experiment logging.
- CLI and notebook structure.
- Testing infrastructure.

The GMM and ZF-Net modules may provide architectural starting points, but their purpose, features, training data, objectives and outputs need to be reconsidered.

**Do not simply relabel the old environmental noise classifier as an atypical speech model.**

---

# 10. Research Tasks and Experiments

The following are the proposed research directions. Their order and feasibility should be evaluated before implementation.

## Experiment 1: Real atypical Telugu speech analysis

Analyse all 189 recordings.

Objectives:

- Extract acoustic features.
- Examine temporal, prosodic and spectral patterns.
- Compare recordings across speakers.
- Identify recurring and speaker-specific characteristics.
- Investigate possible recording artefacts and confounding variables.

Deliverable: A documented acoustic analysis report with plots, tables and evidence-supported observations.

## Experiment 2: Typical versus atypical Telugu comparison

Compare the real atypical recordings against suitable typical Telugu recordings.

Investigate whether selected features differ meaningfully.

Control for:

- Speaker variation.
- Recording conditions.
- Utterance content.
- Duration.
- Sample rate.
- Background noise.
- Dataset-specific collection differences.

Do not interpret every difference as an impairment-related characteristic.

## Experiment 3: Synthetic data generation

Based on Experiment 1, select a limited number of supported acoustic characteristics.

Generate controlled synthetic samples using suitable typical Telugu recordings.

Compare multiple transformation strengths and methods.

Maintain source-level traceability.

## Experiment 4: Synthetic-to-real validation

Determine whether synthetic speech resembles the acoustic patterns observed in held-out real atypical Telugu recordings.

Evaluate:

- Feature distributions.
- Temporal characteristics.
- Spectral characteristics.
- Prosodic characteristics.
- Linguistic content preservation.
- Naturalness.
- Intelligibility.

Human Telugu listener assessment should be considered where feasible.

## Experiment 5: Atypical speech detection baselines

Investigate whether acoustic features can distinguish typical and atypical recordings.

Potential baselines:

- Logistic regression.
- SVM.
- Random Forest.
- GMM-based statistical models.
- Pretrained speech embeddings, if computationally feasible.

This is a research baseline, not a clinical diagnosis system.

## Experiment 6: Intelligibility assessment

Investigate whether acoustic characteristics can help estimate speech understandability.

The Vaani Low/Medium/High difficulty field may be used as preliminary understandability information, with its limitations clearly documented.

Do not treat it as a clinical severity score.

## Experiment 7: Speech enhancement and reconstruction

Investigate whether the adapted GMM + ZF-Net architecture can improve the intelligibility of synthetic and real atypical Telugu speech.

Compare against suitable baselines.

Investigate whether a Telugu ASR-to-TTS reconstruction approach could serve as a comparison method.

Clearly distinguish:

- Signal enhancement.
- Speech reconstruction.
- Content-preserving normalization.
- Speaker voice restoration.

These are not interchangeable objectives.

---

# 11. Evaluation Strategy

Evaluation is a major challenge because the real atypical dataset does not contain paired clean recordings from the same speakers.

## 11.1 Synthetic paired evaluation

For synthetic examples, the original typical source can serve as a paired reference.

Possible metrics:

- SI-SDR.
- STOI.
- PESQ, where applicable.
- Spectral distance.
- Mel-cepstral distance.
- ASR character error rate (CER).
- ASR word error rate (WER).

The applicability and limitations of each metric must be assessed.

## 11.2 Real atypical speech evaluation

For real Vaani atypical recordings, there is no corresponding clean version from the same speaker.

Therefore, PESQ or STOI against an unrelated typical recording must not be presented as valid paired evaluation.

Consider:

- ASR CER/WER using appropriate Telugu ASR systems.
- Human intelligibility assessment by Telugu listeners.
- Speech naturalness.
- Linguistic content preservation.
- Acoustic feature changes.
- Speaker identity preservation, if technically measurable.
- Expert evaluation, if available.

A decrease in an acoustic deviation is not automatically proof of improved intelligibility.

A more typical-sounding output is not necessarily a better output.

## 11.3 Speaker-independent evaluation

There are only seven Telugu speakers.

This creates a serious risk of data leakage.

We should use speaker-independent evaluation.

Potential approaches:

- Leave-one-speaker-out evaluation.
- Grouped train/validation/test splits.

All recordings and synthetic descendants from one speaker or source must remain in the same partition.

Held-out evaluation speakers must not be used for:

- Synthetic data generation for training.
- Training.
- Model selection.
- Hyperparameter tuning.
- Transformation parameter selection.

The final evaluation should measure generalization to unseen speakers as far as the small dataset permits.

---

# 12. Technical Constraints and Development Environment

Current available resources:

- Laptop with 8 GB RAM.
- Integrated graphics.
- No local NVIDIA GPU.
- Free Google Colab environment.
- T4 GPU availability is not guaranteed.
- Google Drive for dataset storage.
- GitHub for code versioning.
- Antigravity/Cursor and AI coding agents for local development.

Current known data storage:

```text
Google Drive/
    Intelli-Speech-data/
        vaani_telugu_transcribed/
```

The project should prioritize:

- Lightweight feature extraction.
- Incremental processing.
- Avoiding full dataset loading into RAM.
- Reproducible notebooks.
- Checkpointing.
- GPU-aware training.
- CPU fallback.
- Efficient experiment tracking.

Do not assume paid GPUs, cloud services or expensive proprietary APIs are available.

---

# 13. Research Literature and Scientific Grounding

The research must be grounded in existing work on:

- Atypical speech processing.
- Dysarthric speech enhancement and synthesis.
- Speech intelligibility assessment.
- Speech data augmentation.
- Telugu speech processing.
- Acoustic feature analysis.
- GMM-based speech modelling.
- CNN-based speech enhancement.
- ZF-Net and related convolutional architectures.
- Neural speech reconstruction.

Three papers identified for initial investigation:

### Paper 1

**Accurate synthesis of dysarthric speech for ASR data augmentation (2024)**

https://www.sciencedirect.com/science/article/pii/S0167639324000839

DOI: 10.1016/j.specom.2024.103112

Investigate its controllable speech synthesis methods, augmentation strategies, severity control and reported results.

### Paper 2

**Improving Dysarthric Speech Segmentation With Emulated and Synthetic Augmentation (2024)**

https://pmc.ncbi.nlm.nih.gov/articles/PMC11008804/

DOI: 10.1109/JTEHM.2024.3375323

Investigate healthy-speaker emulation, speaking-rate and loudness modifications, synthetic speech and its evaluation methodology.

Remember that its primary task is speech segmentation, not generalized atypical speech enhancement.

### Paper 3

**Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis (Interspeech 2024)**

https://www.isca-archive.org/interspeech_2024/leung24_interspeech.html

DOI: 10.21437/Interspeech.2024-1645

Investigate its diffusion-based TTS approach and how synthetic dysarthric speech was used for ASR training.

These are starting points, not a complete literature review.

Verify bibliographic details, read the full papers and examine their limitations.

Search for relevant Telugu and Indian-language atypical speech research as well.

Do not assume that a method developed for English dysarthric speech will transfer directly to Telugu.

---

# 14. Important Scientific and Ethical Constraints

These rules must be followed throughout the project.

1. Do not invent clinical diagnoses or label individual speakers without supporting information.
2. Do not assume all atypical speech has the same acoustic characteristics.
3. Do not generate arbitrary synthetic distortions without evidence from real recordings.
4. Do not combine Hindi and Marathi speech patterns with Telugu for the current synthetic dataset.
5. Do not describe synthetic samples as genuine clinical speech.
6. Do not claim that synthetic data increases independent speaker diversity.
7. Do not use held-out speakers to create training data.
8. Do not treat Vaani difficulty ratings as clinical severity labels.
9. Do not claim that the old environmental noise suppression model already performs atypical speech enhancement.
10. Do not claim enhancement success without appropriate evaluation.
11. Do not assume that making speech sound more typical necessarily improves communication.
12. Preserve the original linguistic message wherever possible.
13. Consider speaker identity, personal speech characteristics and the possibility of over-normalization.
14. Clearly distinguish established facts, experimental observations, hypotheses and future proposals.

The intended system is an assistive technology research prototype, not a diagnostic or medical decision-making system.

---

# 15. What I Need You to Do

Act as a research engineer and technical research collaborator specializing in:

- Digital speech processing.
- Atypical speech analysis.
- Speech enhancement.
- Speech synthesis and augmentation.
- Telugu phonetics and speech technology.
- Deep learning.
- Statistical acoustic modelling.
- GMM-based modelling.
- CNN architectures and ZF-Net.
- Research methodology and experimental validation.

I need you to understand the entire project before suggesting implementation changes.

## Stage A — Understand and audit

1. Read this entire document.
2. Inspect the existing project repository.
3. Inspect the existing notebooks, datasets, model code, backend and frontend.
4. Identify what is already implemented.
5. Identify what is only proposed.
6. Identify what is experimentally validated and what is not.
7. Compare the existing implementation against the intended atypical speech research direction.

Do not assume that the code matches the architecture document.

## Stage B — Literature review

1. Read the three initial papers.
2. Find additional relevant research.
3. Prioritize Telugu and Indian-language research where available.
4. Compare different synthetic atypical speech generation approaches.
5. Investigate suitable GMM and ZF-Net adaptations.
6. Identify limitations and unresolved research problems.

Present a literature comparison containing:

- Research problem.
- Dataset.
- Method.
- Architecture.
- Evaluation metrics.
- Results.
- Limitations.
- Relevance to Intelli-Speech.

## Stage C — Dataset analysis plan

Design a reproducible analysis workflow for the 189 Telugu atypical recordings.

Specify:

- Data validation.
- Metadata inspection.
- Acoustic feature extraction.
- Speaker-level analysis.
- Confound analysis.
- Visualization.
- Statistical comparisons.
- Candidate characteristics for synthetic generation.

Explain exactly how we will determine whether a characteristic is sufficiently supported by the recordings to reproduce synthetically.

## Stage D — Synthetic dataset strategy

Compare feasible approaches to synthetic atypical Telugu speech generation.

For each approach, discuss:

- Technical feasibility.
- Telugu language support.
- Controllability.
- Realism.
- Computational requirements.
- Licensing.
- Content preservation.
- Validation requirements.

Recommend a staged strategy rather than immediately choosing an expensive or complex model.

## Stage E — GMM + ZF-Net redesign

Propose a technically justified architecture for the current research objective.

Investigate:

1. How GMM can model atypical acoustic characteristics.
2. Which acoustic features or embeddings should be used.
3. How GMM outputs could condition ZF-Net.
4. Whether FiLM conditioning is appropriate.
5. What ZF-Net should predict.
6. Which loss functions are suitable.
7. How synthetic paired training can be constructed.
8. How the model can be evaluated on unpaired real atypical recordings.
9. What alternative architectures should be compared.

Keep the original HDNZF-Net concept in consideration, but do not force it into a design if the evidence shows that a different formulation is more appropriate.

## Stage F — Complete execution roadmap

Prepare an implementation roadmap with:

- Ordered phases.
- Exact objectives.
- Required input data.
- Expected outputs.
- Models and tools.
- Experiments.
- Evaluation metrics.
- Compute requirements.
- Risks.
- Completion criteria.

Prioritize a feasible prototype using our current hardware and available datasets.

---

# 16. Immediate Instructions

**Do not modify the repository, generate synthetic data, train models or replace existing architecture yet.**

First complete the repository audit, literature review, dataset assessment and architecture proposal.

I want to understand the reasoning behind every major technical decision.

For every recommendation, explain:

- What problem it solves.
- Why it is appropriate.
- What evidence supports it.
- What assumptions it requires.
- What limitations it has.
- How we will test whether it works.

Separate conclusions into:

- Established facts.
- Findings from our actual dataset.
- Research hypotheses.
- Proposed experiments.
- Future possibilities.

Do not present a proposed experiment as an accomplished result.

Do not rush into implementation simply because a model architecture looks attractive.

The priority is to establish a scientifically defensible research direction for Intelli-Speech, then implement it incrementally.

**The central goal is to develop an evidence-based Telugu atypical speech processing system that can ultimately improve speech intelligibility while preserving the speaker's intended linguistic content and important individual speech characteristics.**