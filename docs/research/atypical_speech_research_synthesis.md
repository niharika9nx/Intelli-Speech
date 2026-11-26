# Intelli-Speech: Atypical Speech Research Synthesis

**As of:** 2026-11-26. **Live copy (editable):** https://claude.ai/artifact/QNsyPRrCY6ANivYZBvDc9q

Slurred (dysarthric) speech is defined mainly by articulation and prosody, not timing alone, so our timing-only transform cannot sound impaired.
- **Next step:** measure articulation, prosody and voice in our 7 Telugu speakers, then add only the transforms those measurements support.
- **Gate:** no published signal-processing method has listener evidence that it sounds impaired, so a blinded listening test must gate any synthetic dataset.
- **Evidence labels:** **Established** (replicated or review-level), **Supported** (one or a few studies), **Hypothesis** (plausible, untested).
- **Sources:** five research-agent reports and one Perplexity report, with 14 key citations checked against PubMed. Condensed agent notes are in the appendix.

## Comparison matrix

Of the 18 characteristics, 8 can be measured fully or partly from audio alone with at least supported evidence. Of those, vowel centralization, F2 transitions and monopitch are the ones most tied to slurring.

| Characteristic | Condition | Key metric | Audio only? | Evidence | Synthesis route (feasibility) | Telugu suitability |
| --- | --- | --- | --- | --- | --- | --- |
| Vowel centralization | PD, CP, stroke dysarthria | FCR, VAI (labelled vowels); AAVS, automatic VSA (unlabelled) | Partly: AAVS/automatic VSA yes | Established (FCR, PD); Supported (audio-only variants) | Formant warping toward centroid, WORLD (medium; Hypothesis) | Medium: 5 qualities × length; separate short/long vowels |
| Shallow F2 transitions | PD, stroke, ataxic | F2 slope, its variability | Standard needs words; slope distribution is Hypothesis | Supported | Temporal smoothing of the spectral envelope (medium; Hypothesis) | Medium |
| Imprecise consonants, spirantization | PD (core), most dysarthrias | Spirantization ratio, burst presence | No: needs stop segmentation | Established perceptually; acoustic Supported | Burst attenuation + closure noise (low; needs segmentation) | Low–medium: retroflex, aspiration, geminates |
| VOT change | PD, PSP/MSA | VOT | No | Supported (incl. Hindi parkinsonism) | Aspiration-interval scaling (low) | Low: aspiration and prevoicing are phonemic |
| Monopitch | PD | F0 SD (semitones) | Yes | Established | F0 compression, WORLD/PSOLA (high) | High, needs local intonation norms |
| Monoloudness | PD | Intensity SD (dB) | Yes (relative only) | Established | Intensity-envelope compression (high) | High |
| Hypophonia | PD, presbyphonia | Mean SPL | Absolute needs calibration; relative only | Established | Gain + spectral-tilt co-variation (medium; Hypothesis) | Medium |
| Slow articulation rate | MS, HD, PSP/MSA, ageing (mild); **not typical PD** | Articulation rate | Yes | Established | Time-varying WSOLA (high; built) | High (our Exp 1 finding) |
| More / longer pauses | Ageing, MCI/AD, many dysarthrias | Pause rate, log-duration mixture | Yes | Established | Pause insertion from a fitted log-normal mixture (high; built) | High; fit Telugu mixture first |
| Acceleration, short rushes | PD | Within-utterance rate change | Yes | Supported | Time-varying rate increase (medium; Hypothesis) | Medium |
| Rhythm disturbance | Dysarthrias | EMS (audio-only); nPVI, %V (segmented) | EMS yes | Supported | A consequence of other edits; use as a manipulation check | Medium: syllable-timed baseline |
| Breathiness | Presbyphonia, PD | CPP(S), H1*–H2*, L/H ratio | CPP yes; H1–H2 needs formants | Supported | WORLD aperiodicity, aspiration noise (medium) | Medium; channel-sensitive |
| Roughness (jitter, shimmer) | Presbyphonia | Jitter, shimmer | Sustained vowels preferred | Supported, inconsistent as age marker | Pulse perturbation (medium; not perceptually independent of noise) | Low on phone audio |
| F0 tremor | Ageing, neurological tremor | Tremor rate and extent | Needs sustained vowels | Supported | F0 sinusoidal + random modulation (high; listener-validated for perceived age) | Low on connected speech |
| Hypernasality | HD, cleft; **not common in PD** | 1/3-octave band near 1 kHz on /i/ | Needs sustained vowels | Supported | Nasal pole/zero (low; Hypothesis) | Low: phonemic nasals |
| Sound/syllable repetitions | Developmental, neurogenic stuttering | %SS, repetition units | No: needs transcript and annotation | Established clinically; labels unreliable (κ 0.40) | Splice injection (low quality reported) or TTS tokens | Medium: needs Telugu syllabification |
| Prolongations, blocks | Stuttering | Duration z-score, intra-word silence | No | Established clinically; κ 0.11 / 0.25 | Phone stretch, gap insertion (low) | Low: long vowels and geminates confound |
| Lexical diversity, word-finding pauses | Ageing, MCI/AD | MATTR, pre-content-word pause | No: transcripts | Supported | Text rewriting + TTS only (out of scope) | Low: agglutinative morphology |

**Abbreviations:** PD = Parkinson's disease; CP = cerebral palsy; PSP/MSA = progressive supranuclear palsy / multiple system atrophy; HD = Huntington's disease; MCI/AD = mild cognitive impairment / Alzheimer's disease.

## Cross-check of the reports

The six reports agree on the framing. They disagree on three facts and contain four citation errors, all in the Perplexity report.

**Agreement (all reports):**

- Model specific conditions, not a generic "atypical" effect.
- The Vaani corpus supports description, not diagnosis.
- Signal-processing simulation has ASR evidence but no listener evidence of realism (Vachhani 2018; Geng 2020).
- Telugu norms do not exist for any characteristic reviewed.

**Contradictions resolved:**

| Claim | Resolution | Basis |
| --- | --- | --- |
| PD speech is slow, with longer pauses (Perplexity) | **Wrong as stated.** 121 PD vs 70 controls: no overall articulation-rate difference, stronger acceleration, fewer but longer pauses. | Skodda & Schlegel 2008, checked in PubMed |
| Older speakers talk more slowly | Healthy ageing slows rate and adds pauses (Bóna 2014; Jacewicz 2009/2010), but Harnsberger 2010 describes *faster* rate as common in older male voices. **Unresolved:** treat the direction as Hypothesis for Telugu. | All three checked |
| Ageing centralizes vowels | **No.** Vowel space was larger with age in 100 Czech speakers; DDK rate is age-stable from 65 to 86. Centralization indicates disease, not age. | Tykalova 2021; Pierce 2013 |
| Jitter and shimmer mark ageing | **Weakly.** They rise only from 80–89, depend on sex and software, and listeners do not hear them apart from noise. | Rojas 2020; Kreiman & Gerratt 2005 |
| Hypernasality is part of the PD profile | **Not common in PD.** Acoustic hypernasality was found in 27% of PD vs 19% of controls. | Novotný 2016 |
| Telugu has "tone-like" pitch expectations (Perplexity) | **Unsupported.** Telugu is not tonal. Dropped. | No source given |

**Citation errors in the Perplexity report (DOIs correct, attributions wrong):**

- "Roy et al. 2020" is Rojas, Kefalianos & Vogel 2020.
- "Ingham et al. 2004" is O'Brian, Packman, Onslow & O'Brian 2004.
- "Brabenec et al. 2023" is Atalar, Oguz & Genc 2023 (*Şişli Etfal* bulletin). That abstract lists an *increased* speech rate in PD. Its claimed "6 s minimum phonation time" is not in the abstract, so it is unverified.
- The weighted stuttering-measure source is university lecture notes, not peer-reviewed.

The five research agents' citations checked in PubMed (9 of 9) matched their DOIs and claims.

## What our data can measure

Our recordings are connected, image-prompted speech on uncalibrated phones, so only audio-only measures of connected speech are usable now. Sustained-vowel, calibrated and clinical measures need new recordings.

| Requirement | Characteristics | Usable on our data now? |
| --- | --- | --- |
| Audio only, connected speech | Articulation and speech rate; pause rate and log-duration mixture; F0 SD (monopitch); relative intensity SD (monoloudness); CPP(S); EMS rhythm; automatic VSA / AAVS; F2-slope distribution (Hypothesis) | **Yes.** Rate and pauses were already measured in Exp 1. |
| Formant tracking on voiced frames (no labels) | H1*–H2*, automatic vowel space, F2 slopes | Yes, with care: the 1.2–5 kHz channel spread distorts formants and tilt |
| Transcript or forced alignment | FCR/VAI on labelled vowels; VOT; consonant spirantization; %SS; pause location; prolongation durations | Partly. Atypical clips have intended Telugu-script transcripts. Only 443 of 4,914 typical source clips do, and no Telugu forced aligner has been verified. |
| Sustained vowels or dedicated tasks | Jitter/shimmer (reliable), tremor, MPT, hypernasality, DDK | **No.** New recordings needed. |
| Calibrated recording | Absolute loudness (SPL) | **No.** Phone gain control destroys it. |
| Clinical assessment | Any diagnosis: stuttering type, PD, presbyphonia, MCI | **No, never inferred.** Needs clinicians and labelled participants. |

## Ranking

Timing and prosody rank highest: strong evidence, and measurable and transformable on our audio now. Articulation transforms are next, but they need Experiment 2 and a listener test first.

The live document shows this as a 2×2 drawing. "Feasible" means measurable on our connected phone speech without transcripts, and transformable with a deterministic method.

| | Hard on our audio now | Feasible on our audio now |
| --- | --- | --- |
| **Strong evidence** | **Needs labelled or new recordings:** vowel centralization (FCR), imprecise consonants, stuttering events, absolute loudness (SPL) | **Build now:** slower articulation (built), more/longer pauses (built), monopitch (F0 compression), monoloudness (intensity compression) |
| **Weaker evidence** | **Leave out for now:** VOT shifts, jitter and shimmer, hypernasality, lexical and word-finding | **Test first (Exp 2, then listeners):** F2 undershoot (envelope smoothing), centralization via AAVS + warp, breathiness (aperiodicity), F0 tremor and F0 noise (age) |

## Recommended metrics

Experiment 2 should add eight audio-only metrics to the 14 timing features Exp 1 confirmed. Each is computed per recording and compared at speaker level, as in Exp 1.

| Metric | Definition | Measures | Basis |
| --- | --- | --- | --- |
| AAVS | sqrt(det Σ), Σ = covariance of frame-wise (F1, F2) over voiced frames | Vowel-space use, no labels | Whitfield 2019; Sandoval 2013 (automatic VSA) |
| Vowel-space dispersion | Area of the convex hull of the (F1, F2) 5th–95th percentile cloud, per speaker, Bark scale | Centralization proxy | Sandoval 2013; Bark per Fletcher 2017 |
| F2-slope distribution | Median and SD of \|dF2/dt\| (Hz/ms) over voiced frames with stable tracking | Transition speed (Hypothesis without words) | Kim et al. 2009; van Brenk 2024 |
| F0 SD | SD over voiced frames of 12·log2(F0 / median F0) | Monopitch | Rusz 2021 guidelines |
| Intensity SD | SD of frame dB over speech frames, pauses excluded | Monoloudness (relative) | Rusz 2021 |
| CPPS | Smoothed cepstral peak minus regression line, connected speech | Breathiness, dysphonia | Patel 2018; Buckley 2023 |
| EMS | Spectrum of the 0–10 Hz amplitude envelope, full band and octave bands: peak frequency, energy below / around / above 4 Hz | Rhythm (manipulation check) | Liss 2010 |
| Pause log-mixture | Gaussian mixture on log pause duration: weights and means of brief / medium / long components | Pause structure, threshold-free | Campione & Véronis 2002 |

All eight run on audio alone. Each is also checked against the channel descriptors (bandwidth, RMS) as in Exp 1, and reported at several pause thresholds where relevant.

## Generation strategy

The recommendation is parametric WORLD-vocoder transforms layered on the existing timing module, each enabled only if Experiment 2 shows that pattern in our speakers. Learned voice conversion and dysarthric TTS are the only methods with listener evidence, but both need labelled atypical training data we do not have. They stay a later comparison.

**Layers (each a separately switchable, deterministic module):**

1. **Timing (built, v0.2):** articulation slowing with a steady-vs-transient contour, and pause insertion. Next: fit the pause durations to a Telugu log-normal mixture instead of a single log-normal.
2. **Articulation (proposed):** formant movement undershoot by low-pass filtering the WORLD spectral envelope over time, and vowel centralization by warping formants toward the speaker's (F1, F2) centroid. Both are Hypotheses, enabled only if AAVS or F2-slope differences are confirmed.
3. **Prosody (proposed):** F0 compression, F0' = exp(μ + α(log F0 − μ)) with 0 < α < 1, plus intensity-envelope compression. Speaker-specific α, because Exp 1 found F0 variation was not shared.
4. **Voice (optional age layer):** raised aperiodicity for breathiness, plus F0 tremor and random F0 modulation. Of all the age cues, only these, together with speaking rate, have listener evidence of shifting perceived age (Harnsberger 2010).

**Left out, with the reason:**

- Stuttering injection: needs alignment and annotation, splice injection is reported as low quality, and stuttering is outside the chosen scope.
- Hypernasality: not common in PD and confounded by Telugu's phonemic nasals.
- VOT and consonant edits: need stop segmentation and per-category Telugu norms. Revisit after a Telugu aligner exists.
- Lexical and cognitive features: text-level, not acoustic.
- Gain-only loudness: undone by normalisation and unrealistic.

**Controls built into generation:**

- Parameters are fitted per leave-one-speaker-out fold, never from the held-out speaker.
- Synthetic and real audio are kept apart, with full provenance.
- Added pauses use the clip's own background noise, so a model cannot learn digital silence.
- A v0.2-only condition (timing alone) is kept as the ablation baseline.

## Validation protocol

A synthetic condition passes only if it passes three tests in order: it matches the real distributions, it adds no channel shortcut, and blinded listeners hear it as impaired. ASR gains do not count as evidence of realism.

1. **Acoustic match (per fold).**
   - Re-measure synthetic clips with the same extractor as the real data.
   - Compare synthetic and held-out real speaker distributions: Wasserstein distance and KS test on log pause duration, rate, F0 SD, AAVS and CPPS.
   - Use bootstrap confidence intervals over speakers.
   - Pass: the held-out speaker lies inside the synthetic P5–P95 range (the existing coverage check) for the targeted features.
2. **Manipulation checks.**
   - Each layer moves its own metric in the intended direction and leaves the others within tolerance. Examples: F0 compression must not change articulation rate; envelope smoothing must not change F0 SD.
   - EMS is reported as a rhythm consequence.
3. **Artefact and channel controls.**
   - Click score, bandwidth and RMS are matched between real and synthetic sets.
   - A shallow classifier trained only on channel descriptors must **fail** to separate synthetic from real typical speech. Otherwise any detector would learn the artefact (synthesis agent; Hypothesis-level risk).
4. **Blinded listener test.**
   - Native Telugu listeners, plus at least one speech-language pathologist if available.
   - Each hears real atypical, real typical and synthetic clips, matched for duration and channel and in random order.
   - Tasks: (a) typical vs impaired; (b) natural vs processed; (c) which sounds more impaired (forced choice, as in Huang et al. 2021).
   - Target: synthetic clips rated impaired at least as often as the 65% SLP identification reported for ALS-style conversion (Jiao 2018), with the "processed" rate no higher than for real atypical clips.
   - MOS alone is not used: it confuses severity with unnaturalness (Illa et al. 2021).
5. **Leakage audit.**
   - Confirm no held-out speaker contributed to any fitted parameter, profile or reference pool.
   - Confirm no synthetic clip sourced from a speaker in the evaluation controls.
6. **Downstream (last).**
   - Only after steps 1–5: train the GMM and ZF-Net on synthetic data, test on real held-out speakers, and run ablations by layer.

## Roadmap

Seven steps, each gated on the one before. No synthetic dataset is generated before gate 5.

1. **Experiment 2: articulation, prosody and voice measurement.** Add the eight recommended metrics to the Exp 1 pipeline and rerun the speaker-level catalogue with its robustness checks. *Gate:* which features are class-1 (shared), speaker-specific or channel-confounded.
2. **Telugu pause model.** Fit the log-normal pause mixture on the 4,914 typical source clips, then compare it with the atypical speakers. *Gate:* parameters per fold.
3. **Build only the supported layers.** Add WORLD-based articulation, prosody or voice layers for features passing gate 1, each with unit tests and manipulation checks. *Gate:* each layer moves only its own metric.
4. **Closed-loop validation.** Calibration, coverage and channel checks for every layer combination, extending `TEMPORAL_V1`. *Gate:* the held-out speaker covered, and channel classifier at chance.
5. **Blinded listener test.** Real vs synthetic vs typical clips, rated by Telugu listeners and ideally a speech-language pathologist. *Gate:* the realism target in the validation protocol.
6. **Generate and train.** On-the-fly synthetic pairs per fold, then GMM detection and ZF-Net localisation, with layer ablations. *Gate:* improvement on real held-out speakers, not on synthetic test data.
7. **Data and ethics, in parallel from now.**
   - Ethics approval and a recording protocol for older Telugu speakers with dysarthria: sustained vowels, DDK, reading and picture description, with clinical labels.
   - Request access to the Tamil SSNCE dysarthric corpus (LDC2021S04) for method development only.

## Research gaps

The largest gap is that no Telugu norms or labelled Telugu clinical speech exist for any characteristic reviewed. Every literature value is therefore a prior, not a target.

- **No Telugu data.** No Telugu dysarthria, stuttering, ageing-voice or MCI corpus was found. The nearest are Tamil SSNCE (cerebral palsy dysarthria), Hindi parkinsonism VOT, Kannada stuttering, and South Indian AVQI/ABI voice norms.
- **Unvalidated tools.** Syllable-nucleus detection (validated on Dutch), automatic vowel-space measures (English, small samples) and EMS rhythm metrics have not been tested on Telugu.
- **Phonology interactions.** How phonemic vowel length, gemination, retroflex and aspirated stops interact with prolongation, centralization and VOT measures is unknown (Hypothesis).
- **No listener validation of signal-processing transforms.** Envelope smoothing, formant centralization, F0 compression and burst attenuation have none. Only tremor and F0-noise resynthesis has listener evidence, and that is for perceived age, not impairment.
- **Cause cannot be identified.** Slow rate and long pauses occur in ageing, MCI, MS, HD, atypical parkinsonism, task effects and channel differences. Our 7 speakers' timing pattern does not match classic PD, and no cause should be claimed.
- **Ageing rate direction.** Studies disagree on whether older adults speak faster or slower.
- **Channel confound.** Atypical and typical recordings differ in bandwidth and normalisation, and noise itself alters VOT, spirantization and CPP measures.
- **Small listener panels.** Published panels have 1–3 SLPs, and there is no agreed protocol for judging synthetic impaired speech.

**Open questions for the team:**

- Can we reach a hospital or speech-therapy department for labelled recordings and listener raters?
- Do any of the 7 Vaani speakers have intended transcripts clean enough to align, so FCR and VOT can be computed on labelled vowels and stops?

## Sources

Items marked ✓ were checked against PubMed by the lead. The others come from the research-agent reports, which cite only retrieved sources. arXiv items are preprints.

**Clinical and acoustic**

- ✓ [Skodda & Schlegel 2008, speech rate and rhythm in PD](https://doi.org/10.1002/mds.21996)
- ✓ [Rusz et al. 2021, recording and acoustic-analysis guidelines for movement-disorder dysarthrias](https://doi.org/10.1002/mds.28465)
- ✓ [Sapir et al. 2010, formant centralization ratio](https://doi.org/10.1044/1092-4388(2009/08-0184))
- ✓ [Fletcher et al. 2017, vowel-centralization methods](https://doi.org/10.1044/2016_JSLHR-S-15-0355)
- ✓ [Sandoval et al. 2013, automatic vowel space area](https://doi.org/10.1121/1.4826150)
- ✓ [Kim, Weismer, Kent & Duffy 2009, F2 slope and severity](https://doi.org/10.1159/000252849)
- ✓ [Liss, LeGendre & Lotto 2010, envelope modulation spectra](https://doi.org/10.1044/1092-4388(2010/09-0121))
- ✓ [Kent & Kim 2003, acoustic typology of motor speech disorders](https://doi.org/10.1080/0269920031000086248)
- ✓ [Allison & Hustad 2018, acoustic predictors of dysarthria in cerebral palsy](https://doi.org/10.1044/2017_JSLHR-S-16-0414)
- ✓ [Carl & Icht 2021, vowel centralization in Hebrew developmental dysarthria](https://doi.org/10.1111/1460-6984.12598)
- ✓ [Novotný et al. 2016, hypernasality and basal ganglia](https://doi.org/10.7717/peerj.2530)
- ✓ [Atalar, Oguz & Genc 2023, hypokinetic dysarthria narrative review](https://doi.org/10.14744/SEMB.2023.29560)
- ✓ [Cao et al. 2025, speech biomarkers for PD](https://doi.org/10.1038/s41531-025-00913-4)
- [Whitfield et al. 2019, sentence-level vowel space (AAVS)](https://pubs.asha.org/doi/10.1044/2019_JSLHR-S-MSC18-18-0189)
- [van Brenk, Lowit & Tjaden 2024, F2-transition variability](https://doi.org/10.1159/000534337)
- [Tykalova et al. 2017, consonant articulation in PD, PSP, MSA](https://doi.org/10.1016/j.bandl.2016.11.005)
- [Saxena et al. 2014, Hindi parkinsonism speech](https://doi.org/10.1016/j.parkreldis.2014.04.024)

**Ageing**

- ✓ [Rojas, Kefalianos & Vogel 2020, voice ageing meta-analysis](https://doi.org/10.1044/2019_JSLHR-19-00099)
- ✓ [Harnsberger et al. 2010, noise and tremor in perceived vocal age](https://doi.org/10.1016/j.jvoice.2009.01.003)
- ✓ [Tykalova et al. 2021, ageing and Czech vowel formants](https://doi.org/10.1016/j.jvoice.2020.02.022)
- ✓ [Bóna 2014, age and speech-style temporal characteristics](https://doi.org/10.1121/1.4885482)
- ✓ [Saccente-Kennedy et al. 2026, presbyphonia intervention review](https://doi.org/10.1016/j.jvoice.2023.12.010)
- ✓ [Buckley, Abur & Stepp 2023, CPP norms across the lifespan](https://doi.org/10.1044/2023_AJSLP-22-00264)
- [Kreiman & Gerratt 2005, perception of aperiodicity](https://doi.org/10.1121/1.1858351)
- [Jayakumar et al. 2022, AVQI lifespan norms in India](https://doi.org/10.1016/j.jvoice.2020.05.025)
- [Pierce, Cotton & Perry 2013, motion rates in older adults](https://doi.org/10.1111/1460-6984.12001)
- [Campione & Véronis 2002, multilingual silent-pause durations](https://mail.sprosig.org/sp2002/pdf/campione-veronis.pdf)
- [Slegers et al. 2018, connected speech in Alzheimer's disease](https://doi.org/10.3233/JAD-170881)

**Stuttering**

- ✓ [O'Brian, Packman, Onslow & O'Brian 2004, measuring stuttering in adults](https://doi.org/10.1044/1092-4388(2004/080))
- [Lea et al. 2021, SEP-28k dataset](https://arxiv.org/abs/2102.12394)
- [Venkatagiri, Nataraja & Deepthi 2017, stuttering and Kannada morphophonemics](https://doi.org/10.1080/02699206.2016.1259353)

**Synthesis methods**

- [Vachhani, Bhat & Kopparapu 2018, healthy-speech augmentation for dysarthric ASR](https://doi.org/10.21437/Interspeech.2018-1751)
- [Geng et al. 2020, augmentation for disordered speech recognition](https://doi.org/10.21437/Interspeech.2020-1161)
- [Jiao et al. 2018, simulating dysarthric speech](https://arxiv.org/abs/1804.10325)
- [Soleymanpour et al. 2022, dysarthric TTS with severity and pause control](https://arxiv.org/abs/2201.11571)
- [Wang et al. 2023, DuTa-VC](https://arxiv.org/abs/2306.10588)
- [Huang et al. 2021, normal-to-dysarthric voice conversion](https://arxiv.org/abs/2110.08213)
- [Illa et al. 2021, pathological voice adaptation](https://arxiv.org/abs/2106.08427)
- [Li et al. 2025, voice conversion for low-resource dysarthric ASR](https://arxiv.org/abs/2505.14874)
- [Morise, Yokomori & Ozawa 2016, WORLD vocoder](https://doi.org/10.1587/transinf.2015EDP7457)
- [Vaani Atypical Speech Corpus](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Atypical-Speech-Corpus)
- [SSNCE Tamil dysarthric speech, LDC2021S04](https://catalog.ldc.upenn.edu/LDC2021S04)

---

# Appendix: condensed research-agent notes and citation verification

The notes below were condensed by the lead from the full agent reports (2026-11-26). They keep each report's key claims, evidence labels and DOIs. Numbers come from the cited sources as reported by the agents. The ✓-marked items above were re-checked in PubMed.


## Agent 1 — Stuttering & fluency (condensed by lead, 2026-11-26)
Core: SLDs = part-word/sound-syllable repetitions, prolongations, blocks (ESTABLISHED). Typical adults: 6–8 total disfluencies, <1.5 SLD /100 syll (Roberts 2009, English men, 10.1016/j.jcomdis.2009.06.001). Adult prevalence ~0.6–0.7 % (Gattie 2025, 10.1016/j.jfludis.2024.106085). Onset mostly <5 y (Yairi & Ambrose 2013, 10.1016/j.jfludis.2012.11.002).
Metrics: %SS; SLD/100 syll; RU mean; CV of IOI; prolongation duration z; spectral-flux stationarity; intra-word silence; SLD/(SLD+OD); SR, AR; %V, ΔC, VarcoV, nPVI (Boecher 2025, 10.1044/2024_JSLHR-24-00076); F2 transition extent/slope (Dehqan 2016, 10.1016/j.jfludis.2016.01.005); locus equations (Verdurand 2020); VOT CV (Falk 2016).
Annotation reliability low: SEP-28k Fleiss κ block 0.25, prolongation 0.11, sound rep 0.40, word rep 0.62, interj 0.57 (Lea 2021, arXiv 2102.12394).
Needs transcripts/alignment for most (%SS denominators, block vs linguistic pause, word-level location); audio-only only via trained detectors; clinical judgement for pathology.
Synthesis: acoustic splice injection (LibriStutter/VCTK-Stutter) -> reported LOW audio quality (Zhou 2024 arXiv 2408.15297; Zhang 2025 LLM-Dys arXiv 2505.22029); TTS with control tokens (Stutter-TTS, Zhang 2022 arXiv 2211.09731: event F1 0.63–0.84, 5.7 % rel WER gain). Realism not validated perceptually/clinically.
Neurogenic: adult onset after injury/disease/drugs (antipsychotics 57 % of drug-induced; Nikvarz 2022, 10.5498/wjp.v12.i2.236); co-occurs with dysarthria/aphasia; SLD clusters (Penttilä 2019, 10.1016/j.jfludis.2019.01.001). Cluttering: no operational acoustic criteria/datasets.
Indian/Dravidian: Kannada stutter rate tracks word length (25.3 % variance) vs morphophonemic complexity 7.5 % (Venkatagiri 2017, 10.1080/02699206.2016.1259353); %SS differs across languages in Kannada–English bilinguals (Kashyap & Maruthy 2020, 10.1080/02699206.2019.1637459). IIITH-TISA (Indian English, arXiv 2411.17149); Boli (multilingual Indian, CC BY 4.0, arXiv 2501.15877; only ~2 Telugu speakers per snippet — unverified). No Telugu stuttering corpus/norms.
Telugu HYPOTHESES: phonemic vowel length & geminates confound prolongation/block detection; aspirated/retroflex stops shift VOT baselines.
Datasets: SEP-28k, FluencyBank (+Timestamped 10.1044/2024_JSLHR-24-00070), UCLASS (10.1044/1092-4388(2009/07-0129)), KSoF (LREC 2022), IIITH-TISA, Boli.
Confidence: repetitions medium; prolongation low–medium; blocks low; rate medium; cluttering low; fluent-speech coarticulation low.
Agent caveats: IIITH-TISA size, Boli Telugu count, LLM-Dys size from snippets only; some formulas standard (not from retrieved source).

## Agent 2 — Presbyphonia / vocal ageing (condensed by lead, 2026-11-26)
Mechanism: presbylaryngis (lamina propria + TA atrophy, bowing, glottal insufficiency) — ESTABLISHED (Martins 2014 10.1007/s40520-013-0143-5; Rosow & Pan 2019 10.1016/j.otc.2019.03.005; Mallick 2019 10.1097/MOO.0000000000000540). Diagnosis of exclusion, needs laryngoscopy (Johns 2011 10.1177/0194599811404640). Atrophy ≠ always dysphonia.
F0: women fall with age; men stable/slight rise ≥70 (Nishio & Niimi 2008 10.1159/000118510, n=374 Japanese; Goy 2013 10.1016/j.jvoice.2013.03.002). Pooled 80–89 > 60–69 (Rojas 2020 10.1044/2019_JSLHR-19-00099). Harnsberger 2010: F0 shift alone did NOT change perceived age.
Tremor: PF0T rises with age in both sexes; amplitude tremor no age effect; physiological, below neuro thresholds (Bošković 2026 10.1007/s11357-026-02382-w). Cepstral F0 variability predicts age (Schultz 2023 10.1016/j.jvoice.2021.06.007). Tremor + random F0 noise resynthesised raised perceived age up to ~12 y, additive with speaking rate (Harnsberger 2010 10.1016/j.jvoice.2009.01.003) — KEY perceptual-validation precedent. Tremor needs sustained vowels; connected speech: only F0 SD/range (intonation-confounded).
Jitter/shimmer: 80–89 > 60–69 (Rojas); men higher perturbation/NHR; small vs inter-speaker; sustained vowels needed; listeners don't perceive jitter/shimmer independently from noise (Kreiman & Gerratt 2005 10.1121/1.1858351). Therapy: jitter −0.62 %, shimmer −1.05 % (Saccente-Kennedy 2026 10.1016/j.jvoice.2023.12.010). Shimmer biased on mobiles (Fahed 2025 10.1016/j.jvoice.2022.10.006); jitter OK on phone (Pommée 2025 10.1016/j.jvoice.2022.08.027).
CPP/CPPS: no age effect (Buckley 2023 10.1044/2023_AJSLP-22-00264, n=150 18–91; software + stimulus matter); lower CPP older (Kuruvilla-Dugdale 2020 10.1016/j.jcomdis.2020.105995); Cantonese small/ns after FDR (Gao 2026); CPPS declines >65 longitudinally (Stager 2020). Dysphonia cutoffs (not ageing) Praat CPP 14.45 dB /a/, 9.33 dB Rainbow (Murton 2020 10.1044/2020_AJSLP-20-00001). ASHA protocol (Patel 2018 10.1044/2018_AJSLP-17-0009). CPP from connected speech works without transcripts.
Breathiness: periodicity measures dominate breathiness perception (Hillenbrand 1994 10.1044/jshr.3704.769; 1996 10.1044/jshr.3902.311); H1*-H2*, tilt, L/H ratio; ABI validated Malayalam/Kannada cutoff >3.66 (Jayakumar & Benoy 2024 10.1016/j.jvoice.2022.05.006).
Loudness: absolute SPL impossible on uncalibrated phone data; only relative contours; gain-only change unrealistic (should co-vary tilt — HYPOTHESIS).
MPT: needs dedicated task — NOT derivable from Vaani; ~16–18 s older Japanese (Shinoda 2017).
Overall: AVQI higher in older Indians, Indian values slightly higher than European/East Asian (Jayakumar 2022 10.1016/j.jvoice.2020.05.025); South Indian AVQI cutoffs (10.1016/j.jvoice.2022.02.019).
Synthesis: WORLD (Morise 2016 10.1587/transinf.2015EDP7457) F0/aperiodicity; LF source model; UCLA synthesizer (Kreiman 2010 10.3758/BRM.42.4.1030) copy synthesis; tremor models perceptually validated (Kreiman 2003 10.1044/1092-4388(2003/016)).
Datasets: Saarbrücken (German, CC BY 4.0 per sbvoicedb); no Telugu ageing-voice corpus; Indian norms = statistics only.
Gaps: no Telugu norms; ageing effects small, sex-dependent, hardware-dependent -> synthetic shifts may be within channel spread; acoustics cannot separate presbyphonia from PD/tremor/lesions; frame as "ageing-like acoustic shifts".
Feasible on Vaani connected phone speech: F0, F0 SD, CPP(S), jitter (with care); NOT: MPT, absolute SPL, reliable shimmer/HNR, tremor (needs vowels).

## Agent 3 — Hypokinetic & age-associated dysarthria (condensed by lead)
PD profile: reduced loudness, monopitch/monoloudness, imprecise consonants, vowel centralization, shallow F2 transitions, rate/rhythm disturbance incl. ACCELERATION and fewer but longer pauses (Darley 1969 10.1044/jshr.1202.246; Rusz 2021 guidelines 10.1002/mds.28465; Rusz 2024 10.1016/j.neubiorev.2024.105922).
KEY: PD articulation rate often NOT slowed (Skodda & Schlegel 2008 10.1002/mds.21996: 121 PD vs 70 HC). Slowed AR + longer pauses in MS (Rusz 2019 10.1016/j.bandl.2019.04.009), HD (Kouba 2023 10.1111/ene.15726), PSP/MSA (Tykalova 2017 10.1016/j.bandl.2016.11.005). Vaani speakers' slow+pausing profile fits classic PD LESS -> don't label "Parkinsonian".
Ageing: VSA larger with age (Tykalova 2021); DDK age-stable 65–86 (Pierce 2013 10.1111/1460-6984.12001); voice ageing (presbyphonia) more than articulation.
Vowels: tVSA, FCR (Sapir 2010), VAI = 1/FCR (Skodda 2011 10.1016/j.jvoice.2010.01.009; progression Skodda 2012 10.1371/journal.pone.0032132); need vowel labels. AUDIO-ONLY: automatic VSA (Sandoval 2013 JASA 10.1121/1.4826150), AAVS = sqrt(det Σ(F1,F2)) (Whitfield 2019 JSLHR) — mixed replication, larger in men.
F2 slope: reduced in PD + stroke (Kim, Weismer, Kent, Duffy 2009 10.1159/000252849); slope variability predicts intelligibility, rate-dependent (van Brenk 2024 10.1159/000534337); tongue pressure r=0.368 (Tamura 2022). Audio-only |dF2/dt| distribution = HYPOTHESIS.
Consonants: imprecise consonants core (ESTABLISHED); spirantization trend n.s. (Cushnie-Sparrow 2016; noise itself alters spirantization/VOT/intensity!); phonological posteriors [continuant] (Wayland 2025). Needs segmentation.
VOT: prolonged PSP/MSA > PD (Tykalova 2017); Hindi parkinsonism delayed VOT + lower intensity (Saxena 2014 10.1016/j.parkreldis.2014.04.024) — Indian evidence; Telugu aspiration/prevoicing phonemic -> per-category norms.
Hypophonia: needs calibration; levodopa no intensity effect (Pinho 2018). Only relative measures on Vaani.
Monopitch: F0 SD st audio-only; Kerala study pitch monotonicity in extrapyramidal dysarthria (Thoppil 2017 10.4103/aian.AIAN_130_17). Synthesis: F0' = exp(mu + a(logF0 - mu)).
Rhythm: EMS audio-only, 84–100 % classification dysarthria types (Liss 2010 10.1044/1092-4388(2010/09-0121)); rhythm metrics DFA >80 % (Liss 2009 10.1044/1092-4388(2009/08-0208)); cross-language caution (Liss 2013 10.1159/000350030). Use EMS as manipulation check.
Hypernasality: NOT common in PD (Novotný 2016 10.7717/peerj.2530: acoustic 27 % PD vs 19 % HC); needs sustained vowels.
Synthesis (all HYPOTHESIS, no listener validation for DSP): formant warping toward centroid, envelope temporal smoothing, burst attenuation, F0/intensity compression, WORLD aperiodicity.
Datasets: TORGO, UA-Speech, PC-GITA, SSNCE (TAMIL, 20 CP dysarthric + 10 controls, ~8 h, phonetic transcripts, LDC2021S04) — closest Dravidian; DAU-KDAH (Hindi/Marathi/Gujarati, unverified). No Telugu dysarthric corpus.

## Agent 4 — Age-related temporal & cognitive-linguistic (condensed by lead)
Healthy ageing: somewhat slower AR, more unfilled pauses, repetitions/revisions, TOT; semantics preserved (Walters 2026 10.1044/2026_JSLHR-25-00799 n=272; Jacewicz 2009/2010 10.1121/1.3459842; Burke & Shafto 2004).
MCI/AD: larger timing changes + lexical-semantic loss (Slegers 2018 SR 10.3233/JAD-170881; Vincze 2020 10.1080/02699206.2020.1827043; Balogh 2022 10.1017/S1355617721001454; Yang 2026 Chinese+English, language did not moderate).
Timing NON-SPECIFIC: PD, nfvPPA (García 2022 AUC 0.95 AR), T2 diabetes (Imre 2022), alcohol (Farkas 2026) -> ESTABLISHED; task-dependent (Pistono 2019).
AR = motor; pause location / share / lexical diversity = cognitive planning (needs transcripts).
Pause durations log-normal, multimodal: read ~129 ms/541 ms; spontaneous French 78/426/1585 ms (k .06/.80/.14); thresholds can reverse conclusions (Campione & Véronis 2002, Speech Prosody — not PubMed). Use log-domain stats, multiple thresholds; fit mixture on typical Telugu before perturbing.
Pause location: between-utterance pauses in MCI-AD (Pistono 2016) — needs transcripts.
Fillers: Telugu inventory unknown; synthesis via TTS filled-pause phone (Dall 2016 SSW9; Székely 2019).
Lexical diversity: MATTR, idea density (Snowdon 1996); agglutinative Telugu inflates types; Bengali AD fewer pronouns (opposite English) (Bose 2021 10.3389/fnagi.2021.707628) — transfer caution. Not acoustic-transformable.
Syllable nuclei detection (de Jong & Wempe 2009 10.3758/BRM.41.2.385) validated Dutch only -> Telugu unverified.
Datasets: ADReSS 2020, DementiaBank (Pitt), DementiaBankHindi, Indic dementia pilot (Telugu/Tamil/Hindi, translated, unverified), Bengali AD (6+8). No Telugu clinical ageing corpus.
For Vaani: describe, don't attribute.

## Agent 5 — Synthesis methods (condensed by lead, 2026-11-26)
Key: most synthetic atypical speech work = ASR augmentation; few perceptual tests. ASR gains ≠ realism.
M1 tempo/speed/WSOLA/PSOLA + pauses: ASR ESTABLISHED (Vachhani 2018 10.21437/Interspeech.2018-1751: +4.24 % tempo, +2 % speed abs; Geng 2020 10.21437/Interspeech.2020-1161: speed best, −2.92 % abs WER UASpeech; papers disagree tempo vs speed). NO perceptual realism evidence. Matches our "poor audio quality" finding. Noise-filled pauses = channel artefact a classifier can exploit.
M2 VTLP / envelope temporal smoothing: VTLP ASR weak; DCGAN spectro-temporal (Jin 2021 arXiv 2108.00899) −3.05 % abs WER but needs paired data; envelope smoothing as imprecise articulation = HYPOTHESIS (no listener evidence); may sound like codec damage on narrowband audio.
M3 WORLD F0/ap, jitter/shimmer/tremor: deterministic CPU, no atypical data; no listener validation for presbyphonia; jitter/shimmer contested age markers; WORLD may buzz on noisy narrowband; DDSP-QbE (Ghosh 2024 arXiv 2410.15500) preserves 12 clinical attributes (anonymisation). pyworld designed ≥16 kHz; Harvest better at low SNR.
M4 VC healthy->atypical: CycleGAN-VC+PSOLA (Halpern 2021 arXiv 2107.00308; objective only; ~10 min non-parallel per target); DuTa-VC diffusion (Wang 2023 arXiv 2306.10588; 2 SLPs: severity/type preserved); seq2seq (Huang 2021 arXiv 2110.08213; 3 SLPs forced choice); autoencoder VC (Illa 2021 SSW, arXiv 2106.08427: naturalness similar to real for low/mid intelligibility); cross-lingual VC UASpeech->FLEURS, Tamil ASR gains (Li 2025 arXiv 2505.14874 — only Dravidian evidence, ASR); kNN-VC (Baas 2023 arXiv 2305.18975, MIT, English vocoder) untested on atypical. LEAKAGE: rebuild per LOSO fold; VC copies channel.
M5 TTS severity/pause: Soleymanpour 2022 (arXiv 2201.11571; 2023 arXiv 2308.08438): 12.2 % WER + 6.5 % from controls; dysarthricness ≈ real at high severity; few-shot FS2 insufficient quality (Hermann 2023); DARS (arXiv 2603.01369) MCD 4.29; Lillini 2026 Grad-TTS (10.1007/s11517-026-03705-x); Moell 2025 voice cloning: 1 SLP misjudged 30 % synthetic as real. Needs transcribed dysarthric data; Telugu base TTS: AI4Bharat Indic-TTS, smtiitm Fastspeech2_HS (licences conflicting, Telugu unverified); no per-speaker diagnosis -> no severity supervision.
M6 disfluency injection: LibriStutter (secondary); audio injection low quality (LLM-Dys 2505.22029); Stutter-TTS F1 0.63–0.84, −5.7 % rel WER.
Evaluation protocols: SLP judgements (1–3 raters typical); naive MOS confounds severity with unnaturalness; classifier-based realism (Vachhani severity classifier); objective frameworks (Halpern); WER dominant but usefulness-only. Recommended: blind forced-choice real vs synthetic + SLP severity with real controls; matched bandwidth.
Gaps: no Telugu/Dravidian perceptual validation; no diagnosis -> per-speaker conditioning only under LOSO; channel confound; no validated presbyphonia generator; kNN-VC on atypical untested.

## Perplexity report (user-supplied, 2026-11-26) — verification notes by lead

Source: user pasted "Deep Research Report: Stuttering, Age-Related Speech Changes, and Synthetic Speech Generation" (Perplexity).

## Checked against PubMed (2026-11-26)
| Perplexity citation | Verified record | Verdict |
|---|---|---|
| "Roy, N., et al. (2020)" How Does Our Voice Change as We Age? DOI 10.1044/2019_JSLHR-19-00099 | Rojas S, Kefalianos E, Vogel A. JSLHR 2020;63(2):533-551, PMID 32083980 | DOI + findings correct; AUTHOR WRONG |
| "Ingham, R. J., et al. (2004)" Measurement of Stuttering in Adults DOI 10.1044/1092-4388(2004/080) | O'Brian S, Packman A, Onslow M, O'Brian N. JSLHR 2004;47(5):1081-7, PMID 15603463 | DOI + finding correct; AUTHOR WRONG |
| "Brabenec, L., et al. (2023)" Hypokinetic Dysarthria in PD narrative review, PMC10600629 | Atalar MS, Oguz O, Genc G. Sisli Etfal Hastan Tip Bul 2023;57(2):163-170, DOI 10.14744/SEMB.2023.29560 | AUTHOR + journal WRONG. Abstract lists monopitch, reduced/monotonous loudness, breathy/hoarse voice, INCREASED speech rate, rapid phoneme repetition, imprecise consonants. "6 s minimum phonation time optimal" NOT in abstract -> unverified |
| PMC11933288 "Speech and language biomarkers for PD" (2025) | Cao F, Vogel AP, Gharahkhani P, Renteria ME. NPJ Parkinsons Dis 2025;11:57, DOI 10.1038/s41531-025-00913-4 | exists; narrative review; detailed feature list (VSA, VAI, VOT, CPP...) not verifiable from abstract |
| Presbyphonia SR (jitter -0.62 %, shimmer -1.05 %) PMC11228133 | Saccente-Kennedy B et al. J Voice 2026;40(3):706-732 (epub 2024), DOI 10.1016/j.jvoice.2023.12.010 | numbers CORRECT; also MPT +5.37 s; presbyphonia most common dysphonia cause >65 y |

## Content issues to resolve in synthesis
- PD speech rate: Perplexity implies slow/longer pauses; Atalar 2023 lists INCREASED rate / short rushes -> rate in PD is heterogeneous; do not model PD as "slow".
- "Telugu ... tone-like perceptual expectations": unsupported; Telugu is not tonal -> drop/flag.
- Pipeline assumes Telugu forced alignment: needs transcripts; our atypical intended transcripts exist (Telugu script) but verbatim fields are mixed-script; typical source set has transcripts for only 443/4,914 usable clips.
- uts.edu PDF reference (dated 2026-07-28) for weighted SLD: unverified lecture notes, not peer-reviewed.
- Agrees with lead's prior: condition-specific parameterised synthesis; Vaani = exploratory only; no diagnosis inference; perceptual validation required.
