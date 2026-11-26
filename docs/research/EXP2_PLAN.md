# Experiment 2 plan: articulation, formant dynamics, rhythm and pause structure

**Status:** v0.2.1, 2026-11-26. v0.2 was revised after an independent methods review (see §11). v0.2.1 adds the user's clarifications and fixes operational details **before any Stage A data was processed** (§13). **Stage A implemented; Stage B not implemented.**

> **Scope statement.** This experiment measures acoustic articulation (vowel-space use, formant dynamics) and pause structure in atypical Telugu speech. It does **not** establish, test or imply stuttering, presbyphonia, Parkinsonian speech, dysarthria type or any other clinical diagnosis, for any speaker or group.
**Basis:** `docs/research/atypical_speech_research_synthesis.md` (roadmap step 1).
**Scope:** slurred (dysarthric) speech; age-related voice changes are secondary.

## 1. Question

Do the 7 Vaani atypical Telugu speakers differ from typical Telugu speakers in **articulation**, beyond the timing differences Exp 1 confirmed? Articulation here means vowel-space use and formant movement. The answer decides which articulation transform layers have evidence in our own data.

**Out of scope:**
- No diagnosis, cause, age or sex is inferred for any speaker.
- A null result is not treated as proof of absence. It is classified with equivalence tests (§7).

## 2. Two stages

1. **Stage A: measurement validity, on typical speech only.** Establish that the label-free measures track labelled vowel measures and are robust to our channel. Stage B runs only for measures that pass Stage A.
2. **Stage B: group comparison.** Atypical vs typical speakers, using the pre-registered primary analysis.

## 3. Pre-registered hypotheses (Stage B)

| ID | Hypothesis (direction vs typical) | Primary measure | Unlocks layer | In FDR family |
| --- | --- | --- | --- | --- |
| H1 | Reduced vowel-space use | `aavs_log` | Formant warp toward centroid | Yes |
| H2 | Smaller formant excursion per syllable | `f2_excursion_st_median` | Envelope temporal smoothing | Yes |
| H3 | Slower formant movement (rate-adjusted within the overlapping rate range) | `f2_speed_fit_oct_s` | Envelope temporal smoothing | Yes |
| H4 | Higher share of long pauses (> 1 s), on duration-matched windows | `pause_long_share_1s_win` | Pause mixture (timing layer) | Yes. A **replication/extension** of Exp 1, not independent evidence |
| C1 | Lower 3–5 Hz envelope modulation | `ems_3_5hz_ratio` (full clip and speech-only) | None | **No.** A manipulation check, largely implied by the slower rate |

F0 SD and range, intensity SD and CPPS are re-reported from Exp 1, not re-tested.

## 4. Data

| Set | Role | Size | Transcripts |
| --- | --- | --- | --- |
| Vaani Atypical Telugu | Target | 189 clips, 7 speakers, 109 min | **Intended** meaning, not verbatim (dataset card), so alignment is unreliable |
| `typical_controls_v2` | **Primary control** (same platform and task as the atypical corpus, and the Exp 1 control) | 126 clips, 40 speakers, 20 F / 20 M | Yes, verbatim crowd transcripts |
| `vaani_telugu_transcribed` | Replication control | 85 clips, 41 speakers | Yes, mixed script |
| `synth_source_v1_full`, train split, sampled | Large typical reference: norms, sex-stratified bands, reliability, power simulation | ≤ 1,200 clips, ≤ 400 speakers | 443 usable clips have transcripts |

All four sets come from the Vaani crowd-sourced phone platform. Atypical clips average about 35 s and typical clips about 6 s, so length-sensitive measures use duration-matched 5.8 s windows (§6.5).

## 5. Signal conditioning and channel descriptors

1. **Channel cutoff descriptor (new).** `upper_cutoff_hz` = the highest frequency where the smoothed long-term spectrum lies within 40 dB of its maximum. `low_ratio_db` = energy below 300 Hz relative to 300–1000 Hz. The Exp 1 `effective_bandwidth_hz` (99 % energy point) measures spectral balance, not the channel cutoff, so it is kept only as a confound descriptor.
2. **Inclusion.** A clip is used for formant measures if `upper_cutoff_hz` ≥ 3400 Hz. Sensitivity runs at 3000 and 3800 Hz. Speakers with fewer than 3 eligible clips or fewer than 300 selected frames are marked **insufficient**, never imputed.
3. **Harmonisation.** Every included clip, atypical and control, is low-pass filtered at 3400 Hz and high-pass filtered at 100 Hz before formant analysis. This gives one common analysis band.
4. **Confound descriptors** used in every check: `upper_cutoff_hz`, `low_ratio_db`, `effective_bandwidth_hz`, `rms_dbfs`, an SNR estimate (from the source-quality screen) and clip duration.

## 6. Measures (feature version `exp2-features-0.2`)

### 6.1 Formant tracking

- **Settings.** Praat Burg via parselmouth: 3 formants, a **fixed** ceiling of 3700 Hz, 25 ms window, 10 ms step.
- **Ceiling sensitivity.** Pre-registered runs at 3450 and 3950 Hz. The ceiling is the same for every speaker, so it cannot differ by group.
- **Rejected alternative.** Per-speaker variance-minimising ceilings were rejected: without vowel labels they minimise vowel-space size itself, which is circular with H1.
- **Frame selection:**
  - voiced;
  - within ±30 ms of an Exp 1 acoustic syllable nucleus;
  - intensity at or above the speaker's median speech-frame intensity;
  - F1 within 200–1000 Hz;
  - F2 within 600 Hz–3300 Hz (the band limit; the share of frames clipped at each bound is reported);
  - B1 < 400 Hz and B2 < 600 Hz.
- **Nasal-murmur exclusion.** Frames where energy below 500 Hz exceeds 500–3400 Hz energy by more than 15 dB, together with F1 bandwidth above 300 Hz, are dropped. This is a heuristic; its share is reported.
- **Selection diagnostics by group:** nuclei per second, selected frames per second, and frames per clip.

### 6.2 Vowel space (no labels; log domain)

In log formant units, vocal-tract-length scaling is an additive shift, as in Nearey-style log-mean normalisation, so area and covariance measures need no sex labels. Lobanov z-scoring is **not** used, because it removes the dispersion H1 tests.

| Measure | Definition | Status |
| --- | --- | --- |
| `aavs_log` | sqrt(det Σ), where Σ is the covariance of (ln F1, ln F2) over selected frames, per recording. Log-unit variant of Whitfield & Goberman | Primary (if Stage A passes) |
| `vsa_hull_log` | Convex-hull area of the central 90 % (Mahalanobis-trimmed) of the (ln F1, ln F2) cloud | Secondary |
| `diag_extent_log` | P95 − P05 of (ln F2 − ln F1) and of (ln F1 + ln F2): diagonal extremes of the cloud, approximating the /i/–/a/ and /u/–/a/ axes | Exploratory |

The quantile-based FCR proxy from v0.1 was dropped: marginal quantiles do not identify corner vowels.

### 6.3 Formant dynamics

- **Contour fitting.** Within each selected voiced run of at least 60 ms, ln F2 is fitted with a 4-coefficient DCT. All dynamics come from the fit, never from raw frame differences.
- `f2_excursion_st_median` (**primary for dynamics**): the range of the fitted F2 in semitones between consecutive nuclei within a run.
- `f2_speed_fit_oct_s`: the median absolute slope of the fitted ln F2 (octaves per second). Only transitions with extent of at least 1 semitone over at least 20 ms count.
- **Rate adjustment (H3).** Fit speed ~ articulation rate on typical recordings, using only the articulation-rate interval where both groups overlap. Atypical recordings outside that interval are excluded from the adjusted analysis. Rate is a possible mediator, so **adjusted and unadjusted results are reported with equal weight**.
- **Exploratory:** a stiffness ratio, speed / excursion.

### 6.4 Rhythm (C1)

- **Method.** Hilbert envelope of the full band and of octave bands at 500, 1000 and 2000 Hz. The 250 Hz band is reported with a flag, because it sits near the phone high-pass. The envelope is low-passed at 10 Hz and resampled to 80 Hz.
- **Measures:** peak frequency, and the 3–5 Hz energy ratio over 0.5–10 Hz.
- **Input.** Computed on full clips and on **speech-only** concatenated segments (pauses removed), after Liss et al. 2010.

### 6.5 Pause structure (H4)

- **Windows.** Computed **only on duration-matched 5.8 s windows** for both groups. Only pauses fully inside a window count; pauses are at least 100 ms, from the Exp 1 VAD.
- **Model.** Primary: a mixed logistic regression, pause > 1 s ~ group + (1 | speaker). Per-speaker shares are reported only for speakers with at least 20 contained pauses.
- **Mixture fits.** Log-duration Gaussian mixtures are fitted per speaker where there are at least 50 pauses, and hierarchically pooled otherwise (2–3 components by BIC). They are descriptive and feed the pause-model parameters.

### 6.6 Descriptive covariates

F0 median, as a rough age/sex-related covariate (not inferred as sex), plus all channel descriptors from §5.

## 7. Analysis plan

### 7.1 Primary analysis cell (fixed in advance)

- Cutoff gate 3400 Hz and ceiling 3700 Hz.
- Primary control `typical_controls_v2`.
- Unit: per-recording measures; full clips for H1–H3, windows for H4.
- Speaker medians for the Exp 1-style catalogue.

All other cells are sensitivity analyses outside the FDR family.

### 7.2 Tests

- **Exp 1 statistics on speaker medians:** Cliff's δ, Hedges g, speaker-level permutation p, bootstrap over speakers.
- **FDR.** BH correction over **H1–H4** (4 tests).
- **Class-1 criteria (unchanged from Exp 1).** All of:
  - at least 5 of 7 speakers outside the control P10–P90 range, in the hypothesised direction;
  - leave-one-speaker-out sign stability;
  - q < 0.10;
  - |δ| ≥ 0.474;
  - max |Spearman ρ| with every §5 descriptor < 0.5;
  - passing the Stage A channel tolerance.
- **Equivalence.** TOST with bounds δ = ±0.33 (Lakens 2017), on speaker medians via bootstrap. A measure that fails class 1 is classed as **equivalent**, **inconclusive** or **different but below criteria**. "Not supported" is never used for an inconclusive result.
- **Mixed-effects sensitivity.** A linear mixed model at recording level: measure ~ group + upper_cutoff + low_ratio + SNR + duration + (1 | speaker), with Satterthwaite degrees of freedom. It must agree in direction with the primary result.
- **Replications required for robust class 1:**
  1. against `vaani_telugu_transcribed`;
  2. on duration-matched windows (H1–H3);
  3. against both the female-only and male-only P10–P90 bands of the large reference.

### 7.3 Power (simulated before Stage B)

- **Simulation.** Simulate 7 atypical vs 40 control speakers using the large reference's between-speaker and within-speaker variance per measure.
- **Effect sizes.** Inject shifts of d = 0.5, 0.7, 1.0. The d ≈ 0.7–1.0 range is from automated VSA and F2 slope in ALS picture descriptions (Shellikeri et al. 2024; English, so treated as a prior).
- **Output.** For each measure, the probability of meeting class-1 criteria. It is reported before Stage B, so negative results can be interpreted.

## 8. Stage A: measurement validity (gate 0)

Each primary measure must pass every test. A measure that fails is exploratory and cannot unlock a layer.

1. **Concurrent validity (primary gate).**
   - Align the verbatim transcripts of `typical_controls_v2` and the transcribed reference clips with **MMS CTC forced alignment** (torchaudio `MMS_FA`, romanised Telugu).
   - Gate on alignment confidence, and hand-check 30 utterances for vowel boundary error.
   - Compute labelled corner-vowel FCR and log-VSA per speaker, from Telugu /i a u/ including long vowels.
   - **Pass:** speaker-level Spearman ρ ≥ 0.5 between `aavs_log` / `vsa_hull_log` and labelled VSA, and ρ ≤ −0.5 against FCR.
2. **Frame purity.** On aligned clips, at least 80 % of selected frames must fall inside aligned vowels. Otherwise the measured purity is stated and the measure is demoted.
3. **Synthetic recovery.** Cascade-formant vowels (F0 100–300 Hz), passed through the same band harmonisation and fixed ceiling: F1 and F2 median error within ±10 %.
4. **Reliability.** Split-half speaker-median ICC(2,1) ≥ 0.6 in the large reference **and** within the atypical speakers. The atypical speakers have more clips per speaker.
5. **Channel robustness.** Control clips are passed through:
   - low-pass at 3000/3400/4000 Hz;
   - high-pass at 300 Hz;
   - 8 kHz narrowband resampling (G.711-like);
   - Opus at 8 and 12 kbps;
   - AMR-NB if the Colab ffmpeg build supports it.

   **Pass:** each condition shifts the measure by less than **0.25 × the between-speaker SD of the controls**. This tolerance is fixed in advance.
6. **Transform detectability (not validity).**
   - WORLD envelope warp toward the centroid, and envelope temporal smoothing, must move the measures monotonically.
   - This only shows the measures can **detect** those transforms. It cannot count as evidence for H1–H3, and it cannot unlock those layers by itself.
   - WSOLA rate change is reported descriptively only. It preserves formant targets, so it is not a natural-rate test.
7. **Lexical comparability.** Vowel-category proportions from aligned control transcripts are compared with those implied by the atypical intended transcripts (text only). A large difference is flagged as a cloud-composition confound for H1.

## 9. Decision rules

| Outcome | Consequence |
| --- | --- |
| Robust class 1 | Build the layer (roadmap step 3), fitted per LOSO fold |
| Speaker-specific (1–2 speakers) | Per-synthetic-speaker random option only |
| Equivalent (TOST) | No layer: evidence of no meaningful difference |
| Inconclusive or underpowered | No layer yet; flagged for labelled clinical data (roadmap step 7) |
| Fails Stage A, or channel-sensitive | No layer; recorded as a measurement limitation |

If no articulation measure reaches class 1, the listener test (roadmap step 5) uses timing plus prosody only, and articulation layers wait for labelled clinical Telugu data.

**Parallel step:**
- A small blinded rating of articulatory imprecision by 3 or more native listeners, on duration- and channel-matched atypical and control clips.
- This anchors the term "slurred" for this data.
- It is a separate task and does not gate Stage B.

## 10. Implementation (after approval of this plan)

**New files:**
- `intelli_speech/articulation_features.py`
- `intelli_speech/alignment.py` (MMS wrapper)
- `scripts/exp2_stage_a.py`
- `scripts/extract_features_exp2.py`
- `scripts/analyze_exp2.py`, which reuses `analyze_exp1.analyse` and adds TOST, a mixed model (statsmodels) and power simulation
- `EXP2_ARTICULATION.ipynb`
- synthetic-signal tests

Exp 1 code and outputs stay unchanged.

**Compute (free Colab):**
- formants and EMS on CPU: about 30–45 min;
- MMS alignment on a T4 GPU for about 1,400 transcribed typical clips: under 30 min (estimate, unverified);
- codec round-trips through ffmpeg.

## 11. Changes from v0.1 (methods review, 2026-11-26)

An independent reviewer validated v0.1 against the literature. The verdict was **proceed with changes**. Adopted changes:

| # | Change | Reason |
| --- | --- | --- |
| 1 | Fixed 3-formant / 3700 Hz ceiling replaces the per-speaker variance-minimising ceiling, with ±250 Hz sensitivity | Without labels, the variance rule minimises vowel-space size (circular with H1). Escudero et al. 2009 rely on within-vowel variance. |
| 2 | Common analysis band (100–3400 Hz) for all clips; new `upper_cutoff_hz` and `low_ratio_db` descriptors | Phone transmission raises F1 (about 29 % in mobile recordings, Byrne & Foulkes 2004; telephone F1 effect per Künzel 2001). The 99 %-energy bandwidth is not a cutoff. |
| 3 | MMS forced-alignment validity gate on verbatim control transcripts (ρ ≥ 0.5 vs labelled VSA/FCR) | Label-free measures need concurrent validity (Sandoval 2013; Sapir 2010). |
| 4 | WORLD warp and smoothing tests relabelled "transform detectability" | Using the synthesis transforms to validate the measures is circular. |
| 5 | H4 only on duration-matched windows, mixed logistic model | 6 s control clips censor pauses over 1 s; per-speaker pause counts are small. |
| 6 | Primary cell, TOST equivalence, simulated power, mixed-model sensitivity | 7 speakers: negative results need formal footing (Lakens 2017). |
| 7 | EMS removed from the FDR family; computed on speech-only segments too | EMS tracks rate (Liss et al. 2010). |
| 8 | Excursion primary for dynamics; speed from DCT fits, minimum extent, rate adjustment only within overlapping range | Frame-difference speed is jitter-dominated; rate is a mediator and atypical rates fall outside the control range. |
| 9 | Frame purity ≥ 80 % and selection diagnostics by group; nasal-murmur exclusion | The nucleus detector was validated on syllable counts (Dutch), not vowel identity. |
| 10 | Channel robustness adds 300 Hz high-pass and narrowband codecs; tolerance fixed in advance | The original tested only the upper cutoff, with a tolerance set after seeing the data. |

**Lead's additions** (not raised by the reviewer):
- The atypical transcripts record **intended** meaning, so alignment-based labelled measures run on controls only.
- The bandwidth gate now uses a true cutoff descriptor, because the 99 %-energy point would have excluded most typical clips (median about 2.1 kHz).

**Not adopted now:**
- Phonological-class posteriors, which are trained on Spanish.
- Consonant spectral moments, which need aligned consonants on atypical speech, and we have no verbatim transcripts for it.
- External checking on Tamil SSNCE, whose licence is not yet obtained.

All three are listed for later work.

## 12. Known risks and limits

- **Weak validity evidence.** Unlabelled formant-cloud measures are validated on English and small samples, and Telugu's 5 vowel qualities × length may shape the cloud differently. Stage A tests this directly on Telugu controls.
- **Alignment quality.** MMS quality for Telugu is unverified, hence the hand-check.
- **Sex.** It is unknown for atypical speakers. Log-domain measures reduce, but do not remove, vocal-tract differences.
- **Sample size.** 7 speakers, 2 of them with only 5 clips. Power is limited, and the simulation (§7.3) states it before any result.
- **Rate and vowel space.** Slower or longer vowels can enlarge vowel space: in Tykalova 2021, older speakers' larger vowel space was partly related to longer vowel duration. This works against H1, so a positive H1 is conservative.
- **Task.** Image-prompted speech only.

## 13. v0.2.1 clarifications (fixed before any Stage A data was processed)

These are the user's clarifications, plus operational details that v0.2 left open. None relaxes a v0.2 threshold. The added thresholds are fixed now, before any result exists.

### 13.1 Scope and reliability wording

- **Scope.** See the scope statement at the top of this file.
- **Typical reliability (part of the Stage A gate).** §8.4 split-half ICC(2,1) ≥ 0.6 is computed on **typical speakers only**: the large reference, with `typical_controls_v2` reported alongside.
- **Atypical reliability (a separate feasibility check, not a gate).** The split-half ICC and the data sufficiency of the 7 atypical speakers are reported separately. To avoid unblinding the group comparison, that run writes **only ICCs and counts**: no per-recording or per-speaker atypical feature values.

### 13.2 Which Stage A tests apply to which measure

| Measure | Synthetic recovery | Concurrent validity (+ alignment acceptance) | Frame purity | Typical reliability | Channel robustness | Transform detectability |
| --- | --- | --- | --- | --- | --- | --- |
| `aavs_log`, `vsa_hull_log` (H1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (envelope warp) |
| `f2_excursion_st_median` (H2), `f2_speed_fit_oct_s` (H3) | ✓ | — (no labelled counterpart defined in v0.2) | ✓ | ✓ | ✓ | ✓ (temporal smoothing) |
| `pause_long_share_1s_win` (H4) | — | — | — | ✓ | ✓ | — |

**Status per measure:**
- **pass:** every applicable test passes.
- **fail:** any applicable test fails.
- **incomplete:** a test could not run for technical reasons (for example a missing codec or `pyworld`). An incomplete measure cannot unlock a layer.

### 13.3 MMS alignment: exclusion rules and the manual validation protocol

**Text cleaning.**
- Bracketed spans (`<…>`, `{…}`, `[…]`) are removed.
- An utterance is **excluded** if:
  - a bracketed span contains Telugu letters (the spoken content is uncertain);
  - it contains digits;
  - Latin letters are more than 30 % of its letters.

**Alignment confidence.**
- Token posterior = the mean CTC posterior of the token over its frames.
- An utterance with mean token posterior **< 0.40** is excluded from labelled measures and purity.
- Vowel tokens with posterior **< 0.20**, or shorter than **40 ms**, are excluded.

**Labelled measures.**
- Corner vowels: /i/ = {i, ii}, /a/ = {a, aa}, /u/ = {u, uu}. Short and long vowels are pooled; a long-vowel-only sensitivity uses {ii, aa, uu}.
- A speaker needs at least 3 accepted tokens per corner category.
- Per-token value = median ln F1 / ln F2 over the central 40 % of the token, using the same band and tracker as §6.1 (range and bandwidth filters only).
- FCR = (F2u + F2a + F1i + F1u) / (F2i + F1a) in Hz.
- log-VSA = triangle area of the corner means in (ln F1, ln F2).

**Manual check sample (30 utterances, deterministic).**
- Taken from successfully aligned typical utterances.
- At least 10 distinct speakers, at most 3 per speaker.
- Every vowel category a, aa, i, ii, u, uu, e, ee, o, oo appears in at least 3 selected utterances, where available. Short and long vowels are therefore both covered.
- At least 5 utterances come from the lowest-confidence quartile.
- Up to 6 vowel tokens per utterance (at most 180 in total) are chosen to balance categories.

**What the checker does.** In Praat, using the exported TextGrid, the checker records for each token:
- the corrected start and end, or "=" if the automatic boundary is correct;
- whether a vowel is present;
- whether the category is correct.

For each utterance, the checker also marks gross misalignment: a word misplaced by more than its own length.

**Reported:**
- the median and P90 absolute boundary error;
- the share of boundaries within 20 ms and within 50 ms, by category and by short vs long;
- the utterance failure rate (gross misalignment), plus the automatic failure counts by reason;
- category agreement;
- Spearman ρ between confidence and error.

**Acceptance (all required before the concurrent-validity result counts).**
- Median absolute boundary error ≤ 30 ms.
- At least 80 % of boundaries within 50 ms.
- Utterance failure rate ≤ 20 %.
- Category agreement ≥ 85 %.

If acceptance fails, the H1 measures remain exploratory. The thresholds are not revisited.

### 13.4 Other operational definitions

**Channel cutoff.** `upper_cutoff_hz` = the highest frequency whose Welch PSD (1024-point, smoothed over about 200 Hz) lies within 40 dB of the PSD maximum.

**Channel robustness shift.** For each condition, the shift is the median over control recordings of (condition − original). It passes if |shift| < 0.25 × the SD of the original speaker medians across `typical_controls_v2` speakers.

**Conditions.**
- Numerical (always run): low-pass at 3000/3400/4000 Hz; high-pass at 300 Hz; 8 kHz resampling with G.711 μ-law.
- ffmpeg codecs: Opus at 8/12 kbps, and AMR-NB at 12.2 kbps if the encoder exists.
- A codec that is unavailable is reported as "not run", which makes the result **incomplete**, not a pass.

**H4 windows.**
- Consecutive 5.8 s windows. A final partial window is kept if it is at least 4.0 s.
- A clip shorter than 5.8 s counts as one window if it is at least 4.0 s.
- Only pauses fully inside a window, and at least 100 ms long, are counted.

**Synthetic recovery.** A cascade formant synthesiser generates a grid:
- F1 ∈ {300, 450, 650, 800} Hz;
- F2 ∈ {900, 1300, 1800, 2300} Hz, with F2 ≥ F1 + 300;
- F0 ∈ {100, 150, 200, 250, 300} Hz.

The pass rule is the median |relative error| ≤ 10 % for F1 and F2 separately, over the central 60 % of each vowel.

**Lexical comparability (a flag, not a gate).** Corner-category shares (pooled short and long) are compared between aligned control transcripts and atypical intended transcripts, as text only. The flag is raised if any share differs by more than 0.10 absolute, or if the Jensen–Shannon distance exceeds 0.10.

**Reference sample.**
- `synth_source_v1_full`, **train split only**: speakers with at least 3 usable clips, up to 400 speakers, 3 clips each, chosen by salted hash.
- All usable **train-split** clips with transcripts are used for alignment.
