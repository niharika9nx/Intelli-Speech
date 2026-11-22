# Targets — Intelli-Speech status

**As of:** 2026-11-22. Detailed evidence is in `PROJECT_NOTES.md` §0.

## Done

### 1. Noise-reduction system (v1)
- Built the GMM + ZF-Net model, the training and evaluation scripts, the inference server and the frontend connection. Pushed in `102ace8`.
- This handles background noise only, which is a separate goal from atypical speech.

### 2. Dataset checks (Experiment 0 + follow-up)
- **Atypical Telugu corpus:** 189 clips from 7 speakers, about 109 min. No broken files and no duplicates.
- **Clipping:** almost none; the clips were simply normalised to full volume.
- **Transcriptions:**
  - The "intended" text is all Telugu script.
  - The "verbatim" text mixes Telugu and romanised Telugu with annotation tags, so the verbatim-vs-intended error rate is **not** an intelligibility measure.
- **Difficulty labels (Low/Medium/High):** the dataset doesn't say which way the scale runs. They describe the data only and are never used as severity.
- **Typical-speech comparison set:** downloaded from regular Vaani. 40 speakers (20 women, 20 men), 126 clips, 12.4 min, with source details recorded.

### 3. Speech characterisation (Experiment 1)
- Measured pauses, speaking rate, rhythm, pitch, voice quality and spectral features on 400 recordings.
- Compared speakers rather than individual clips, then ran three robustness checks.
- **Confirmed: all 7 atypical speakers differ from typical speakers in timing (14 features).**
  - They speak more slowly.
  - They pause more often, for longer, and less regularly.
  - More of their time is silence.
- **Not shared across speakers:** pitch, voice quality and spectral shape differ only for individual speakers.

### 4. Housekeeping
- The code is tested (156 tests pass) and pushed; the latest commit is `2a2c43d`.
- `PROJECT_NOTES.md` has a progress tracker with the results and their limitations.
- The opencode free models are assigned per role, and the rules for using Codex are in `AGENTS.md`.

## Left

### Phase 3: synthetic data and models
**Component roles (decided 2026-11-22):**
- **GMM:** identifies atypical speech and how strongly.
- **ZF-Net:** finds where the atypical events are, frame by frame, and does spectral clean-up.
- **DNN:** decides the correction settings.
- **DSP module:** makes the timing edits.

Each part's contribution will be proven by ablation.

1. **Source speech: regular Vaani (decided).**
   - Pilot: about 60 speakers and about 1,000 clips, in progress.
   - Full run: about 300 speakers, about 6,000 clips, about 10 h.
   - No overlap with the comparison speakers, a quality filter, and a train/validation/test split by speaker.
2. **Timing-change module:** slow the speech without changing pitch, and add and lengthen pauses. The settings are taken from the measured patterns, separately for each held-out speaker. Work on this has started, using test signals only.
3. **Pitch changes:** varied at random for each synthetic speaker, since there is no shared pitch pattern.
4. **Build the synthetic dataset,** kept clearly separate from the real recordings, with each sample's source recorded.
5. **HDNZF-Net:** build it with the roles above (GMM detector, ZF-Net frame-level localiser and spectral clean-up, DNN correction settings), trained on the synthetic pairs.

### Phase 4: evaluation
- Test on real atypical speakers, holding out one speaker at a time.
- Compare against an ASR→TTS baseline.
- Have native Telugu listeners rate intelligibility; the listeners still need to be found.

### Phase 5: prototype and write-up
- Connect the atypical-speech pipeline to the app, keeping noise reduction as a separate module.
- Write up the research report with methods, results and limitations.

### Small items
- One small fix to the analysis script is local only and not yet pushed.
- The notebooks with outputs (`AUDIT_V2`, `TRAIN_V1`, `MAINFILE`, `RECOVERY`) stay uncommitted, because their outputs include research-only speaker IDs.
