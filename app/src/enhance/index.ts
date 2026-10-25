import demoAudioUrl from '../assets/demo-sample.wav?url';

/**
 * Boundary between the frontend and the future speech-enhancement engine.
 *
 * The real engine is not connected yet. This module currently returns a bundled
 * demonstration sample and never processes the supplied recording. When a real
 * engine becomes available, replace the implementation of `enhanceAudio` while
 * keeping this function signature so the UI does not need to change.
 */
export interface EnhancementInput {
  audioBlob: Blob;
  mimeType: string;
}

export interface EnhancementResult {
  audioUrl: string;
  isDemo: true;
  disclosure: string;
}

const DEMO_LOADING_DELAY_MS = 700;

export async function enhanceAudio(_input: EnhancementInput): Promise<EnhancementResult> {
  // Demo behaviour only: wait briefly so the interface can show its loading
  // state, then return a sample clip. No model is called and the user's
  // recording is not read, uploaded, or modified.
  await new Promise((resolve) => setTimeout(resolve, DEMO_LOADING_DELAY_MS));

  return {
    audioUrl: demoAudioUrl,
    isDemo: true,
    disclosure:
      'This is a sample clip, not your recording, and no speech enhancement has been applied to it. The real enhancement engine is not connected yet.',
  };
}
