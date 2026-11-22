import demoAudioUrl from '../assets/demo-sample.wav?url';
import { base64WavToObjectUrl, blobTo16kMonoWav } from './wav';

/**
 * Boundary between the frontend and the speech-enhancement engine.
 *
 * When `VITE_ENHANCE_API_URL` is set, the recording is converted to 16 kHz mono
 * WAV and sent to the local Intelli-Speech inference server (see `server/`).
 * When it is not set, this module returns a bundled, clearly-labelled
 * demonstration sample and never processes the supplied recording.
 */
export interface EnhancementInput {
  audioBlob: Blob;
  mimeType: string;
}

export interface EnhancementResult {
  audioUrl: string;
  isDemo: boolean;
  disclosure: string;
  noiseType?: string;
  noiseProbs?: Record<string, number>;
  modelVersion?: string;
  processingMs?: number;
}

interface EnhanceApiResponse {
  audio_wav_base64: string;
  sample_rate: number;
  noise_type: string;
  noise_probs: Record<string, number>;
  model_version: string;
  duration_sec: number;
  processing_ms: number;
}

const DEMO_LOADING_DELAY_MS = 700;

function getApiUrl(): string | undefined {
  const raw = import.meta.env.VITE_ENHANCE_API_URL as string | undefined;
  const trimmed = raw?.trim();
  return trimmed ? trimmed.replace(/\/+$/, '') : undefined;
}

/** True when a real enhancement server is configured for this build. */
export function isRealEngineConfigured(): boolean {
  return getApiUrl() !== undefined;
}

async function enhanceDemo(): Promise<EnhancementResult> {
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

async function readErrorDetail(response: Response): Promise<string | undefined> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === 'string' ? body.detail : undefined;
  } catch {
    return undefined;
  }
}

async function enhanceRemote(apiUrl: string, input: EnhancementInput): Promise<EnhancementResult> {
  let wav: Blob;
  try {
    wav = await blobTo16kMonoWav(input.audioBlob);
  } catch {
    throw new Error('Your recording could not be prepared for enhancement. Please record again.');
  }

  const form = new FormData();
  form.append('file', wav, 'recording.wav');

  let response: Response;
  try {
    response = await fetch(`${apiUrl}/enhance`, { method: 'POST', body: form });
  } catch {
    throw new Error(
      'The enhancement server could not be reached. Check that it is running and try again.',
    );
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response);
    if (response.status === 503) {
      throw new Error('The enhancement model is not available on the server right now.');
    }
    if (response.status === 400) {
      throw new Error(detail ?? 'The recording could not be enhanced. Please try a shorter clip.');
    }
    throw new Error('Enhancement failed on the server. Please try again.');
  }

  let body: EnhanceApiResponse;
  try {
    body = (await response.json()) as EnhanceApiResponse;
  } catch {
    throw new Error('The enhancement server returned an unexpected response.');
  }
  if (typeof body.audio_wav_base64 !== 'string' || body.audio_wav_base64.length === 0) {
    throw new Error('The enhancement server returned no audio.');
  }

  return {
    audioUrl: base64WavToObjectUrl(body.audio_wav_base64),
    isDemo: false,
    disclosure: `Enhanced by Intelli-Speech research model ${body.model_version} (ZF-Net + GMM, v1). Research prototype; results may vary.`,
    noiseType: body.noise_type,
    noiseProbs: body.noise_probs,
    modelVersion: body.model_version,
    processingMs: body.processing_ms,
  };
}

export async function enhanceAudio(input: EnhancementInput): Promise<EnhancementResult> {
  const apiUrl = getApiUrl();
  if (apiUrl === undefined) {
    return enhanceDemo();
  }
  return enhanceRemote(apiUrl, input);
}
