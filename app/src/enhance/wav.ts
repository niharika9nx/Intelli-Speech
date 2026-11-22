/**
 * Browser-side audio helpers for talking to the Intelli-Speech inference server.
 */

export const TARGET_SAMPLE_RATE = 16000;

type AudioContextConstructor = typeof AudioContext;

function getAudioContextConstructor(): AudioContextConstructor {
  const ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: AudioContextConstructor }).webkitAudioContext;
  if (!ctor) {
    throw new Error('This browser cannot decode audio for enhancement.');
  }
  return ctor;
}

async function decode(blob: Blob): Promise<AudioBuffer> {
  const Ctor = getAudioContextConstructor();
  const context = new Ctor();
  try {
    const data = await blob.arrayBuffer();
    return await context.decodeAudioData(data);
  } finally {
    void context.close();
  }
}

async function toMono16k(buffer: AudioBuffer): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil(buffer.duration * TARGET_SAMPLE_RATE));
  // A mono destination downmixes all input channels (speaker down-mix rules).
  const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

function encodePcm16Wav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/** Decode any browser-supported recording and re-encode it as 16 kHz mono 16-bit PCM WAV. */
export async function blobTo16kMonoWav(blob: Blob): Promise<Blob> {
  const decoded = await decode(blob);
  const mono = await toMono16k(decoded);
  return encodePcm16Wav(mono, TARGET_SAMPLE_RATE);
}

/** Turn a base64-encoded WAV into an object URL for an <audio> element. */
export function base64WavToObjectUrl(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}
