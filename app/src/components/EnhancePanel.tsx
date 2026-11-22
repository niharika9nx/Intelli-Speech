import type { EnhancementResult } from '../enhance';

export type DemoState =
  | { status: 'idle' }
  | { status: 'preparing' }
  | { status: 'ready'; result: EnhancementResult }
  | { status: 'error'; message: string };

interface EnhancePanelProps {
  state: DemoState;
  onEnhance: () => void;
  /** True when a real enhancement server is configured (see `isRealEngineConfigured`). */
  realEngine: boolean;
}

const NOISE_LABELS: Record<string, string> = {
  DKITCHEN: 'Kitchen',
  NPARK: 'Park',
  OOFFICE: 'Office',
  STRAFFIC: 'Street traffic',
  TBUS: 'Bus',
};

function noiseLabel(code: string): string {
  return NOISE_LABELS[code] ?? code;
}

function formatProcessingTime(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function DemoResult({ result }: { result: EnhancementResult }) {
  return (
    <div className="demo-result">
      <div className="notice notice--demo" role="note">
        <p>
          <strong>Demo mode.</strong> {result.disclosure}
        </p>
      </div>

      <h3>Demo audio sample</h3>
      <audio
        className="audio-player"
        controls
        preload="metadata"
        src={result.audioUrl}
        aria-label="Demo audio sample playback"
      >
        Your browser does not support audio playback.
      </audio>

      <p className="helper">
        When the real engine is connected, this area will play audio enhanced from your own
        recording.
      </p>
    </div>
  );
}

function RealResult({ result }: { result: EnhancementResult }) {
  const confidence =
    result.noiseType !== undefined ? result.noiseProbs?.[result.noiseType] : undefined;

  return (
    <div className="demo-result">
      <div className="notice" role="note">
        <p>{result.disclosure}</p>
      </div>

      <h3>Enhanced audio</h3>
      <audio
        className="audio-player"
        controls
        preload="metadata"
        src={result.audioUrl}
        aria-label="Enhanced recording playback"
      >
        Your browser does not support audio playback.
      </audio>

      {result.noiseType !== undefined && (
        <p className="helper">
          Detected background noise: <strong>{noiseLabel(result.noiseType)}</strong>
          {confidence !== undefined && ` (${Math.round(confidence * 100)}% confidence)`}
        </p>
      )}
      {result.processingMs !== undefined && (
        <p className="helper">Processing time: {formatProcessingTime(result.processingMs)}</p>
      )}
    </div>
  );
}

export default function EnhancePanel({ state, onEnhance, realEngine }: EnhancePanelProps) {
  const isPreparing = state.status === 'preparing';
  const isReady = state.status === 'ready';

  const buttonLabel = realEngine
    ? isPreparing
      ? 'Enhancing…'
      : isReady
        ? 'Enhance again'
        : 'Enhance my recording'
    : isPreparing
      ? 'Loading demo sample…'
      : isReady
        ? 'Load demo sample again'
        : 'Enhance audio (demo)';

  return (
    <section className="card" aria-labelledby="enhance-heading">
      <p className="step-label">Step 3</p>
      {realEngine ? (
        <>
          <h2 id="enhance-heading">Enhance your recording</h2>
          <p className="helper">
            Your recording is sent to the Intelli-Speech research model, which tries to reduce
            background noise. It is a research prototype, so results may vary.
          </p>
        </>
      ) : (
        <>
          <h2 id="enhance-heading">Try the enhancement demo</h2>
          <p className="helper">
            The real speech enhancement engine is not connected yet. This step uses a short
            sample clip so you can see how the finished experience will work.
          </p>
        </>
      )}

      <div className="actions">
        <button
          type="button"
          className={`button ${isReady ? 'button--quiet' : 'button--primary'}`}
          onClick={onEnhance}
          disabled={isPreparing}
        >
          {buttonLabel}
        </button>
      </div>

      {isPreparing && (
        <p className="processing">
          <span className="spinner" aria-hidden="true" />
          <span>
            {realEngine
              ? 'Enhancing your recording. This can take a few seconds.'
              : 'Loading the demo sample. Your recording is not being processed in demo mode.'}
          </span>
        </p>
      )}

      {state.status === 'error' && (
        <div className="notice notice--error" role="alert">
          <p>{state.message}</p>
        </div>
      )}

      {state.status === 'ready' &&
        (state.result.isDemo ? (
          <DemoResult result={state.result} />
        ) : (
          <RealResult result={state.result} />
        ))}
    </section>
  );
}
