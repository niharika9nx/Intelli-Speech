import type { EnhancementResult } from '../enhance';

export type DemoState =
  | { status: 'idle' }
  | { status: 'preparing' }
  | { status: 'ready'; result: EnhancementResult }
  | { status: 'error'; message: string };

interface EnhancePanelProps {
  state: DemoState;
  onEnhance: () => void;
}

export default function EnhancePanel({ state, onEnhance }: EnhancePanelProps) {
  const isPreparing = state.status === 'preparing';
  const isReady = state.status === 'ready';

  return (
    <section className="card" aria-labelledby="enhance-heading">
      <p className="step-label">Step 3</p>
      <h2 id="enhance-heading">Try the enhancement demo</h2>
      <p className="helper">
        The real speech enhancement engine is not connected yet. This step uses a short
        sample clip so you can see how the finished experience will work.
      </p>

      <div className="actions">
        <button
          type="button"
          className={`button ${isReady ? 'button--quiet' : 'button--primary'}`}
          onClick={onEnhance}
          disabled={isPreparing}
        >
          {isPreparing
            ? 'Loading demo sample…'
            : isReady
              ? 'Load demo sample again'
              : 'Enhance audio (demo)'}
        </button>
      </div>

      {isPreparing && (
        <p className="processing">
          <span className="spinner" aria-hidden="true" />
          <span>
            Loading the demo sample. Your recording is not being processed in demo mode.
          </span>
        </p>
      )}

      {state.status === 'error' && (
        <div className="notice notice--error" role="alert">
          <p>{state.message}</p>
        </div>
      )}

      {state.status === 'ready' && (
        <div className="demo-result">
          <div className="notice notice--demo" role="note">
            <p>
              <strong>Demo mode.</strong> {state.result.disclosure}
            </p>
          </div>

          <h3>Demo audio sample</h3>
          <audio
            className="audio-player"
            controls
            preload="metadata"
            src={state.result.audioUrl}
            aria-label="Demo audio sample playback"
          >
            Your browser does not support audio playback.
          </audio>

          <p className="helper">
            When the real engine is connected, this area will play audio enhanced from your
            own recording.
          </p>
        </div>
      )}
    </section>
  );
}
