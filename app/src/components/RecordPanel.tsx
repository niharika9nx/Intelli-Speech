import ErrorNotice from './ErrorNotice';
import { MAX_RECORDING_SECONDS, type RecorderStatus } from '../hooks/useRecorder';

interface RecordPanelProps {
  status: RecorderStatus;
  elapsedSeconds: number;
  error: string | null;
  supportIssue: string | null;
  onStart: () => void;
  onStop: () => void;
  onDismissError: () => void;
}

const STATUS_TEXT: Record<RecorderStatus, string> = {
  unsupported: 'Recording is not available in this browser.',
  idle: 'Ready when you are.',
  requesting: 'Waiting for microphone access…',
  recording: 'Recording… speak naturally.',
  recorded: 'Recording complete. Listen to it below.',
  error: 'The recording did not start or finish correctly.',
};

function formatTimer(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function MicrophoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
      <path
        d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M8.5 21h7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" />
    </svg>
  );
}

export default function RecordPanel({
  status,
  elapsedSeconds,
  error,
  supportIssue,
  onStart,
  onStop,
  onDismissError,
}: RecordPanelProps) {
  const isRecording = status === 'recording';
  const isRequesting = status === 'requesting';
  const isDisabled = Boolean(supportIssue) || isRequesting;

  let actionLabel = 'Start recording';
  if (isRecording) {
    actionLabel = 'Stop recording';
  } else if (isRequesting) {
    actionLabel = 'Starting…';
  } else if (status === 'recorded') {
    actionLabel = 'Record again';
  } else if (status === 'error') {
    actionLabel = 'Try recording again';
  }

  const showTimer = isRecording || status === 'recorded';

  return (
    <section className="card" aria-labelledby="record-heading">
      <p className="step-label">Step 1</p>
      <h2 id="record-heading">Record your voice</h2>
      <p className="helper" id="record-help">
        Choose the microphone button, then allow access when your browser asks.
        Keep each recording under {Math.floor(MAX_RECORDING_SECONDS / 60)} minutes.
      </p>

      <div className="record-area">
        <button
          type="button"
          className={`record-button${isRecording ? ' record-button--recording' : ''}`}
          onClick={isRecording ? onStop : onStart}
          disabled={isDisabled}
          aria-describedby="record-help"
        >
          <span className="record-button__icon">
            {isRecording ? <StopIcon /> : <MicrophoneIcon />}
          </span>
          <span className="record-button__label">{actionLabel}</span>
        </button>

        <p className="status-line">
          <span
            className={`status-dot status-dot--${status}`}
            aria-hidden="true"
          />
          <span>{STATUS_TEXT[status]}</span>
        </p>

        {showTimer && (
          <p className="timer" aria-hidden="true">
            {formatTimer(elapsedSeconds)}
          </p>
        )}
      </div>

      {supportIssue && (
        <div className="notice notice--warning" role="note">
          <p>{supportIssue}</p>
        </div>
      )}

      {error && <ErrorNotice message={error} onDismiss={onDismissError} />}
    </section>
  );
}
