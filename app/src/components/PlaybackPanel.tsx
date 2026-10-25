import type { Recording } from '../hooks/useRecorder';

interface PlaybackPanelProps {
  recording: Recording;
  onRecordAgain: () => void;
}

function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function fileExtension(mimeType: string): string {
  const base = mimeType.split(';')[0].trim().toLowerCase();
  if (base.includes('mp4')) {
    return 'm4a';
  }
  if (base.includes('ogg')) {
    return 'ogg';
  }
  if (base.includes('wav')) {
    return 'wav';
  }
  return 'webm';
}

export default function PlaybackPanel({ recording, onRecordAgain }: PlaybackPanelProps) {
  return (
    <section className="card" aria-labelledby="playback-heading">
      <p className="step-label">Step 2</p>
      <h2 id="playback-heading">Listen to your recording</h2>
      <p className="helper">
        Length: {formatDuration(recording.durationSeconds)}. Your recording stays in this
        browser and is not uploaded.
      </p>

      <audio
        className="audio-player"
        controls
        preload="metadata"
        src={recording.url}
        aria-label="Playback of your recording"
      >
        Your browser does not support audio playback. You can{' '}
        <a href={recording.url} download={`intelli-speech-recording.${fileExtension(recording.mimeType)}`}>
          download the recording
        </a>
        .
      </audio>

      <div className="actions">
        <button type="button" className="button button--secondary" onClick={onRecordAgain}>
          Record again
        </button>
        <a
          className="button button--quiet"
          href={recording.url}
          download={`intelli-speech-recording.${fileExtension(recording.mimeType)}`}
        >
          Download recording
        </a>
      </div>
    </section>
  );
}
