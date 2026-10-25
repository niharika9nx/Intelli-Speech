import { useCallback, useEffect, useRef, useState } from 'react';
import EnhancePanel, { type DemoState } from './components/EnhancePanel';
import PlaybackPanel from './components/PlaybackPanel';
import RecordPanel from './components/RecordPanel';
import StatusAnnouncer from './components/StatusAnnouncer';
import { enhanceAudio } from './enhance';
import { MAX_RECORDING_SECONDS, useRecorder } from './hooks/useRecorder';
import './styles.css';

export default function App() {
  const {
    status,
    recording,
    elapsedSeconds,
    error,
    supportIssue,
    lastStopReason,
    startRecording,
    stopRecording,
    dismissError,
  } = useRecorder();

  const [announcement, setAnnouncement] = useState('');
  const [demo, setDemo] = useState<DemoState>({ status: 'idle' });
  const previousStatusRef = useRef(status);

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    previousStatusRef.current = status;

    if (previousStatus === status) {
      return;
    }

    if (status === 'requesting') {
      setAnnouncement('Waiting for microphone access. Your browser may ask for permission.');
    } else if (status === 'recording') {
      setAnnouncement('Recording started. Speak naturally, then choose Stop recording.');
    } else if (status === 'recorded') {
      setAnnouncement(
        lastStopReason === 'limit'
          ? `Recording finished at the ${MAX_RECORDING_SECONDS / 60} minute limit. You can listen to it now.`
          : 'Recording finished. You can listen to your recording now.',
      );
    }
  }, [lastStopReason, status]);

  useEffect(() => {
    if (status === 'requesting') {
      setDemo({ status: 'idle' });
    }
  }, [status]);

  const handleStart = useCallback(() => {
    void startRecording();
  }, [startRecording]);

  const handleEnhance = useCallback(async () => {
    if (!recording) {
      return;
    }

    setDemo({ status: 'preparing' });
    setAnnouncement(
      'Loading the demo sample. In demo mode, your recording is not processed.',
    );

    try {
      const result = await enhanceAudio({
        audioBlob: recording.blob,
        mimeType: recording.mimeType,
      });
      setDemo({ status: 'ready', result });
      setAnnouncement(
        'Demo audio is ready. It is a sample clip, not your recording, and no enhancement has been applied.',
      );
    } catch {
      setDemo({
        status: 'error',
        message: 'The demo sample could not be loaded. Please try again.',
      });
    }
  }, [recording]);

  const isBusy = status === 'requesting' || status === 'recording';
  const visibleRecording = isBusy ? null : recording;

  return (
    <div className="page">
      <header className="site-header">
        <p className="eyebrow">Research demonstration</p>
        <h1>Intelli-Speech</h1>
        <p className="lede">
          Record a short sample, listen back, and try the speech enhancement
          demonstration.
        </p>
      </header>

      <main className="main">
        <RecordPanel
          status={status}
          elapsedSeconds={elapsedSeconds}
          error={error}
          supportIssue={supportIssue}
          onStart={handleStart}
          onStop={stopRecording}
          onDismissError={dismissError}
        />

        {visibleRecording && (
          <PlaybackPanel recording={visibleRecording} onRecordAgain={handleStart} />
        )}

        {visibleRecording && <EnhancePanel state={demo} onEnhance={handleEnhance} />}
      </main>

      <footer className="site-footer">
        <p>
          Intelli-Speech is a research project. This interface is a demonstration — the
          speech enhancement engine is not connected yet.
        </p>
        <p>Your recording stays in your browser and is never uploaded.</p>
      </footer>

      <StatusAnnouncer message={announcement} />
    </div>
  );
}
