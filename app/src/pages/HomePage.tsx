import { useCallback, useEffect, useRef, useState } from 'react';
import EnhancePanel, { type DemoState } from '../components/EnhancePanel';
import PlaybackPanel from '../components/PlaybackPanel';
import RecordPanel from '../components/RecordPanel';
import StatusAnnouncer from '../components/StatusAnnouncer';
import { useProfile } from '../context/ProfileContext';
import { useRecording } from '../context/RecordingContext';
import { enhanceAudio } from '../enhance';
import { MAX_RECORDING_SECONDS } from '../hooks/useRecorder';

export default function HomePage() {
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
  } = useRecording();
  const { settings } = useProfile();

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
  const displayName = settings.displayName.trim();

  return (
    <div className="page page--narrow">
      <header className="page-intro">
        <p className="eyebrow">Record and listen</p>
        <h1 id="page-heading" className="page-heading" tabIndex={-1}>
          {displayName ? `Hello, ${displayName}` : 'Try Intelli-Speech'}
        </h1>
        <p className="page-lede">
          Record a short sample, listen back, and try the enhancement demonstration.
        </p>
      </header>

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

      <StatusAnnouncer message={announcement} />
    </div>
  );
}
