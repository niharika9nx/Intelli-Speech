import { useCallback, useEffect, useRef, useState } from 'react';

/** Recordings are capped so a single demo recording cannot grow without bound. */
export const MAX_RECORDING_SECONDS = 120;

const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
] as const;

export type RecorderStatus =
  | 'unsupported'
  | 'idle'
  | 'requesting'
  | 'recording'
  | 'recorded'
  | 'error';

export type StopReason = 'manual' | 'limit';

export interface Recording {
  blob: Blob;
  url: string;
  mimeType: string;
  durationSeconds: number;
}

export interface UseRecorderResult {
  status: RecorderStatus;
  recording: Recording | null;
  elapsedSeconds: number;
  error: string | null;
  supportIssue: string | null;
  lastStopReason: StopReason | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  dismissError: () => void;
}

function getSupportIssue(): string | null {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return 'Recording is not available in this environment.';
  }
  if (!window.isSecureContext) {
    return 'Microphone recording needs a secure connection. Open this page over HTTPS or on localhost.';
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return 'This browser cannot access the microphone. Try a recent version of Chrome, Edge, Firefox, or Safari.';
  }
  if (typeof MediaRecorder === 'undefined') {
    return 'This browser cannot save audio recordings. Try a recent version of Chrome, Edge, Firefox, or Safari.';
  }
  return null;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder.isTypeSupported !== 'function') {
    return undefined;
  }
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

function describeMicrophoneError(error: unknown): string {
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Microphone access was blocked. Allow microphone access in your browser settings, then try again.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No microphone was found. Connect a microphone and try again.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'Your microphone is busy or unavailable. Close other apps that use it and try again.';
      case 'SecurityError':
        return 'Recording needs a secure connection. Open this page over HTTPS or on localhost.';
      case 'OverconstrainedError':
        return 'The microphone does not support the requested settings. Try a different microphone.';
      default:
        return 'The microphone could not be started. Please try again.';
    }
  }
  return 'The microphone could not be started. Please try again.';
}

export function useRecorder(): UseRecorderResult {
  const [supportIssue] = useState<string | null>(getSupportIssue);
  const [status, setStatus] = useState<RecorderStatus>(() =>
    getSupportIssue() ? 'unsupported' : 'idle',
  );
  const [recording, setRecording] = useState<Recording | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastStopReason, setLastStopReason] = useState<StopReason | null>(null);

  const statusRef = useRef<RecorderStatus>(status);
  const mountedRef = useRef(true);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const limitTimerRef = useRef<number | null>(null);
  const startedAtRef = useRef<number | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const stopReasonRef = useRef<StopReason>('manual');
  const failedRef = useRef(false);
  const finalizingRef = useRef(false);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const clearTimers = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (limitTimerRef.current !== null) {
      window.clearTimeout(limitTimerRef.current);
      limitTimerRef.current = null;
    }
  }, []);

  const releaseStream = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    stream?.getTracks().forEach((track) => track.stop());
  }, []);

  const finalize = useCallback(() => {
    if (finalizingRef.current) {
      return;
    }
    finalizingRef.current = true;

    clearTimers();
    releaseStream();

    const recorder = mediaRecorderRef.current;
    mediaRecorderRef.current = null;

    const duration =
      startedAtRef.current === null
        ? 0
        : Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000));
    startedAtRef.current = null;

    const chunks = chunksRef.current;
    chunksRef.current = [];
    const stopReason = stopReasonRef.current;

    if (!mountedRef.current) {
      return;
    }

    if (failedRef.current) {
      failedRef.current = false;
      setStatus('error');
      setError('Recording failed unexpectedly. Please try again.');
      return;
    }

    const blob = new Blob(chunks, {
      type: recorder?.mimeType || chunks[0]?.type || 'audio/webm',
    });

    if (blob.size === 0) {
      setStatus('error');
      setError('No audio was captured. Check that your microphone is working and try again.');
      return;
    }

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    const url = URL.createObjectURL(blob);
    objectUrlRef.current = url;

    setRecording({
      blob,
      url,
      mimeType: blob.type || 'audio/webm',
      durationSeconds: duration,
    });
    setElapsedSeconds(duration);
    setLastStopReason(stopReason);
    setStatus('recorded');
  }, [clearTimers, releaseStream]);

  const stopRecordingInternal = useCallback(
    (reason: StopReason) => {
      stopReasonRef.current = reason;
      clearTimers();

      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === 'inactive') {
        finalize();
        return;
      }

      try {
        recorder.stop();
      } catch {
        finalize();
      }
    },
    [clearTimers, finalize],
  );

  const startRecording = useCallback(async () => {
    if (
      supportIssue ||
      statusRef.current === 'requesting' ||
      statusRef.current === 'recording' ||
      mediaRecorderRef.current !== null
    ) {
      return;
    }

    setError(null);
    setLastStopReason(null);
    setStatus('requesting');
    finalizingRef.current = false;
    failedRef.current = false;
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      const mimeType = pickMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        failedRef.current = true;
        clearTimers();
        if (recorder.state !== 'inactive') {
          try {
            recorder.stop();
          } catch {
            finalize();
          }
        } else {
          finalize();
        }
      };

      recorder.onstop = finalize;

      stopReasonRef.current = 'manual';
      startedAtRef.current = Date.now();
      setElapsedSeconds(0);

      recorder.start();
      setStatus('recording');

      timerRef.current = window.setInterval(() => {
        const startedAt = startedAtRef.current;
        if (startedAt !== null) {
          setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
        }
      }, 250);

      limitTimerRef.current = window.setTimeout(() => {
        stopRecordingInternal('limit');
      }, MAX_RECORDING_SECONDS * 1000);
    } catch (caught) {
      releaseStream();
      mediaRecorderRef.current = null;
      if (!mountedRef.current) {
        return;
      }
      setError(describeMicrophoneError(caught));
      setStatus('error');
    }
  }, [clearTimers, finalize, releaseStream, stopRecordingInternal, supportIssue]);

  const stopRecording = useCallback(() => {
    stopRecordingInternal('manual');
  }, [stopRecordingInternal]);

  const dismissError = useCallback(() => {
    setError(null);
    if (mediaRecorderRef.current === null) {
      setStatus(recording ? 'recorded' : 'idle');
    }
  }, [recording]);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      clearTimers();

      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        recorder.onerror = null;
        try {
          recorder.stop();
        } catch {
          // Ignore: the page is being torn down.
        }
      }
      mediaRecorderRef.current = null;

      const stream = streamRef.current;
      streamRef.current = null;
      stream?.getTracks().forEach((track) => track.stop());

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [clearTimers]);

  return {
    status,
    recording,
    elapsedSeconds,
    error,
    supportIssue,
    lastStopReason,
    startRecording,
    stopRecording,
    dismissError,
  };
}
