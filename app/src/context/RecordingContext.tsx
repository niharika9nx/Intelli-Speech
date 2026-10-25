import { createContext, useContext, type ReactNode } from 'react';
import { useRecorder, type UseRecorderResult } from '../hooks/useRecorder';

const RecordingContext = createContext<UseRecorderResult | null>(null);

/**
 * Keeps an in-progress or completed recording alive while the user navigates
 * between pages. Recording data still never leaves the browser.
 */
export function RecordingProvider({ children }: { children: ReactNode }) {
  const recorder = useRecorder();

  return (
    <RecordingContext.Provider value={recorder}>
      {children}
    </RecordingContext.Provider>
  );
}

export function useRecording(): UseRecorderResult {
  const context = useContext(RecordingContext);

  if (!context) {
    throw new Error('useRecording must be used within a RecordingProvider.');
  }

  return context;
}
