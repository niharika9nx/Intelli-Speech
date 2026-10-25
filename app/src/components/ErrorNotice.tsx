import type { ReactNode } from 'react';

interface ErrorNoticeProps {
  message: ReactNode;
  onDismiss?: () => void;
}

export default function ErrorNotice({ message, onDismiss }: ErrorNoticeProps) {
  return (
    <div className="notice notice--error" role="alert">
      <p>{message}</p>
      {onDismiss && (
        <button
          type="button"
          className="button button--quiet button--small"
          onClick={onDismiss}
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
