interface StatusAnnouncerProps {
  message: string;
}

/**
 * Screen-reader-only live region. Keep it mounted even when empty so state
 * changes are announced reliably.
 */
export default function StatusAnnouncer({ message }: StatusAnnouncerProps) {
  return (
    <div
      className="visually-hidden"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {message}
    </div>
  );
}
