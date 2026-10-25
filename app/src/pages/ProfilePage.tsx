import { useState } from 'react';
import StatusAnnouncer from '../components/StatusAnnouncer';
import { useProfile } from '../context/ProfileContext';

export default function ProfilePage() {
  const { settings, updateSettings, resetProfile, storageAvailable } = useProfile();
  const [announcement, setAnnouncement] = useState('');

  const hasSavedSettings =
    settings.displayName.trim() !== '' || settings.largeText || settings.reducedMotion;

  const handleLargeText = (enabled: boolean) => {
    updateSettings({ largeText: enabled });
    setAnnouncement(`Larger text ${enabled ? 'turned on' : 'turned off'}.`);
  };

  const handleReducedMotion = (enabled: boolean) => {
    updateSettings({ reducedMotion: enabled });
    setAnnouncement(
      `Reduced motion ${enabled ? 'turned on' : 'turned off'}.`,
    );
  };

  const handleReset = () => {
    resetProfile();
    setAnnouncement('Saved profile settings have been cleared.');
  };

  return (
    <div className="page page--narrow">
      <header className="page-intro">
        <p className="eyebrow">Your preferences</p>
        <h1 id="page-heading" className="page-heading" tabIndex={-1}>
          Profile
        </h1>
        <p className="page-lede">
          Adjust the app to suit you. These settings are stored only in this browser.
        </p>
      </header>

      {!storageAvailable && (
        <div className="notice notice--warning" role="note">
          <p>
            This browser is blocking local storage. Preferences will work for this visit,
            but they will not be remembered next time.
          </p>
        </div>
      )}

      <section className="card" aria-labelledby="details-heading">
        <h2 id="details-heading">Your details</h2>

        <div className="field">
          <label htmlFor="display-name">
            Display name <span className="field-optional">(optional)</span>
          </label>
          <input
            id="display-name"
            className="text-input"
            type="text"
            value={settings.displayName}
            onChange={(event) =>
              updateSettings({ displayName: event.target.value.slice(0, 50) })
            }
            maxLength={50}
            autoComplete="name"
            placeholder="For example, Niharika"
            aria-describedby="display-name-help"
          />
          <p id="display-name-help" className="field-help">
            Used only to greet you on the Home page. It never leaves this browser.
          </p>
        </div>
      </section>

      <section className="card" aria-labelledby="accessibility-heading">
        <h2 id="accessibility-heading">Accessibility</h2>
        <p className="helper">
          Choose the options that make the app easier to use. Changes apply immediately.
        </p>

        <div className="settings-list">
          <div className="setting-row">
            <div className="setting-copy">
              <label htmlFor="large-text">Larger text</label>
              <p id="large-text-help" className="field-help">
                Increase the text size across the whole app.
              </p>
            </div>
            <input
              id="large-text"
              className="setting-checkbox"
              type="checkbox"
              checked={settings.largeText}
              onChange={(event) => handleLargeText(event.target.checked)}
              aria-describedby="large-text-help"
            />
          </div>

          <div className="setting-row">
            <div className="setting-copy">
              <label htmlFor="reduced-motion">Reduce motion</label>
              <p id="reduced-motion-help" className="field-help">
                Turn off pulsing and spinning animations. Your device setting is respected
                automatically as well.
              </p>
            </div>
            <input
              id="reduced-motion"
              className="setting-checkbox"
              type="checkbox"
              checked={settings.reducedMotion}
              onChange={(event) => handleReducedMotion(event.target.checked)}
              aria-describedby="reduced-motion-help"
            />
          </div>
        </div>
      </section>

      <section className="card" aria-labelledby="data-heading">
        <h2 id="data-heading">Saved in this browser</h2>
        <p className="helper">
          No account is created and nothing is sent to a server. Clearing your browser
          data will remove these preferences.
        </p>
        <div className="actions">
          <button
            type="button"
            className="button button--quiet"
            onClick={handleReset}
            disabled={!hasSavedSettings}
          >
            Clear saved profile
          </button>
        </div>
      </section>

      <StatusAnnouncer message={announcement} />
    </div>
  );
}
