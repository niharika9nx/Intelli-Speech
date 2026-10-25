import { Link } from 'react-router-dom';
import { MicrophoneIcon } from '../components/icons';

export default function OpeningPage() {
  return (
    <>
      <section className="hero" aria-labelledby="page-heading">
        <div className="hero-copy">
          <p className="eyebrow">Speech enhancement research</p>
          <h1 id="page-heading" className="page-heading" tabIndex={-1}>
            A clearer way to hear speech
          </h1>
          <p className="hero-lede">
            Intelli-Speech is a research project exploring speech enhancement for noisy
            speech, with the goal of supporting people with speech impairments and older
            adults.
          </p>
          <div className="hero-actions">
            <Link className="button button--primary button--large" to="/home">
              Get started
            </Link>
            <Link className="button button--quiet" to="/profile">
              Adjust preferences
            </Link>
          </div>
          <p className="hero-note">
            <strong>Demo status:</strong> recording and playback are fully working. The
            enhancement step uses a clearly labelled sample clip until the real engine is
            connected.
          </p>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="hero-ring hero-ring--outer" />
          <div className="hero-ring hero-ring--inner" />
          <div className="hero-orb">
            <MicrophoneIcon />
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="how-heading">
        <h2 id="how-heading">How it works</h2>
        <p className="section-lede">
          Three simple steps, with no account or setup required.
        </p>

        <ol className="step-grid">
          <li className="step-card">
            <span className="step-number" aria-hidden="true">
              1
            </span>
            <h3>Record your voice</h3>
            <p>
              Choose the large microphone button and allow access when your browser asks.
              Your audio stays on your device.
            </p>
          </li>
          <li className="step-card">
            <span className="step-number" aria-hidden="true">
              2
            </span>
            <h3>Listen back</h3>
            <p>
              Play your recording, download it if you want, or record again for a clearer
              take.
            </p>
          </li>
          <li className="step-card">
            <span className="step-number" aria-hidden="true">
              3
            </span>
            <h3>Try the demo</h3>
            <p>
              Preview the enhancement experience with a sample clip. The interface says
              plainly when audio is a demo.
            </p>
          </li>
        </ol>
      </section>

      <section className="card landing-note" aria-labelledby="status-heading">
        <p className="step-label">Project status</p>
        <h2 id="status-heading">About this demonstration</h2>
        <p>
          Recording and playback are real browser features. The speech enhancement engine
          is still being researched and is not connected to this app yet, so the demo
          never claims that your recording has been enhanced.
        </p>
      </section>
    </>
  );
}
