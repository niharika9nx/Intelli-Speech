import { Link } from 'react-router-dom';

export default function NotFoundPage() {
  return (
    <div className="page page--narrow">
      <section className="card empty-state" aria-labelledby="page-heading">
        <p className="eyebrow">Page not found</p>
        <h1 id="page-heading" className="page-heading" tabIndex={-1}>
          We couldn&rsquo;t find that page
        </h1>
        <p className="page-lede">
          The address may be mistyped, or the page may have moved.
        </p>
        <div className="actions">
          <Link className="button button--primary" to="/">
            Back to welcome
          </Link>
          <Link className="button button--quiet" to="/home">
            Go to Home
          </Link>
        </div>
      </section>
    </div>
  );
}
