import { useEffect, useRef } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import Footer from './Footer';
import { BrandMark } from './icons';

const PAGE_TITLES: Record<string, string> = {
  '/': 'Welcome — Intelli-Speech',
  '/home': 'Home — Intelli-Speech',
  '/profile': 'Profile — Intelli-Speech',
};

function navClassName({ isActive }: { isActive: boolean }): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link';
}

export default function Layout() {
  const location = useLocation();
  const isFirstRender = useRef(true);

  useEffect(() => {
    document.title = PAGE_TITLES[location.pathname] ?? 'Page not found — Intelli-Speech';

    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    const heading = document.getElementById('page-heading');
    heading?.focus({ preventScroll: true });
  }, [location.pathname]);

  return (
    <div className="shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="site-header">
        <div className="header-inner">
          <Link className="brand" to="/" aria-label="Intelli-Speech welcome page">
            <span className="brand-mark">
              <BrandMark />
            </span>
            <span className="brand-name">Intelli-Speech</span>
          </Link>

          <nav aria-label="Main">
            <ul className="nav-list">
              <li>
                <NavLink to="/" end className={navClassName}>
                  Welcome
                </NavLink>
              </li>
              <li>
                <NavLink to="/home" className={navClassName}>
                  Home
                </NavLink>
              </li>
              <li>
                <NavLink to="/profile" className={navClassName}>
                  Profile
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main id="main-content" className="site-main" tabIndex={-1}>
        <Outlet />
      </main>

      <Footer />
    </div>
  );
}
