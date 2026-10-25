import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { ProfileProvider } from './context/ProfileContext';
import { RecordingProvider } from './context/RecordingContext';
import './styles.css';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element was not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <ProfileProvider>
        <RecordingProvider>
          <App />
        </RecordingProvider>
      </ProfileProvider>
    </BrowserRouter>
  </StrictMode>,
);
