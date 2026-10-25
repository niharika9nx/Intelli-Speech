import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

export interface ProfileSettings {
  displayName: string;
  largeText: boolean;
  reducedMotion: boolean;
}

interface ProfileContextValue {
  settings: ProfileSettings;
  updateSettings: (changes: Partial<ProfileSettings>) => void;
  resetProfile: () => void;
  storageAvailable: boolean;
}

const STORAGE_KEY = 'intelli-speech-profile-v1';

const DEFAULT_SETTINGS: ProfileSettings = {
  displayName: '',
  largeText: false,
  reducedMotion: false,
};

function loadSettings(): ProfileSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_SETTINGS;
    }

    const parsed = JSON.parse(raw) as Partial<ProfileSettings>;

    return {
      displayName:
        typeof parsed.displayName === 'string' ? parsed.displayName.slice(0, 50) : '',
      largeText: parsed.largeText === true,
      reducedMotion: parsed.reducedMotion === true,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function detectStorage(): boolean {
  try {
    const testKey = `${STORAGE_KEY}-test`;
    window.localStorage.setItem(testKey, '1');
    window.localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

const ProfileContext = createContext<ProfileContextValue | null>(null);

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<ProfileSettings>(loadSettings);
  const [storageAvailable] = useState(detectStorage);

  useEffect(() => {
    if (!storageAvailable) {
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Preferences still work for this visit when storage is unavailable.
    }
  }, [settings, storageAvailable]);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.largeText = settings.largeText ? 'true' : 'false';
    root.dataset.reducedMotion = settings.reducedMotion ? 'true' : 'false';
  }, [settings.largeText, settings.reducedMotion]);

  const updateSettings = useCallback((changes: Partial<ProfileSettings>) => {
    setSettings((current) => ({ ...current, ...changes }));
  }, []);

  const resetProfile = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
  }, []);

  const value = useMemo(
    () => ({
      settings,
      updateSettings,
      resetProfile,
      storageAvailable,
    }),
    [settings, updateSettings, resetProfile, storageAvailable],
  );

  return <ProfileContext.Provider value={value}>{children}</ProfileContext.Provider>;
}

export function useProfile(): ProfileContextValue {
  const context = useContext(ProfileContext);

  if (!context) {
    throw new Error('useProfile must be used within a ProfileProvider.');
  }

  return context;
}
