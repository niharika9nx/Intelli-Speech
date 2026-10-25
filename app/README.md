# Intelli-Speech Frontend

A small browser demonstration of the intended Intelli-Speech experience: record
speech, listen back, and preview the enhancement flow.

## Important

This is a **frontend demonstration**, not the speech enhancement engine.

- Recording and playback are real and run entirely in the browser using the
  microphone and `MediaRecorder` APIs. The recording is never uploaded.
- The **Enhance audio (demo)** step loads a bundled sample clip. It does **not**
  process the user's recording and does not run any model.
- The sample is clearly labelled as a demo in the interface. The real
  enhancement engine is not connected yet.
- The bundled demo clip is a locally generated synthetic voice sample, included
  only to demonstrate playback.
- The frontend does not assume or define an API for the future engine. That
  boundary lives in `src/enhance/index.ts`.
- The profile page stores display and accessibility preferences in browser
  `localStorage` only. No account is created and nothing is sent to a server.

## Pages and routes

| Route | Page | Purpose |
|---|---|---|
| `/` | Opening page | Explains the project and links into the app |
| `/home` | Home | Record, listen back, and try the enhancement demo |
| `/profile` | Profile | Display name and accessibility preferences |
| `*` | Not found | Friendly fallback for unknown addresses |

## Run locally

A current Node.js release and npm are required.

```bash
cd app
npm install
npm run dev
```

Open the local address printed by Vite. Microphone access requires `localhost`
or an HTTPS connection.

## Other commands

```bash
npm run build    # Type-check and create a production build in dist/
npm run preview  # Serve the production build locally
```

## Structure

```text
app/
├── src/
│   ├── App.tsx                     # Route definitions
│   ├── main.tsx                    # React entry point, router and providers
│   ├── styles.css                  # Global styles and responsive layout
│   ├── assets/
│   │   └── demo-sample.wav         # Clearly labelled demo clip
│   ├── components/
│   │   ├── Layout.tsx              # Header, navigation, focus handling, footer
│   │   ├── Footer.tsx              # Research and privacy notes
│   │   ├── icons.tsx               # Shared inline icons
│   │   ├── RecordPanel.tsx         # Microphone control and recording status
│   │   ├── PlaybackPanel.tsx       # Playback and download of the recording
│   │   ├── EnhancePanel.tsx        # Demo enhancement action and output
│   │   ├── StatusAnnouncer.tsx     # Screen-reader live region
│   │   └── ErrorNotice.tsx         # Shared error message component
│   ├── context/
│   │   ├── ProfileContext.tsx      # Local preferences and accessibility settings
│   │   └── RecordingContext.tsx    # Recording state shared across pages
│   ├── enhance/
│   │   └── index.ts                # Future engine boundary (demo now)
│   ├── hooks/
│   │   └── useRecorder.ts          # getUserMedia/MediaRecorder workflow
│   └── pages/
│       ├── OpeningPage.tsx         # Welcome/landing page
│       ├── HomePage.tsx            # Main recording workflow
│       ├── ProfilePage.tsx         # Local profile settings
│       └── NotFoundPage.tsx        # Unknown-route fallback
├── index.html
├── package.json
├── tsconfig*.json
└── vite.config.ts
```

## Accessibility notes

- All actions use native buttons and links with visible focus styles.
- Route changes update the page title and move focus to the new page heading.
- Recording status changes are announced through a polite live region.
- Errors use `role="alert"`.
- The profile page can enlarge text and reduce motion, and the layout also
  respects the device's own `prefers-reduced-motion` setting.
- The layout is responsive and includes a skip-to-content link.
