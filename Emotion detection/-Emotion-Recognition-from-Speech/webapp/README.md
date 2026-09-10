# SER Studio — web UI

A React + Framer Motion front end over the trained speech-emotion model.
It does two things: run live predictions on audio, and show the model's
measured performance honestly (including where it fails).

![stack](https://img.shields.io/badge/React-19-informational) ![stack](https://img.shields.io/badge/Framer%20Motion-13-informational) ![stack](https://img.shields.io/badge/Tailwind-4-informational) ![stack](https://img.shields.io/badge/Flask-3-informational)

## Quick start

Two terminals during development.

**1 — API** (loads `models/final_model/`, listens on 5001):

```bash
.venv\Scripts\python.exe webapp\backend\app.py
```

**2 — UI** (Vite dev server on 5173, proxies `/api` to 5001):

```bash
cd webapp\frontend && npm install && npm run dev
```

### Single-port production build

```bash
cd webapp\frontend && npm run build
```

The bundle is emitted straight into `webapp/backend/static/`, so after
building you only need the Flask process — open <http://127.0.0.1:5001>.

## What's on the page

| Section | What it shows |
|---|---|
| **Hero** | Headline metrics and an animated waveform |
| **Analyse** | Drop a file or pick a held-out clip → live prediction, confidence ring with the calibrated threshold marked, and the full 8-class probability distribution |
| **Performance** | Test metrics, every model compared, the one-standard-error selection argument, per-emotion F1, confusion matrix, top confusions, per-speaker accuracy |
| **Method** | The six pipeline stages, including why the split is actor-disjoint |

Clips labelled **degraded channel** are held-out test clips pushed through a
telephone band or a reverberant room. They demonstrate the honest failure
mode: accuracy drops, and confidence drops with it.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/overview` | All dashboard data in one response |
| `GET` | `/api/samples` | Held-out and degraded demo clips |
| `GET` | `/api/audio/<id>` | Stream a demo clip for playback |
| `POST` | `/api/predict` | Multipart `file` → prediction |
| `POST` | `/api/predict/sample/<id>` | Predict a demo clip (includes true label) |
| `GET` | `/api/health` | Liveness + whether the model is loaded |

Uploads are limited to 20 MB and `.wav/.mp3/.flac/.ogg/.m4a`; anything else is
rejected with a JSON error rather than a stack trace.

Every number the UI displays is read from the pipeline's own output files in
`outputs/metrics/` — the front end computes no metrics of its own.

## Design decisions

**Motion is informational, not decorative.** Probability bars spring to their
value, the confidence ring sweeps to its arc, metrics count up. Each animation
communicates a magnitude. `useReducedMotion` is honoured throughout, and the
CSS carries a `prefers-reduced-motion` block, so the whole thing renders
instantly in its final state for users who ask for that.

**Colour is never the only channel.** Every probability bar, matrix cell and
legend entry also carries a text label and a number. The confusion matrix
prints counts in the cells and exposes each as
`"True happy, predicted surprised: 7 clips"` to screen readers.

**Both themes are defined explicitly.** Dark is the default; the toggle
persists to `localStorage` and is wrapped in try/catch so private-mode
browsing doesn't break it.

**Accessibility.** Visible focus rings (restyled, never removed), a skip link,
44px-plus touch targets, ARIA `meter` roles on the probability bars, semantic
table markup for the matrix, and no emoji used as icons — all glyphs are SVG
from `lucide-react`.
