# RV Single-Beat Pressure-Volume Analysis (Web)

Production-quality local-first web port of the MATLAB Single Beat Analysis workflow for right-ventricular pressure-volume analysis.

## Stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS, shadcn-style UI, Konva (mask editor), Plotly (scientific plots)
- **Backend:** FastAPI, NumPy, SciPy, OpenCV, scikit-image, pandas
- **Storage:** `data/sessions/{session_id}/` (images, masks, traces, exports)

## Workflow

1. Upload waveform screenshot
2. Process image → binary mask + median rows
3. Edit mask (erase / trace / threshold)
4. Calibrate axes (optional replace line)
5. Detect & manually edit beats → average
6. Single-beat analysis + Pmax method comparison
7. Export CSV/JSON/ZIP

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Recommended — only reloads when app/ changes (avoids .venv restart loops):
chmod +x scripts/run_dev.sh
./scripts/run_dev.sh

# Or manually:
PYTHONPATH=. python3 -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
```

**Important:** Do not use bare `--reload` without `--reload-dir app` — it watches `.venv` and causes crashes / `socket hang up` in the frontend.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — API requests proxy to port 8000 via `next.config.mjs`.

### Tests

```bash
cd backend
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

### Demo session (no image)

In the UI click **Load synthetic demo session**, or:

```bash
curl -X POST http://127.0.0.1:8000/api/mock-session
```

## Project layout

```
frontend/          Next.js UI
backend/app/       FastAPI + services
backend/tests/     Unit tests
data/sessions/     Per-session artifacts (gitignored contents)
```

## Pmax methods

All methods return the same JSON schema; failures are isolated:

| Method | Kind |
|--------|------|
| Original Piecewise Sinusoid | Default downstream |
| Brimioulle Sine | `P = a + b(1+sin(ct+d))`, isovolumic windows |
| RV IsoMax Sine | Brimioulle core, 3rd/2nd-derivative windows |
| Second Derivative Sine | Brimioulle core, 2nd-derivative windows |
| Kremer/Shih Tangent | Tangent intersection from dP/dt extrema |

## MATLAB port notes / TODO

Without the `.mlapp` source in-repo, several items are **approximate** and marked for calibration against MATLAB:

- [ ] Exact notch/Fourier filter parameters from App Designer
- [ ] Original piecewise sinusoid segment boundaries vs `SingleBeatAnalysisApp_0517_Methods_Compare.mlapp`
- [ ] Event-marker peak detection weights and EDP half-height logic
- [ ] PV loop construction from measured volumes (current PV plot is illustrative)
- [ ] Ees/Ea/EDV from catheter data vs SV-only estimates
- [ ] Cross-correlation beat alignment tuning
- [ ] PNG/SVG plot export from matplotlib on server

Place the MATLAB app at the path referenced in your lab and diff outputs against `exports/` CSVs.

## Scientific safety

- Failed Pmax methods do not abort the run.
- Downstream method is user-selected (default: **Original Piecewise Sinusoid**).
- Formulas include inline comments in `backend/app/services/`.

## Legacy MATLAB distribution

See historical install notes in the original README (Google Drive installers for Mac/Windows standalone apps).

## Contact

Per original project: Sun Moon gmoon3@jh.edu
