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
chmod +x scripts/setup_venv.sh scripts/run_tests.sh scripts/run_dev.sh
./scripts/setup_venv.sh    # creates .venv, installs deps, runs tests
./scripts/run_tests.sh -v  # later runs (always uses .venv/bin/python)
```

### macOS: `pytest` / `TimeoutError: Operation timed out`

This usually means the **virtualenv under `~/Documents` is broken or stalled** (iCloud Drive, antivirus, or a partial install). It is **not** an application logic error.

**Fix:**

```bash
cd backend
rm -rf .venv
./scripts/setup_venv.sh
./scripts/run_dev.sh
```

Use `./scripts/run_tests.sh` instead of `python3 -m pytest` so tests always run inside `.venv`.

If timeouts persist, clone or move the repo to a local non-synced folder (e.g. `~/Developer/RV_App_MATLAB`).

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

## MATLAB port notes

The backend now mirrors the `SingleBeatAnalysisApp_0517_Methods_Compare` logic for:

- **Original Piecewise Sinusoid** — `fitNonlinearFunction` (fsolve) between dP/dt max/min
- **Brimioulle / RV IsoMax / Second-derivative sine** — `fitBrimioulleSineLMFromWindows` with multi-start LM
- **Kremer/Shih tangent** — paper formula for Pmax from dP/dt tangents
- **Beat averaging** — `alignAndNormalizeBeatsForAverage` (foot alignment + xcorr refinement)
- **Peak selection** — robust (d²P/dt²)² event marker; ESP = 3rd peak; EDP = half-height before 1st peak
- **Hemodynamics** — `Ees = (Pmax − ESP) / SV`, anchored EDPVR, logistic tau from dP/dt min

Still approximate or UI-only vs MATLAB App Designer:

- [ ] 2D notch filter bank on process (optional 1D notch exists; MATLAB uses full FFT2 grid)
- [ ] `adaptthresh` binary mask (web app uses fixed threshold; default **95** matches MATLAB `CurrentThreshold`)
- [ ] Interactive peak-selection UI smoothing slider (server accepts client peak indices)
- [ ] PV loop spline display (`optimizeSpline`) — exports volumes; loop plot not yet in web UI
- [ ] Golden-file numeric regression against saved MATLAB CSVs per patient

Calibrate against MATLAB by diffing `exports/patient_data.csv` and `pmax_method_summary.csv` for the same trace.

## Scientific safety

- Failed Pmax methods do not abort the run.
- Downstream method is user-selected (default: **Original Piecewise Sinusoid**).
- Formulas include inline comments in `backend/app/services/`.

## Legacy MATLAB distribution

See historical install notes in the original README (Google Drive installers for Mac/Windows standalone apps).

## Contact

Per original project: Sun Moon gmoon3@jh.edu
