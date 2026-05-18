# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

This is a **distribution-only repository** for **SingleBeatAnalysisApp**, a MATLAB-compiled standalone desktop GUI application for cardiac/beat analysis (Johns Hopkins University). The repository contains **no source code** — only pre-built application archives (`.zip` and `.7z`) for Mac and Windows.

### Key constraints

- **No buildable source code**: MATLAB `.m`/`.mlapp` source files are not in this repo. The actual development happens outside this repository.
- **No package manager, tests, linting, or CI**: There is no `package.json`, `Makefile`, `requirements.txt`, test suite, or lint configuration.
- **Cannot run the application in Cloud VMs**: The compiled binaries target Mac (`.app` bundles) and Windows (`.exe`). There is no Linux build. Additionally, MATLAB Runtime R2025a (~2GB+ proprietary) is required.
- **No development workflow**: Changes to this repo are limited to adding/updating distribution archives and editing `README.md`.

### What agents can do

- Verify archive integrity: `unzip -t <file>.zip` for zip files.
- Inspect archive contents: `unzip -l <file>.zip` to list files.
- Read/edit `README.md` for documentation updates.
- Add or replace distribution archive files.

### What agents cannot do

- Build, compile, or run the MATLAB application (no source code, no MATLAB Runtime).
- Run tests or linting (none exist).
- Start any development server or service.
