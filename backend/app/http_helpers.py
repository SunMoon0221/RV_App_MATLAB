"""Shared HTTP / filesystem helpers for the API."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.session_store import SessionState, SessionStore, store


def get_session_or_404(session_id: str) -> SessionState:
    try:
        return store.get(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found") from None


def safe_session_file(session_id: str, filename: str) -> Path:
    """Resolve a file under the session directory only (no path traversal)."""
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")
    base = store._path(session_id).resolve()
    candidates = [
        store.image_path(session_id, filename),
        store._path(session_id) / filename,
        store.export_path(session_id, filename),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(base) and resolved.exists():
            return resolved
    raise HTTPException(404, "File not found")


def clamp_index(idx: int, length: int, name: str = "index") -> int:
    if length < 1:
        raise HTTPException(400, "Waveform is empty")
    if idx < 0 or idx >= length:
        return int(max(0, min(idx, length - 1)))
    return int(idx)
