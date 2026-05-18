import type {
  BeatSegment,
  HemodynamicResults,
  PmaxMethodResult,
  SessionStatus,
  WaveformData,
} from "./types";

const API = "/api";

function parseErrorMessage(res: Response, bodyText: string): string {
  try {
    const data = JSON.parse(bodyText) as { detail?: string | { msg: string }[] };
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg).join("; ");
    }
  } catch {
    /* use raw text */
  }
  return bodyText || res.statusText;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(parseErrorMessage(res, text));
  }
  return res.json() as Promise<T>;
}

export async function uploadImage(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return json<{
    session_id: string;
    width: number;
    height: number;
    preview_url: string;
  }>(await fetch(`${API}/session/upload`, { method: "POST", body: fd }));
}

export async function getSessionStatus(sessionId: string) {
  return json<SessionStatus>(await fetch(`${API}/session/${sessionId}/status`));
}

export async function processImage(
  sessionId: string,
  opts?: { threshold?: number; apply_notch_filter?: boolean }
) {
  return json<{ mask_preview_url: string; median_rows_url: string | null }>(
    await fetch(`${API}/session/${sessionId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts ?? {}),
    })
  );
}

export async function getMasks(sessionId: string) {
  return json<{
    width: number;
    height: number;
    base_mask_b64: string;
    manual_erase_b64: string;
    trace_keep_b64: string;
    image_url: string;
    mask_preview_url: string;
    median_rows: { x: number[]; y: (number | null)[] };
  }>(await fetch(`${API}/session/${sessionId}/masks`));
}

export async function updateMasks(
  sessionId: string,
  body: {
    manual_erase_keep_mask_b64?: string;
    trace_keep_mask_b64?: string;
    threshold?: number;
  }
) {
  return json<{ mask_preview_url: string }>(
    await fetch(`${API}/session/${sessionId}/masks/update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function finalizeMasks(sessionId: string) {
  return updateMasks(sessionId, {});
}

export async function calibrate(
  sessionId: string,
  body: {
    horizontal_line: [number, number, number, number];
    vertical_line: [number, number, number, number];
    origin: [number, number];
    time_span_seconds: number;
    pressure_span_mmhg: number;
  }
) {
  return json<{ x_ratio: number; y_ratio: number; calibrated_trace_url: string }>(
    await fetch(`${API}/session/${sessionId}/calibrate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function getCalibratedTrace(sessionId: string) {
  return json<WaveformData>(await fetch(`${API}/session/${sessionId}/calibrated_trace.json`));
}

export async function replaceLine(
  sessionId: string,
  point1: [number, number],
  point2: [number, number]
) {
  return json<{ status: string }>(
    await fetch(`${API}/session/${sessionId}/replace-line`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point1, point2 }),
    })
  );
}

export async function detectBeats(sessionId: string) {
  return json<{ beats: BeatSegment[] }>(
    await fetch(`${API}/session/${sessionId}/detect-beats`, { method: "POST" })
  );
}

export async function averageBeats(sessionId: string, beats: BeatSegment[]) {
  return json<{ averaged_waveform_url: string; mean_correlation: number }>(
    await fetch(`${API}/session/${sessionId}/average-beats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ beats }),
    })
  );
}

export async function getAveragedWaveform(sessionId: string) {
  return json<WaveformData>(await fetch(`${API}/session/${sessionId}/averaged_waveform.json`));
}

export async function runSingleBeatAnalysis(
  sessionId: string,
  body: {
    stroke_volume_ml: number;
    pmax_scale_factor: number;
    selected_pmax_method: string;
    peak_selection?: {
      edp_peak_indices: number[];
      esp_peak_indices: number[];
      event_marker_peak_indices: number[];
      smoothing_sigma_ms: number;
    };
    filter_cutoff_hz?: number;
    gaussian_sigma_ms?: number;
  }
) {
  return json<{
    hemodynamics: HemodynamicResults;
    pmax_methods: PmaxMethodResult[];
    selected_method: string;
  }>(
    await fetch(`${API}/session/${sessionId}/single-beat-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  );
}

export async function getAnalysis(sessionId: string) {
  return json<{
    hemodynamics: HemodynamicResults;
    pmax_methods: PmaxMethodResult[];
    selected_method: string;
  }>(await fetch(`${API}/session/${sessionId}/analysis`));
}

export function exportZipUrl(sessionId: string) {
  return `${API}/session/${sessionId}/export/zip`;
}

export async function createMockSession() {
  return json<{ session_id: string; is_mock?: boolean }>(
    await fetch(`${API}/mock-session`, { method: "POST" })
  );
}
