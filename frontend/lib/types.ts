export type StepId =
  | "upload"
  | "process"
  | "edit_mask"
  | "calibrate"
  | "average_beats"
  | "single_beat"
  | "export";

export type StepStatus =
  | "not_ready"
  | "ready"
  | "needs_calibration"
  | "processed"
  | "averaged"
  | "analysis_complete"
  | "warning";

export interface BeatSegment {
  start_idx: number;
  end_idx: number;
  peak_idx: number;
  start_time: number;
  end_time: number;
  peak_time: number;
  peak_pressure: number;
  keep: boolean;
  qc_passed?: boolean;
  qc_message?: string;
}

export interface PmaxMethodResult {
  name: string;
  short_name: string;
  kind: string;
  success: boolean;
  message: string;
  raw_pmax: number | null;
  scaled_pmax: number | null;
  scale_factor: number;
  fit_r2: number | null;
  pt1?: { t: number; p: number } | null;
  pt2?: { t: number; p: number } | null;
  pt3?: { t: number; p: number } | null;
  pt4?: { t: number; p: number } | null;
  t_fit?: number[] | null;
  p_fit_raw?: number[] | null;
  p_fit_scaled?: number[] | null;
  t_used1?: number[] | null;
  p_used1?: number[] | null;
  t_used2?: number[] | null;
  p_used2?: number[] | null;
  peak_time?: number | null;
  t_cross?: number | null;
  p_cross?: number | null;
  is_selected: boolean;
}

export interface HemodynamicResults {
  esp?: number | null;
  edp?: number | null;
  pmax?: number | null;
  ees?: number | null;
  ea?: number | null;
  sv?: number | null;
  dpdt_max?: number | null;
  dpdt_min?: number | null;
  beta?: number | null;
  tau?: number | null;
  ees_ea?: number | null;
  eed?: number | null;
  eed_linear?: number | null;
  esv?: number | null;
  edv?: number | null;
  pmax_scale?: number | null;
}

export interface SessionStatus {
  session_id: string;
  step_status: Record<string, StepStatus>;
  has_image: boolean;
  has_mask: boolean;
  has_calibration: boolean;
  has_beats: boolean;
  has_average: boolean;
  has_analysis: boolean;
}

export interface WaveformData {
  time_s: number[];
  pressure_mmhg: number[];
}
