"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { WaveformData } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

/** Build event-marker signal client-side for peak picking UI */
function eventMarker(pressure: number[], sigma = 15): number[] {
  const n = pressure.length;
  const dp = pressure.map((_, i) => (i ? pressure[i] - pressure[i - 1] : 0));
  const d2 = dp.map((_, i) => (i ? dp[i] - dp[i - 1] : 0));
  const em = dp.map((v, i) => Math.abs(v) + 0.5 * Math.abs(d2[i]));
  const max = Math.max(...em, 1e-9);
  return em.map((v) => v / max);
}

function findPeaks(sig: number[], minDist = 20): number[] {
  const peaks: number[] = [];
  for (let i = 1; i < sig.length - 1; i++) {
    if (sig[i] > sig[i - 1] && sig[i] >= sig[i + 1]) {
      if (!peaks.length || i - peaks[peaks.length - 1] >= minDist) peaks.push(i);
    }
  }
  return peaks;
}

export function PeakSelectionModal({
  waveform,
  open,
  onConfirm,
  onCancel,
}: {
  waveform: WaveformData;
  open: boolean;
  onConfirm: (peaks: {
    event_marker_peak_indices: number[];
    edp_peak_indices: number[];
    esp_peak_indices: number[];
    smoothing_sigma_ms: number;
  }) => void;
  onCancel: () => void;
}) {
  const [sigma, setSigma] = useState(70);
  const [selected, setSelected] = useState<number[]>([]);

  const em = useMemo(() => eventMarker(waveform.pressure_mmhg, sigma), [waveform, sigma]);
  const candidates = useMemo(() => findPeaks(em), [em]);

  if (!open) return null;

  const toggle = (idx: number) => {
    setSelected((s) => {
      if (s.includes(idx)) return s.filter((x) => x !== idx);
      if (s.length >= 4) return s;
      return [...s, idx].sort((a, b) => a - b);
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="max-h-[90vh] w-full max-w-3xl overflow-auto rounded-lg bg-card p-6 shadow-xl">
        <h2 className="mb-2 text-lg font-semibold">Select 4 event-marker peaks</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Gray = pressure, blue = event marker. Click candidate peaks (exactly 4).
        </p>
        <Label>Smoothing (ms scale)</Label>
        <Input type="range" min={20} max={120} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} />
        <Plot
          data={[
            {
              x: waveform.time_s,
              y: waveform.pressure_mmhg,
              type: "scatter",
              mode: "lines",
              name: "Pressure",
              line: { color: "#94a3b8" },
            },
            {
              x: waveform.time_s,
              y: em.map((v) => v * Math.max(...waveform.pressure_mmhg)),
              type: "scatter",
              mode: "lines",
              name: "Event marker",
              line: { color: "#3b82f6" },
              yaxis: "y",
            },
            {
              x: candidates.map((i) => waveform.time_s[i]),
              y: candidates.map((i) => waveform.pressure_mmhg[i]),
              type: "scatter",
              mode: "markers",
              name: "Candidates",
              marker: { color: "#fbbf24", size: 8 },
            },
            {
              x: selected.map((i) => waveform.time_s[i]),
              y: selected.map((i) => waveform.pressure_mmhg[i]),
              type: "scatter",
              mode: "markers",
              name: "Selected",
              marker: { color: "#22c55e", size: 12, symbol: "diamond" },
            },
          ]}
          layout={{ height: 360, margin: { t: 30, r: 20, b: 40, l: 50 } }}
          config={{ displayModeBar: false }}
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {candidates.map((idx) => (
            <Button
              key={idx}
              size="sm"
              variant={selected.includes(idx) ? "default" : "outline"}
              onClick={() => toggle(idx)}
            >
              t={waveform.time_s[idx].toFixed(3)}s
            </Button>
          ))}
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            disabled={selected.length !== 4}
            onClick={() =>
              onConfirm({
                event_marker_peak_indices: selected,
                edp_peak_indices: [selected[0]],
                esp_peak_indices: [selected[2]],
                smoothing_sigma_ms: sigma,
              })
            }
          >
            Confirm ({selected.length}/4)
          </Button>
        </div>
      </div>
    </div>
  );
}
