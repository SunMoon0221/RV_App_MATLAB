"use client";

import dynamic from "next/dynamic";
import type { HemodynamicResults, PmaxMethodResult, WaveformData } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"];

export function PmaxComparisonPlot({
  waveform,
  methods,
  hemodynamics,
}: {
  waveform: WaveformData;
  methods: PmaxMethodResult[];
  hemodynamics: HemodynamicResults | null;
}) {
  const traces: Record<string, unknown>[] = [
    {
      x: waveform.time_s,
      y: waveform.pressure_mmhg,
      type: "scatter",
      mode: "lines",
      name: "Smoothed pressure",
      line: { color: "#64748b", width: 2 },
    },
  ];

  methods.forEach((m, i) => {
    if (!m.success || m.raw_pmax == null) return;
    const color = COLORS[i % COLORS.length];
    traces.push({
      x: [waveform.time_s[0], waveform.time_s[waveform.time_s.length - 1]],
      y: [m.raw_pmax, m.raw_pmax],
      type: "scatter",
      mode: "lines",
      name: `${m.short_name} Pmax=${m.raw_pmax?.toFixed(1)} (scaled ${m.scaled_pmax?.toFixed(1)}) R²=${m.fit_r2?.toFixed(3) ?? "—"}${m.is_selected ? " ★" : ""}`,
      line: { color, dash: "dash" },
    });
    if (m.t_used1 && m.p_used1) {
      traces.push({
        x: m.t_used1,
        y: m.p_used1,
        type: "scatter",
        mode: "markers",
        name: `${m.short_name} window 1`,
        marker: { color, size: 7, symbol: "circle-open", line: { width: 2 } },
        showlegend: false,
      });
    }
    if (m.t_used2 && m.p_used2) {
      traces.push({
        x: m.t_used2,
        y: m.p_used2,
        type: "scatter",
        mode: "markers",
        name: `${m.short_name} window 2`,
        marker: { color, size: 7, symbol: "circle-open", line: { width: 2 } },
        showlegend: false,
      });
    }
    if (m.t_fit && m.p_fit_raw) {
      traces.push({
        x: m.t_fit,
        y: m.p_fit_raw,
        type: "scatter",
        mode: "lines",
        name: `${m.short_name} fit segment`,
        line: { color, width: 2 },
        showlegend: false,
      });
    }
  });

  return (
    <div className="space-y-4">
      <Plot
        data={traces}
        layout={{
          height: 420,
          margin: { t: 40, r: 20, b: 48, l: 56 },
          xaxis: { title: "Time (s)" },
          yaxis: { title: "Pressure (mmHg)" },
          legend: { orientation: "h", y: 1.12 },
        }}
        config={{ responsive: true }}
      />
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/40">
            <th className="p-2 text-left">Method</th>
            <th className="p-2">Raw Pmax</th>
            <th className="p-2">Scaled</th>
            <th className="p-2">R²</th>
            <th className="p-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m.name} className="border-b">
              <td className="p-2">{m.name}{m.is_selected ? " (downstream)" : ""}</td>
              <td className="p-2 text-center">{m.raw_pmax?.toFixed(2) ?? "—"}</td>
              <td className="p-2 text-center">{m.scaled_pmax?.toFixed(2) ?? "—"}</td>
              <td className="p-2 text-center">{m.fit_r2?.toFixed(3) ?? "—"}</td>
              <td className="p-2 text-xs">{m.success ? "OK" : m.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {hemodynamics && (
        <p className="text-sm text-muted-foreground">
          ESP {hemodynamics.esp?.toFixed(1)} · EDP {hemodynamics.edp?.toFixed(1)} · dP/dt max{" "}
          {hemodynamics.dpdt_max?.toFixed(0)}
        </p>
      )}
    </div>
  );
}
