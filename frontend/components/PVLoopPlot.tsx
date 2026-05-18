"use client";

import dynamic from "next/dynamic";
import type { HemodynamicResults } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

/** Simplified PV loop from ESP/EDP and stroke volume (research placeholder). */
export function PVLoopPlot({ hemo, sv }: { hemo: HemodynamicResults; sv: number }) {
  const esp = hemo.esp ?? 20;
  const edp = hemo.edp ?? 5;
  const edv = (hemo.edv ?? sv * 1.5) || 80;
  const esv = (hemo.esv ?? edv - sv) || 50;

  const v = [esv, edv, edv, esv, esv];
  const p = [esp, edp, edp, esp, esp];

  return (
    <Plot
      data={[
        {
          x: v,
          y: p,
          type: "scatter",
          mode: "lines+markers",
          fill: "toself",
          fillcolor: "rgba(37,99,235,0.15)",
          line: { color: "#2563eb" },
          name: "PV loop",
        },
      ]}
      layout={{
        height: 280,
        margin: { t: 24, r: 16, b: 40, l: 48 },
        xaxis: { title: "Volume (mL)" },
        yaxis: { title: "Pressure (mmHg)" },
      }}
      config={{ displayModeBar: false }}
    />
  );
}
