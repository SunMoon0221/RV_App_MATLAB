"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import type { BeatSegment, WaveformData } from "@/lib/types";
import { averageBeats, detectBeats } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function BeatEditor({
  sessionId,
  trace,
  onAveraged,
}: {
  sessionId: string;
  trace: WaveformData | null;
  onAveraged: () => void;
}) {
  const [beats, setBeats] = useState<BeatSegment[]>([]);
  const [loading, setLoading] = useState(false);
  const [addMode, setAddMode] = useState<"left" | "right" | null>(null);
  const [pending, setPending] = useState<{ left?: number; right?: number }>({});
  const [error, setError] = useState<string | null>(null);

  const keptCount = beats.filter((b) => b.keep).length;

  const runDetect = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await detectBeats(sessionId);
      setBeats(res.beats);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  const toggleKeep = (i: number) => {
    setBeats((b) => b.map((beat, j) => (j === i ? { ...beat, keep: !beat.keep } : beat)));
  };

  const deleteBeat = (i: number) => {
    setBeats((b) => b.filter((_, j) => j !== i));
  };

  const onChartClick = (ev: { points?: { pointIndex?: number }[] }) => {
    const idx = ev.points?.[0]?.pointIndex;
    if (idx == null || !trace) return;
    if (addMode === "left") {
      setPending((p) => ({ ...p, left: idx }));
      setAddMode("right");
    } else if (addMode === "right" && pending.left != null) {
      const left = pending.left;
      const right = idx;
      const seg = trace.pressure_mmhg.slice(Math.min(left, right), Math.max(left, right) + 1);
      const peakRel = seg.indexOf(Math.max(...seg));
      const peakIdx = Math.min(left, right) + peakRel;
      const newBeat: BeatSegment = {
        start_idx: Math.min(left, right),
        end_idx: Math.max(left, right),
        peak_idx: peakIdx,
        start_time: trace.time_s[Math.min(left, right)],
        end_time: trace.time_s[Math.max(left, right)],
        peak_time: trace.time_s[peakIdx],
        peak_pressure: trace.pressure_mmhg[peakIdx],
        keep: true,
        qc_passed: true,
        qc_message: "manual",
      };
      setBeats((b) => [...b, newBeat]);
      setPending({});
      setAddMode(null);
    }
  };

  const submitAverage = async () => {
    if (beats.length === 0) {
      setError("Add or detect at least one beat first.");
      return;
    }
    if (keptCount === 0) {
      setError("Check “Keep” for at least one beat before averaging.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await averageBeats(sessionId, beats);
      onAveraged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Averaging failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => void runDetect()} disabled={loading}>
          Detect beats
        </Button>
        <Button variant="outline" onClick={() => { setAddMode("left"); setPending({}); }}>
          Add beat manually
        </Button>
        <Button onClick={() => void submitAverage()} disabled={loading || beats.length === 0 || keptCount === 0}>
          Average selected beats ({keptCount})
        </Button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {!trace && (
        <p className="text-sm text-amber-800">Calibrated trace not loaded — complete calibration first.</p>
      )}
      {trace && (
        <Plot
          data={[
            {
              x: trace.time_s,
              y: trace.pressure_mmhg,
              type: "scatter",
              mode: "lines",
              name: "Pressure",
              line: { color: "#64748b" },
            },
          ]}
          layout={{
            height: 320,
            margin: { t: 24, r: 16, b: 40, l: 48 },
            xaxis: { title: "Time (s)" },
            yaxis: { title: "mmHg" },
          }}
          config={{ displayModeBar: false }}
          onClick={onChartClick}
        />
      )}
      <div className="max-h-48 overflow-auto rounded border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="p-2 text-left">#</th>
              <th className="p-2">Peak P</th>
              <th className="p-2">Keep</th>
              <th className="p-2">QC</th>
              <th className="p-2" />
            </tr>
          </thead>
          <tbody>
            {beats.map((b, i) => (
              <tr key={i} className="border-b">
                <td className="p-2">{i + 1}</td>
                <td className="p-2 text-center">{b.peak_pressure.toFixed(1)}</td>
                <td className="p-2 text-center">
                  <input type="checkbox" checked={b.keep} onChange={() => toggleKeep(i)} />
                </td>
                <td className="p-2 text-xs text-muted-foreground">{b.qc_message || (b.qc_passed ? "ok" : "fail")}</td>
                <td className="p-2">
                  <Button variant="ghost" size="sm" onClick={() => deleteBeat(i)}>
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
            {beats.length === 0 && (
              <tr>
                <td colSpan={5} className="p-4 text-center text-muted-foreground">
                  No beats — run detection or add manually
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
