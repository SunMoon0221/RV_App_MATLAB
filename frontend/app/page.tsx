"use client";

import { useCallback, useEffect, useState } from "react";
import { WorkflowStepper } from "@/components/WorkflowStepper";
import { ImageUploader } from "@/components/ImageUploader";
import { ProcessingPanel, ProcessContinueButton } from "@/components/ProcessingPanel";
import dynamic from "next/dynamic";

const MaskEditorCanvas = dynamic(() => import("@/components/MaskEditorCanvas").then((m) => m.MaskEditorCanvas), {
  ssr: false,
});
const CalibrationCanvas = dynamic(() => import("@/components/CalibrationCanvas").then((m) => m.CalibrationCanvas), {
  ssr: false,
});
import { BeatEditor } from "@/components/BeatEditor";
import { PeakSelectionModal } from "@/components/PeakSelectionModal";
import { PmaxComparisonPlot } from "@/components/PmaxComparisonPlot";
import { PVLoopPlot } from "@/components/PVLoopPlot";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ReplaceLineEditor } from "@/components/ReplaceLineEditor";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import type { StepId, StepStatus, WaveformData, HemodynamicResults, PmaxMethodResult } from "@/lib/types";
import {
  exportZipUrl,
  getAveragedWaveform,
  getCalibratedTrace,
  getSessionStatus,
  runSingleBeatAnalysis,
  getAnalysis,
} from "@/lib/api";

export default function HomePage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [maskPreviewUrl, setMaskPreviewUrl] = useState<string | null>(null);
  const [step, setStep] = useState<StepId>("upload");
  const [statuses, setStatuses] = useState<Record<string, StepStatus>>({});
  const [calibrated, setCalibrated] = useState<WaveformData | null>(null);
  const [averaged, setAveraged] = useState<WaveformData | null>(null);
  const [hemo, setHemo] = useState<HemodynamicResults | null>(null);
  const [methods, setMethods] = useState<PmaxMethodResult[]>([]);
  const [selectedMethod, setSelectedMethod] = useState("Original Piecewise Sinusoid");
  const [sv, setSv] = useState(30);
  const [pmaxScale, setPmaxScale] = useState(1);
  const [peakModal, setPeakModal] = useState(false);
  const [pendingAnalysis, setPendingAnalysis] = useState(false);
  const [isMockSession, setIsMockSession] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const refreshStatus = useCallback(async (id: string) => {
    const s = await getSessionStatus(id);
    setStatuses(s.step_status as Record<string, StepStatus>);
  }, []);

  useEffect(() => {
    if (sessionId) void refreshStatus(sessionId);
  }, [sessionId, step, refreshStatus]);

  const resetWorkflowInputs = () => {
    setSv(30);
    setPmaxScale(1);
    setSelectedMethod("Original Piecewise Sinusoid");
    setPeakModal(false);
    setPendingAnalysis(false);
    setGlobalError(null);
  };

  const onUploaded = (id: string, url: string | null, opts?: { isMock?: boolean }) => {
    setSessionId(id);
    setPreviewUrl(url);
    setMaskPreviewUrl(null);
    setCalibrated(null);
    setAveraged(null);
    setHemo(null);
    setMethods([]);
    resetWorkflowInputs();
    const isMock = opts?.isMock ?? false;
    setIsMockSession(isMock);
    setStep(isMock ? "average_beats" : "process");
    void refreshStatus(id).catch(() => undefined);
    if (isMock) {
      void getCalibratedTrace(id)
        .then(setCalibrated)
        .catch((e) => setGlobalError(e instanceof Error ? e.message : "Failed to load demo trace"));
    }
  };

  const loadCalibrated = async () => {
    if (!sessionId) return;
    const t = await getCalibratedTrace(sessionId);
    setCalibrated(t);
  };

  const loadAveraged = async () => {
    if (!sessionId) return;
    const t = await getAveragedWaveform(sessionId);
    setAveraged(t);
  };

  const startAnalysis = () => {
    setPendingAnalysis(true);
    setPeakModal(true);
  };

  const confirmPeaks = async (peaks: {
    event_marker_peak_indices: number[];
    edp_peak_indices: number[];
    esp_peak_indices: number[];
    smoothing_sigma_ms: number;
  }) => {
    if (!sessionId) return;
    setPeakModal(false);
    setGlobalError(null);
    try {
      const res = await runSingleBeatAnalysis(sessionId, {
        stroke_volume_ml: sv,
        pmax_scale_factor: pmaxScale,
        selected_pmax_method: selectedMethod,
        peak_selection: peaks,
      });
      setHemo(res.hemodynamics);
      setMethods(res.pmax_methods);
      setSelectedMethod(res.selected_method);
      setStep("export");
      void refreshStatus(sessionId);
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setPendingAnalysis(false);
    }
  };

  const loadExistingAnalysis = async () => {
    if (!sessionId) return;
    const res = await getAnalysis(sessionId);
    setHemo(res.hemodynamics);
    setMethods(res.pmax_methods);
    setSelectedMethod(res.selected_method);
  };

  const statusBadge = (key: string) => {
    const st = statuses[key] ?? "not_ready";
    const variant =
      st === "analysis_complete" || st === "averaged" || st === "processed" || st === "ready"
        ? "success"
        : st === "warning" || st === "needs_calibration"
          ? "warning"
          : "muted";
    return <Badge variant={variant}>{st.replace(/_/g, " ")}</Badge>;
  };

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">RV Single-Beat Pressure-Volume Analysis</h1>
        <p className="text-sm text-muted-foreground">Local-first web workflow · session {sessionId?.slice(0, 8) ?? "—"}</p>
      </header>
      <WorkflowStepper current={step} statuses={statuses} />
      <div className="flex flex-1 flex-col lg:flex-row">
        <main className="flex-1 overflow-auto p-6">
          {step === "upload" && <ImageUploader onUploaded={onUploaded} />}
          {step === "process" && sessionId && (
            <>
              <ProcessingPanel
                sessionId={sessionId}
                originalPreviewUrl={previewUrl}
                onProcessed={(maskUrl) => {
                  setMaskPreviewUrl(maskUrl);
                  void refreshStatus(sessionId);
                }}
              />
              <ProcessContinueButton
                visible={!!maskPreviewUrl}
                onContinue={() => setStep("edit_mask")}
              />
            </>
          )}
          {step === "edit_mask" && sessionId && (
            <>
              {maskPreviewUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={maskPreviewUrl}
                  alt="Processed mask"
                  className="mb-4 max-h-48 rounded border object-contain"
                />
              )}
              <MaskEditorCanvas
              sessionId={sessionId}
              onFinalized={() => {
                setStep("calibrate");
                void refreshStatus(sessionId);
              }}
            />
            </>
          )}
          {step === "calibrate" && sessionId && (
            <ReplaceLineEditor sessionId={sessionId} onDone={() => void loadCalibrated()} />
          )}
          {step === "calibrate" && sessionId && previewUrl && !isMockSession && (
            <CalibrationCanvas
              sessionId={sessionId}
              imageUrl={previewUrl}
              onCalibrated={() => {
                void loadCalibrated();
                setStep("average_beats");
                void refreshStatus(sessionId);
              }}
            />
          )}
          {step === "average_beats" && sessionId && (
            <BeatEditor
              sessionId={sessionId}
              trace={calibrated}
              onAveraged={() => {
                void loadAveraged();
                setStep("single_beat");
                void refreshStatus(sessionId);
              }}
            />
          )}
          {(step === "single_beat" || step === "export") && averaged && (
            <PmaxComparisonPlot waveform={averaged} methods={methods} hemodynamics={hemo} />
          )}
          {step === "export" && hemo && <div className="mt-6"><PVLoopPlot hemo={hemo} sv={sv} /></div>}
        </main>
        <aside className="w-full border-t border-border bg-card p-6 lg:w-96 lg:border-l lg:border-t-0">
          <div className="mb-4 flex flex-wrap gap-2">
            {statusBadge("process")}
            {statusBadge("calibrate")}
            {statusBadge("average_beats")}
            {statusBadge("single_beat")}
          </div>
          {globalError && (
            <p className="mb-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">{globalError}</p>
          )}
          <nav className="mb-6 flex flex-col gap-2">
            {(
              [
                "upload",
                "process",
                "edit_mask",
                "calibrate",
                "average_beats",
                "single_beat",
                "export",
              ] as StepId[]
            ).map((s) => (
              <Button
                key={s}
                variant={step === s ? "default" : "outline"}
                size="sm"
                className="justify-start"
                onClick={() => {
                  if (s === "upload") {
                    setSessionId(null);
                    setPreviewUrl(null);
                    setMaskPreviewUrl(null);
                    setCalibrated(null);
                    setAveraged(null);
                    setHemo(null);
                    setMethods([]);
                    setIsMockSession(false);
                    resetWorkflowInputs();
                  }
                  setStep(s);
                  if (s === "calibrate" || s === "average_beats") {
                    void loadCalibrated().catch((e) =>
                      setGlobalError(e instanceof Error ? e.message : "Load trace failed")
                    );
                  }
                  if (s === "single_beat" || s === "export") {
                    void loadAveraged().catch((e) =>
                      setGlobalError(e instanceof Error ? e.message : "Load average failed")
                    );
                  }
                  if (s === "export") {
                    void loadExistingAnalysis().catch(() => undefined);
                  }
                }}
              >
                Go to {s.replace(/_/g, " ")}
              </Button>
            ))}
          </nav>
          {(step === "single_beat" || step === "export") && sessionId && (
            <div className="space-y-4 border-t border-border pt-4">
              <div>
                <Label>Stroke volume (mL)</Label>
                <Input type="number" value={sv} onChange={(e) => setSv(Number(e.target.value))} />
              </div>
              <div>
                <Label>Pmax scale factor</Label>
                <Input type="number" step={0.01} value={pmaxScale} onChange={(e) => setPmaxScale(Number(e.target.value))} />
              </div>
              <div>
                <Label>Downstream Pmax method</Label>
                <select
                  className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={selectedMethod}
                  onChange={(e) => setSelectedMethod(e.target.value)}
                >
                  <option>Original Piecewise Sinusoid</option>
                  <option>Brimioulle Sine</option>
                  <option>RV IsoMax Sine</option>
                  <option>Second Derivative Sine</option>
                  <option>Kremer/Shih Tangent</option>
                </select>
              </div>
              <Button onClick={startAnalysis} disabled={!averaged || pendingAnalysis}>
                Run single-beat analysis
              </Button>
            </div>
          )}
          <ResultsPanel hemodynamics={hemo} selectedMethod={selectedMethod} methods={methods} />
          {step === "export" && sessionId && (
            <a href={exportZipUrl(sessionId)} className="mt-4 inline-block">
              <Button className="w-full">Download all exports (ZIP)</Button>
            </a>
          )}
        </aside>
      </div>
      {averaged && (
        <PeakSelectionModal
          waveform={averaged}
          open={peakModal}
          onConfirm={(p) => void confirmPeaks(p)}
          onCancel={() => {
            setPeakModal(false);
            setPendingAnalysis(false);
          }}
        />
      )}
    </div>
  );
}
