"use client";

import { cn } from "@/lib/utils";
import type { StepId, StepStatus } from "@/lib/types";

const STEPS: { id: StepId; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "process", label: "Process" },
  { id: "edit_mask", label: "Edit Mask" },
  { id: "calibrate", label: "Calibrate" },
  { id: "average_beats", label: "Average Beats" },
  { id: "single_beat", label: "Single Beat" },
  { id: "export", label: "Export" },
];

export function WorkflowStepper({
  current,
  statuses,
}: {
  current: StepId;
  statuses: Record<string, StepStatus>;
}) {
  const idx = STEPS.findIndex((s) => s.id === current);
  return (
    <nav className="flex flex-wrap gap-2 border-b border-border bg-card px-4 py-3">
      {STEPS.map((step, i) => {
        const st = statuses[step.id] ?? "not_ready";
        const active = step.id === current;
        const done =
          i < idx ||
          st === "processed" ||
          st === "ready" ||
          st === "averaged" ||
          st === "analysis_complete";
        return (
          <div
            key={step.id}
            className={cn(
              "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm",
              active && "bg-primary text-primary-foreground",
              !active && done && "bg-muted",
              !active && !done && "text-muted-foreground"
            )}
          >
            <span className="font-mono text-xs opacity-70">{i + 1}</span>
            {step.label}
          </div>
        );
      })}
    </nav>
  );
}
