"use client";

import type { HemodynamicResults, PmaxMethodResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

const FIELDS: { key: keyof HemodynamicResults; label: string; unit?: string }[] = [
  { key: "esp", label: "ESP", unit: "mmHg" },
  { key: "edp", label: "EDP", unit: "mmHg" },
  { key: "pmax", label: "Pmax", unit: "mmHg" },
  { key: "ees", label: "Ees" },
  { key: "ea", label: "Ea" },
  { key: "sv", label: "SV", unit: "mL" },
  { key: "dpdt_max", label: "dP/dt max" },
  { key: "dpdt_min", label: "dP/dt min" },
  { key: "beta", label: "β" },
  { key: "tau", label: "τ", unit: "s" },
  { key: "ees_ea", label: "Ees/Ea" },
  { key: "eed", label: "Eed" },
  { key: "eed_linear", label: "Eed linear" },
];

export function ResultsPanel({
  hemodynamics,
  selectedMethod,
  methods,
}: {
  hemodynamics: HemodynamicResults | null;
  selectedMethod: string;
  methods: PmaxMethodResult[];
}) {
  if (!hemodynamics) {
    return <p className="text-sm text-muted-foreground">Run single-beat analysis to view results.</p>;
  }

  const fmt = (v: number | null | undefined) =>
    v == null || Number.isNaN(v) ? "—" : typeof v === "number" ? v.toFixed(3) : String(v);

  return (
    <div className="space-y-4">
      <Badge variant="success">Analysis complete</Badge>
      <p className="text-xs text-muted-foreground">Downstream Pmax: {selectedMethod}</p>
      <dl className="grid grid-cols-2 gap-2 text-sm">
        {FIELDS.map(({ key, label, unit }) => (
          <div key={key} className="rounded-md border border-border bg-muted/20 px-3 py-2">
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="font-mono font-medium">
              {fmt(hemodynamics[key] as number)}
              {unit && hemodynamics[key] != null ? ` ${unit}` : ""}
            </dd>
          </div>
        ))}
      </dl>
      {methods.some((m) => !m.success) && (
        <p className="text-xs text-amber-800">
          {methods.filter((m) => !m.success).length} Pmax method(s) failed — see comparison plot.
        </p>
      )}
    </div>
  );
}
