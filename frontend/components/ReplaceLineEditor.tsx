"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { replaceLine } from "@/lib/api";

export function ReplaceLineEditor({
  sessionId,
  onDone,
}: {
  sessionId: string;
  onDone?: () => void;
}) {
  const [p1, setP1] = useState("");
  const [p2, setP2] = useState("");

  const parse = (s: string): [number, number] | null => {
    const parts = s.split(",").map((x) => parseFloat(x.trim()));
    if (parts.length !== 2 || parts.some((x) => Number.isNaN(x))) return null;
    return [parts[0], parts[1]];
  };

  const [error, setError] = useState<string | null>(null);

  const apply = async () => {
    const a = parse(p1);
    const b = parse(p2);
    if (!a || !b) {
      setError("Enter two points as x,y (pixel coordinates)");
      return;
    }
    setError(null);
    try {
      await replaceLine(sessionId, a, b);
      onDone?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Replace line failed");
    }
  };

  return (
    <div className="mt-4 space-y-2 rounded border border-border p-4">
      <p className="text-sm font-medium">Replace line (optional)</p>
      <p className="text-xs text-muted-foreground">Enter pixel coordinates x,y for two endpoints</p>
      <input
        className="w-full rounded border px-2 py-1 text-sm"
        placeholder="Point 1: x, y"
        value={p1}
        onChange={(e) => setP1(e.target.value)}
      />
      <input
        className="w-full rounded border px-2 py-1 text-sm"
        placeholder="Point 2: x, y"
        value={p2}
        onChange={(e) => setP2(e.target.value)}
      />
      {error && <p className="text-xs text-red-600">{error}</p>}
      <Button size="sm" variant="outline" onClick={() => void apply()}>
        Apply replace line
      </Button>
    </div>
  );
}
