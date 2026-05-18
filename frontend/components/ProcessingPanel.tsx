"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { processImage } from "@/lib/api";

export function ProcessingPanel({
  sessionId,
  onProcessed,
}: {
  sessionId: string;
  onProcessed: () => void;
}) {
  const [threshold, setThreshold] = useState(128);
  const [notch, setNotch] = useState(false);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      await processImage(sessionId, { threshold, apply_notch_filter: notch });
      onProcessed();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <Label>Threshold (0–255)</Label>
        <Input
          type="number"
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={notch} onChange={(e) => setNotch(e.target.checked)} />
        Apply notch / Fourier filter
      </label>
      <Button onClick={() => void run()} disabled={loading}>
        Process Image
      </Button>
    </div>
  );
}
