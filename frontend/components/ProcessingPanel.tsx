"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { processImage } from "@/lib/api";

export function ProcessingPanel({
  sessionId,
  originalPreviewUrl,
  onProcessed,
}: {
  sessionId: string;
  originalPreviewUrl: string | null;
  onProcessed: (maskPreviewUrl: string) => void;
}) {
  const [threshold, setThreshold] = useState(128);
  const [notch, setNotch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await processImage(sessionId, { threshold, apply_notch_filter: notch });
      const url = `${res.mask_preview_url}?t=${Date.now()}`;
      setResultUrl(url);
      onProcessed(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Original</p>
          {originalPreviewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={originalPreviewUrl}
              alt="Original waveform"
              className="max-h-80 w-full rounded border object-contain bg-neutral-100"
            />
          ) : (
            <p className="text-sm text-muted-foreground">No image</p>
          )}
        </div>
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Processed mask (green = trace)
          </p>
          {resultUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={resultUrl}
              alt="Processed mask overlay"
              className="max-h-80 w-full rounded border object-contain bg-neutral-100"
            />
          ) : (
            <div className="flex max-h-80 min-h-[200px] items-center justify-center rounded border border-dashed bg-muted/30 text-sm text-muted-foreground">
              Click Process Image to see mask overlay
            </div>
          )}
        </div>
      </div>
      <div>
        <Label>Threshold (0–255)</Label>
        <p className="mb-1 text-xs text-muted-foreground">Lower = more pixels treated as ink (darker trace)</p>
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
      {error && <p className="text-sm text-red-600">{error}</p>}
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => void run()} disabled={loading}>
          {loading ? "Processing…" : "Process Image"}
        </Button>
        {resultUrl && (
          <Button variant="outline" onClick={() => void run()} disabled={loading}>
            Re-process
          </Button>
        )}
      </div>
    </div>
  );
}

export function ProcessContinueButton({
  visible,
  onContinue,
}: {
  visible: boolean;
  onContinue: () => void;
}) {
  if (!visible) return null;
  return (
    <Button className="mt-2" onClick={onContinue}>
      Continue to Edit Mask →
    </Button>
  );
}
