"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { uploadImage, createMockSession } from "@/lib/api";

export function ImageUploader({
  onUploaded,
}: {
  onUploaded: (sessionId: string, previewUrl: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const res = await uploadImage(file);
        onUploaded(res.session_id, res.preview_url);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setLoading(false);
      }
    },
    [onUploaded]
  );

  const mock = async () => {
    setLoading(true);
    try {
      const res = await createMockSession();
      onUploaded(res.session_id, `/api/session/${res.session_id}/calibrated_trace.json`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Mock failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 p-12 hover:bg-muted/50">
        <span className="text-sm text-muted-foreground">Drop waveform screenshot or click to browse</span>
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleFile(f);
          }}
        />
      </label>
      <Button variant="outline" onClick={() => void mock()} disabled={loading}>
        Load synthetic demo session
      </Button>
      {loading && <p className="text-sm text-muted-foreground">Uploading…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
