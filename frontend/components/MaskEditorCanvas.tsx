"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";
import type Konva from "konva";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { finalizeMasks, getMasks, updateMasks } from "@/lib/api";
import {
  composeMasks,
  decodeMaskPng,
  encodeMaskPng,
  maskToOverlayImage,
} from "@/lib/maskUtils";

type Mode = "erase" | "trace";

function brushStencil(radius: number): { dx: number; dy: number }[] {
  const cache = brushStencil.cache ?? (brushStencil.cache = new Map<number, { dx: number; dy: number }[]>());
  if (cache.has(radius)) return cache.get(radius)!;
  const pts: { dx: number; dy: number }[] = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= r2) pts.push({ dx, dy });
    }
  }
  cache.set(radius, pts);
  return pts;
}
brushStencil.cache = new Map<number, { dx: number; dy: number }[]>();

export function MaskEditorCanvas({
  sessionId,
  onFinalized,
}: {
  sessionId: string;
  onFinalized: () => void;
}) {
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const [bg, setBg] = useState<HTMLImageElement | null>(null);
  const [overlayCanvas, setOverlayCanvas] = useState<HTMLCanvasElement | null>(null);
  const [medianLine, setMedianLine] = useState<number[]>([]);
  const baseRef = useRef<Uint8Array | null>(null);
  const manualRef = useRef<Uint8Array | null>(null);
  const traceRef = useRef<Uint8Array | null>(null);
  const [mode, setMode] = useState<Mode>("erase");
  const [radius, setRadius] = useState(12);
  const [threshold, setThreshold] = useState(128);
  const stageRef = useRef<Konva.Stage>(null);
  const rafRef = useRef<number | null>(null);
  const drawing = useRef(false);
  const tracePath = useRef<{ x: number; y: number }[]>([]);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [overlayImg, setOverlayImg] = useState<HTMLImageElement | undefined>(undefined);

  const rebuildOverlay = useCallback(() => {
    if (!baseRef.current || !manualRef.current || !traceRef.current || !dims.w) return;
    const filtered = composeMasks(baseRef.current, manualRef.current, traceRef.current);
    setOverlayCanvas(maskToOverlayImage(filtered, dims.w, dims.h));
  }, [dims.w, dims.h]);

  useEffect(() => {
    rebuildOverlay();
  }, [revision, rebuildOverlay]);

  const load = useCallback(async () => {
    setError(null);
    const data = await getMasks(sessionId);
    setDims({ w: data.width, h: data.height });
    const [baseDec, manualDec, traceDec] = await Promise.all([
      decodeMaskPng(data.base_mask_b64),
      decodeMaskPng(data.manual_erase_b64),
      decodeMaskPng(data.trace_keep_b64),
    ]);
    baseRef.current = baseDec.mask;
    manualRef.current = manualDec.mask;
    traceRef.current = traceDec.mask;

    const pts: number[] = [];
    const xs = data.median_rows.x;
    const ys = data.median_rows.y;
    for (let i = 0; i < xs.length; i++) {
      const y = ys[i];
      if (y != null && Number.isFinite(y)) {
        pts.push(xs[i], y);
      }
    }
    setMedianLine(pts);

    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = `${data.image_url}?t=${Date.now()}`;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Failed to load image"));
    });
    setBg(img);
    setRevision((n) => n + 1);
  }, [sessionId]);

  useEffect(() => {
    void load().catch((e) => setError(e instanceof Error ? e.message : "Failed to load masks"));
  }, [load]);

  const pointerInImage = (stage: Konva.Stage): { x: number; y: number } | null => {
    const pos = stage.getRelativePointerPosition();
    if (!pos) return null;
    return { x: pos.x, y: pos.y };
  };

  const applyBrush = (x: number, y: number, keep: boolean, target: Uint8Array) => {
    const { w, h } = dims;
    const stencil = brushStencil(radius);
    const xi = Math.round(x);
    const yi = Math.round(y);
    for (const { dx, dy } of stencil) {
      const px = xi + dx;
      const py = yi + dy;
      if (px >= 0 && px < w && py >= 0 && py < h) {
        target[py * w + px] = keep ? 1 : 0;
      }
    }
  };

  const onPointerMove = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (!drawing.current) return;
    const stage = stageRef.current;
    if (!stage || !manualRef.current || !traceRef.current) return;
    const pos = pointerInImage(stage);
    if (!pos) return;
    if (mode === "erase") {
      applyBrush(pos.x, pos.y, false, manualRef.current);
    } else {
      tracePath.current.push({ x: pos.x, y: pos.y });
    }
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        setRevision((n) => n + 1);
      });
    }
  };

  const commitTrace = () => {
    if (!baseRef.current || !traceRef.current || tracePath.current.length < 2) {
      tracePath.current = [];
      return;
    }
    const base = baseRef.current;
    const path = tracePath.current;
    tracePath.current = [];
    const touched = new Set<number>();
    for (const p of path) {
      for (const { dx, dy } of brushStencil(radius)) {
        const px = Math.round(p.x) + dx;
        const py = Math.round(p.y) + dy;
        if (px >= 0 && px < dims.w && py >= 0 && py < dims.h) touched.add(py * dims.w + px);
      }
    }
    const keep = new Uint8Array(dims.w * dims.h);
    const visited = new Uint8Array(dims.w * dims.h);
    const stack: number[] = [];
    for (const idx of touched) {
      if (base[idx]) stack.push(idx);
    }
    while (stack.length) {
      const idx = stack.pop()!;
      if (visited[idx]) continue;
      visited[idx] = 1;
      if (!base[idx]) continue;
      keep[idx] = 1;
      const x = idx % dims.w;
      const y = (idx / dims.w) | 0;
      for (const [nx, ny] of [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ]) {
        if (nx >= 0 && nx < dims.w && ny >= 0 && ny < dims.h) {
          const ni = ny * dims.w + nx;
          if (!visited[ni] && base[ni]) stack.push(ni);
        }
      }
    }
    traceRef.current = keep;
    setRevision((n) => n + 1);
  };

  const save = async () => {
    if (!manualRef.current || !traceRef.current) return;
    setError(null);
    try {
      await updateMasks(sessionId, {
        manual_erase_keep_mask_b64: encodeMaskPng(manualRef.current, dims.w, dims.h),
        trace_keep_mask_b64: encodeMaskPng(traceRef.current, dims.w, dims.h),
        threshold,
      });
      await finalizeMasks(sessionId);
      onFinalized();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const resetErase = () => {
    if (manualRef.current) manualRef.current.fill(1);
    setRevision((n) => n + 1);
  };

  const resetTrace = () => {
    if (traceRef.current) traceRef.current.fill(1);
    setRevision((n) => n + 1);
  };

  const scale = dims.w ? Math.min(1, 900 / dims.w) : 1;
  const stageW = dims.w * scale;
  const stageH = dims.h * scale;

  useEffect(() => {
    if (!overlayCanvas) {
      setOverlayImg(undefined);
      return;
    }
    const img = new window.Image();
    img.onload = () => setOverlayImg(img);
    img.src = overlayCanvas.toDataURL();
  }, [overlayCanvas, revision]);

  if (error) {
    return <p className="text-sm text-red-600">{error}</p>;
  }

  if (!dims.w || !bg) {
    return <p className="text-sm text-muted-foreground">Loading mask editor…</p>;
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Green overlay = extracted trace mask. Eraser removes mask; trace-keep keeps only components you draw over.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button variant={mode === "erase" ? "default" : "outline"} size="sm" onClick={() => setMode("erase")}>
          Eraser
        </Button>
        <Button variant={mode === "trace" ? "default" : "outline"} size="sm" onClick={() => setMode("trace")}>
          Trace keep
        </Button>
        <Button variant="outline" size="sm" onClick={resetErase}>
          Reset erase
        </Button>
        <Button variant="outline" size="sm" onClick={resetTrace}>
          Reset trace
        </Button>
        <Button size="sm" onClick={() => void save()}>
          Finalize mask
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <Label>Brush radius: {radius}</Label>
        <Input
          type="range"
          min={4}
          max={40}
          value={radius}
          onChange={(e) => setRadius(Number(e.target.value))}
          className="w-40"
        />
        <Label>Threshold</Label>
        <Input
          type="number"
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
          className="w-20"
        />
      </div>
      <Stage
        ref={stageRef}
        width={stageW}
        height={stageH}
        scaleX={scale}
        scaleY={scale}
        onMouseDown={(e) => {
          drawing.current = true;
          onPointerMove(e);
        }}
        onMouseUp={() => {
          drawing.current = false;
          if (mode === "trace") commitTrace();
        }}
        onMouseLeave={() => {
          if (drawing.current && mode === "trace") commitTrace();
          drawing.current = false;
        }}
        onMouseMove={onPointerMove}
        onTouchStart={(e) => {
          e.evt.preventDefault();
          drawing.current = true;
          onPointerMove(e);
        }}
        onTouchMove={(e) => {
          e.evt.preventDefault();
          if (drawing.current) onPointerMove(e);
        }}
        onTouchEnd={() => {
          drawing.current = false;
          if (mode === "trace") commitTrace();
        }}
        className="rounded border border-border bg-neutral-900/10 shadow-inner"
      >
        <Layer>
          <KonvaImage image={bg} width={dims.w} height={dims.h} />
          {overlayImg && (
            <KonvaImage image={overlayImg} width={dims.w} height={dims.h} listening={false} />
          )}
          {medianLine.length >= 4 && (
            <Line
              points={medianLine}
              stroke="#f59e0b"
              strokeWidth={2}
              listening={false}
              tension={0}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
}

