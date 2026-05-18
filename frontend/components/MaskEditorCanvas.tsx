"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Circle } from "react-konva";
import type Konva from "konva";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { finalizeMasks, getMasks, updateMasks } from "@/lib/api";

type Mode = "erase" | "trace" | "view";

function decodeMask(b64: string, w: number, h: number): Uint8Array {
  const bin = atob(b64);
  const arr = new Uint8Array(w * h);
  for (let i = 0; i < Math.min(bin.length, w * h); i++) arr[i] = bin.charCodeAt(i) > 127 ? 1 : 0;
  return arr;
}

function encodeMask(mask: Uint8Array, w: number, h: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = mask[i] ? 255 : 0;
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function brushStencil(radius: number): { dx: number; dy: number }[] {
  const key = radius;
  const cache = brushStencil.cache ?? (brushStencil.cache = new Map());
  if (cache.has(key)) return cache.get(key)!;
  const pts: { dx: number; dy: number }[] = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= r2) pts.push({ dx, dy });
    }
  }
  cache.set(key, pts);
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
  const manualRef = useRef<Uint8Array | null>(null);
  const traceRef = useRef<Uint8Array | null>(null);
  const [mode, setMode] = useState<Mode>("erase");
  const [radius, setRadius] = useState(12);
  const [threshold, setThreshold] = useState(128);
  const stageRef = useRef<Konva.Stage>(null);
  const rafRef = useRef<number | null>(null);
  const drawing = useRef(false);
  const tracePath = useRef<{ x: number; y: number }[]>([]);
  const [, bump] = useState(0);

  const load = useCallback(async () => {
    const data = await getMasks(sessionId);
    setDims({ w: data.width, h: data.height });
    manualRef.current = decodeMask(data.manual_erase_b64, data.width, data.height);
    traceRef.current = decodeMask(data.trace_keep_b64, data.width, data.height);
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = data.image_url;
    await new Promise((r) => {
      img.onload = r;
    });
    setBg(img);
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

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

  const onPointer = (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    const stage = stageRef.current;
    if (!stage || !manualRef.current || !traceRef.current) return;
    const pos = stage.getPointerPosition();
    if (!pos) return;
    if (mode === "erase") {
      applyBrush(pos.x, pos.y, false, manualRef.current);
    } else if (mode === "trace") {
      tracePath.current.push({ x: pos.x, y: pos.y });
    }
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        bump((n) => n + 1);
      });
    }
  };

  const commitTrace = async () => {
    // Trace: keep connected components touched by path — simplified: dilate path and AND trace mask
    const data = await getMasks(sessionId);
    const base = decodeMask(data.base_mask_b64, dims.w, dims.h);
    const path = tracePath.current;
    tracePath.current = [];
    if (path.length < 2 || !traceRef.current) return;
    const touched = new Set<number>();
    for (const p of path) {
      for (const { dx, dy } of brushStencil(radius)) {
        const px = Math.round(p.x) + dx;
        const py = Math.round(p.y) + dy;
        if (px >= 0 && px < dims.w && py >= 0 && py < dims.h) touched.add(py * dims.w + px);
      }
    }
    // Flood from touched base pixels
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
    bump((n) => n + 1);
  };

  const save = async () => {
    if (!manualRef.current || !traceRef.current) return;
    await updateMasks(sessionId, {
      manual_erase_keep_mask_b64: encodeMask(manualRef.current, dims.w, dims.h),
      trace_keep_mask_b64: encodeMask(traceRef.current, dims.w, dims.h),
      threshold,
    });
    await finalizeMasks(sessionId);
    onFinalized();
  };

  const resetErase = () => {
    if (manualRef.current) manualRef.current.fill(1);
    bump((n) => n + 1);
  };

  const resetTrace = () => {
    if (traceRef.current) traceRef.current.fill(1);
    bump((n) => n + 1);
  };

  if (!dims.w || !bg) {
    return <p className="text-sm text-muted-foreground">Loading mask editor…</p>;
  }

  const scale = Math.min(1, 900 / dims.w);
  const sw = dims.w * scale;
  const sh = dims.h * scale;

  return (
    <div className="space-y-3">
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
      <div className="flex gap-4">
        <Label>Brush radius</Label>
        <Input type="range" min={4} max={40} value={radius} onChange={(e) => setRadius(Number(e.target.value))} className="w-40" />
        <Label>Threshold</Label>
        <Input type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-20" />
      </div>
      <Stage
        ref={stageRef}
        width={sw}
        height={sh}
        scaleX={scale}
        scaleY={scale}
        onMouseDown={() => {
          drawing.current = true;
        }}
        onMouseUp={() => {
          drawing.current = false;
          if (mode === "trace") void commitTrace();
        }}
        onMouseMove={(e) => {
          if (drawing.current) onPointer(e);
        }}
        className="rounded border border-border bg-black/5"
      >
        <Layer>
          <KonvaImage image={bg} width={dims.w} height={dims.h} />
        </Layer>
      </Stage>
    </div>
  );
}