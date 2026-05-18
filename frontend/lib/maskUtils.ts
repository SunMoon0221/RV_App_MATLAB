/** Decode server PNG masks and build overlay canvases for Konva. */

export async function decodeMaskPng(b64: string): Promise<{ mask: Uint8Array; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = img.width;
      const h = img.height;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas not available"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, w, h);
      const mask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) {
        mask[i] = data.data[i * 4] > 127 ? 1 : 0;
      }
      resolve({ mask, width: w, height: h });
    };
    img.onerror = () => reject(new Error("Failed to decode mask PNG"));
    const src = b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
    img.src = src;
  });
}

export function composeMasks(base: Uint8Array, manual: Uint8Array, trace: Uint8Array): Uint8Array {
  const n = base.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = base[i] && manual[i] && trace[i] ? 1 : 0;
  }
  return out;
}

/** Green semi-transparent overlay for Konva Image layer. */
export function maskToOverlayImage(
  filtered: Uint8Array,
  width: number,
  height: number,
  color: [number, number, number, number] = [0, 255, 100, 140]
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  const [r, g, b, a] = color;
  for (let i = 0; i < width * height; i++) {
    if (filtered[i]) {
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function encodeMaskPng(mask: Uint8Array, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = mask[i] ? 255 : 0;
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}
