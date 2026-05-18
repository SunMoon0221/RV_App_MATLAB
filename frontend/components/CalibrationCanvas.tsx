"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Line, Circle, Image as KonvaImage } from "react-konva";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { calibrate } from "@/lib/api";

type CalStep = "horizontal" | "vertical" | "origin" | "done";

export function CalibrationCanvas({
  sessionId,
  imageUrl,
  onCalibrated,
}: {
  sessionId: string;
  imageUrl: string;
  onCalibrated: () => void;
}) {
  const [bg, setBg] = useState<HTMLImageElement | null>(null);
  const [step, setStep] = useState<CalStep>("horizontal");
  const [hLine, setHLine] = useState<number[] | null>(null);
  const [vLine, setVLine] = useState<number[] | null>(null);
  const [origin, setOrigin] = useState<[number, number] | null>(null);
  const [timeSpan, setTimeSpan] = useState(1);
  const [pressureSpan, setPressureSpan] = useState(30);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [dims, setDims] = useState({ w: 800, h: 400 });

  useEffect(() => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      setBg(img);
      setDims({ w: img.width, h: img.height });
    };
  }, [imageUrl]);

  const onStageClick = useCallback(
    (x: number, y: number) => {
      if (step === "horizontal") {
        if (!dragRef.current) dragRef.current = { x, y };
        else {
          setHLine([dragRef.current.x, dragRef.current.y, x, y]);
          dragRef.current = null;
          setStep("vertical");
        }
      } else if (step === "vertical") {
        if (!dragRef.current) dragRef.current = { x, y };
        else {
          setVLine([dragRef.current.x, dragRef.current.y, x, y]);
          dragRef.current = null;
          setStep("origin");
        }
      } else if (step === "origin") {
        setOrigin([x, y]);
        setStep("done");
      }
    },
    [step]
  );

  const submit = async () => {
    if (!hLine || !vLine || !origin) return;
    await calibrate(sessionId, {
      horizontal_line: hLine as [number, number, number, number],
      vertical_line: vLine as [number, number, number, number],
      origin,
      time_span_seconds: timeSpan,
      pressure_span_mmhg: pressureSpan,
    });
    onCalibrated();
  };

  const scale = Math.min(1, 900 / dims.w);

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Step: {step === "horizontal" && "Click two points on horizontal time span"}
        {step === "vertical" && "Click two points on vertical pressure span"}
        {step === "origin" && "Click origin (x-axis start, y=0)"}
        {step === "done" && "Enter spans and apply calibration"}
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>Time span (s)</Label>
          <Input type="number" value={timeSpan} onChange={(e) => setTimeSpan(Number(e.target.value))} />
        </div>
        <div>
          <Label>Pressure span (mmHg)</Label>
          <Input type="number" value={pressureSpan} onChange={(e) => setPressureSpan(Number(e.target.value))} />
        </div>
      </div>
      {bg && (
        <Stage
          width={dims.w * scale}
          height={dims.h * scale}
          scaleX={scale}
          scaleY={scale}
          onClick={(e) => {
            const pos = e.target.getStage()?.getPointerPosition();
            if (pos) onStageClick(pos.x / scale, pos.y / scale);
          }}
          className="border border-border rounded"
        >
          <Layer>
            <KonvaImage image={bg} width={dims.w} height={dims.h} />
            {hLine && <Line points={hLine} stroke="#3b82f6" strokeWidth={2} />}
            {vLine && <Line points={vLine} stroke="#ef4444" strokeWidth={2} />}
            {origin && <Circle x={origin[0]} y={origin[1]} radius={6} fill="#22c55e" />}
          </Layer>
        </Stage>
      )}
      <Button onClick={() => void submit()} disabled={step !== "done"}>
        Apply calibration
      </Button>
    </div>
  );
}
