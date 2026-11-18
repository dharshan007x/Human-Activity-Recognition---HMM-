import React, { useEffect, useRef, useState } from 'react';

interface SignalCanvasProps {
  timeStamps?: number[];
  accX: number[];
  accY: number[];
  accZ: number[];
  accMag?: number[];
  activeStep?: number;
  onStepChange?: (step: number) => void;
  height?: number;
  title?: string;
}

export const SignalCanvas: React.FC<SignalCanvasProps> = ({
  accX,
  accY,
  accZ,
  accMag,
  activeStep = 0,
  onStepChange,
  height = 220,
  title = 'Triaxial Accelerometer Waveforms (m/s²)',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [visibleChannels, setVisibleChannels] = useState({
    ax: true,
    ay: true,
    az: true,
    mag: true,
  });

  const N = accX.length;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || N === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth || 600;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.scale(dpr, dpr);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 25;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    // Clear background
    ctx.fillStyle = '#FFFCF7'; // surface-card
    ctx.fillRect(0, 0, width, height);

    // Compute min and max across visible channels
    let minVal = -20;
    let maxVal = 20;
    const allVals: number[] = [];
    if (visibleChannels.ax) allVals.push(...accX);
    if (visibleChannels.ay) allVals.push(...accY);
    if (visibleChannels.az) allVals.push(...accZ);
    if (visibleChannels.mag && accMag) allVals.push(...accMag);

    if (allVals.length > 0) {
      const min = Math.min(...allVals);
      const max = Math.max(...allVals);
      const span = Math.max(max - min, 5);
      minVal = Math.floor(min - span * 0.1);
      maxVal = Math.ceil(max + span * 0.1);
    }

    const valToY = (val: number) => {
      const frac = (val - minVal) / (maxVal - minVal);
      return padTop + plotHeight * (1 - frac);
    };

    const idxToX = (idx: number) => {
      return padLeft + (idx / Math.max(N - 1, 1)) * plotWidth;
    };

    // Draw horizontal grid lines & Y labels
    ctx.strokeStyle = '#EFE4D2';
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#9C8C7E';
    ctx.textAlign = 'right';

    const yTicks = [minVal, minVal / 2 + maxVal / 2, maxVal];
    yTicks.forEach((tick) => {
      const y = valToY(tick);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${tick.toFixed(1)}`, padLeft - 6, y + 3);
    });

    // Zero-line
    if (minVal < 0 && maxVal > 0) {
      const zeroY = valToY(0);
      ctx.strokeStyle = '#D1C0A5';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(padLeft, zeroY);
      ctx.lineTo(width - padRight, zeroY);
      ctx.stroke();
    }

    // Helper to draw a channel curve
    const drawChannel = (data: number[], color: string, strokeWidth = 1.8) => {
      if (data.length === 0) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = strokeWidth;
      ctx.lineJoin = 'round';
      ctx.moveTo(idxToX(0), valToY(data[0]));
      for (let i = 1; i < data.length; i++) {
        ctx.lineTo(idxToX(i), valToY(data[i]));
      }
      ctx.stroke();
    };

    // Draw channels
    if (visibleChannels.ax) drawChannel(accX, '#CC3333'); // Red X
    if (visibleChannels.ay) drawChannel(accY, '#5B8266'); // Sage Y
    if (visibleChannels.az) drawChannel(accZ, '#366D67'); // Teal Z
    if (visibleChannels.mag && accMag) drawChannel(accMag, '#C4603A', 2.2); // Terracotta Mag

    // Active Step cursor
    if (activeStep >= 0 && activeStep < N) {
      const cursorX = idxToX(activeStep);
      ctx.strokeStyle = '#2E2118';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cursorX, padTop - 5);
      ctx.lineTo(cursorX, height - padBottom + 5);
      ctx.stroke();
      ctx.setLineDash([]);

      // Little indicator pill
      ctx.fillStyle = '#2E2118';
      ctx.fillRect(cursorX - 16, padTop - 12, 32, 14);
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`t=${activeStep}`, cursorX, padTop - 2);
    }

    // Bottom border
    ctx.strokeStyle = '#E2D3BC';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, height - padBottom);
    ctx.lineTo(width - padRight, height - padBottom);
    ctx.stroke();

    // X Axis ticks
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6B5B4E';
    const xStep = Math.max(Math.floor(N / 8), 1);
    for (let i = 0; i < N; i += xStep) {
      const x = idxToX(i);
      ctx.fillText(`${i}`, x, height - padBottom + 14);
    }
  }, [accX, accY, accZ, accMag, activeStep, height, N, visibleChannels]);

  // Click & Drag on Canvas to Scrub
  const handleInteraction = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !onStepChange || N === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padLeft = 45;
    const padRight = 20;
    const plotWidth = rect.width - padLeft - padRight;

    const relX = Math.max(0, Math.min(x - padLeft, plotWidth));
    const stepIdx = Math.round((relX / plotWidth) * (N - 1));
    onStepChange(stepIdx);
  };

  return (
    <div
      ref={containerRef}
      style={{
        backgroundColor: 'var(--surface-card)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        padding: '14px 16px',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '10px',
        }}
      >
        <span
          style={{
            fontSize: '0.88rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-serif)',
          }}
        >
          {title}
        </span>

        {/* Channel Toggles */}
        <div style={{ display: 'flex', gap: '12px', fontSize: '0.76rem', fontWeight: 600 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', color: '#CC3333' }}>
            <input
              type="checkbox"
              checked={visibleChannels.ax}
              onChange={(e) => setVisibleChannels({ ...visibleChannels, ax: e.target.checked })}
              style={{ accentColor: '#CC3333' }}
            />
            a<sub>x</sub>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', color: '#5B8266' }}>
            <input
              type="checkbox"
              checked={visibleChannels.ay}
              onChange={(e) => setVisibleChannels({ ...visibleChannels, ay: e.target.checked })}
              style={{ accentColor: '#5B8266' }}
            />
            a<sub>y</sub>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', color: '#366D67' }}>
            <input
              type="checkbox"
              checked={visibleChannels.az}
              onChange={(e) => setVisibleChannels({ ...visibleChannels, az: e.target.checked })}
              style={{ accentColor: '#366D67' }}
            />
            a<sub>z</sub>
          </label>
          {accMag && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', color: '#C4603A' }}>
              <input
                type="checkbox"
                checked={visibleChannels.mag}
                onChange={(e) => setVisibleChannels({ ...visibleChannels, mag: e.target.checked })}
                style={{ accentColor: '#C4603A' }}
              />
              ||a|| Mag
            </label>
          )}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        onClick={handleInteraction}
        onMouseMove={(e) => {
          if (e.buttons === 1) handleInteraction(e);
        }}
        style={{ cursor: 'crosshair', display: 'block' }}
      />
    </div>
  );
};
