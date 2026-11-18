import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from './Button';

interface SliderProps {
  min: number;
  max: number;
  value: number;
  onChange: (val: number) => void;
  label?: string;
  step?: number;
  showPlaybackControls?: boolean;
  fps?: number;
  unit?: string;
}

export const Slider: React.FC<SliderProps> = ({
  min,
  max,
  value,
  onChange,
  label,
  step = 1,
  showPlaybackControls = true,
  fps = 10,
  unit = '',
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef<number | null>(null);
  const valueRef = useRef(value);

  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = window.setInterval(() => {
        const next = valueRef.current + step;
        if (next >= max) {
          onChange(max);
          setIsPlaying(false);
        } else {
          onChange(next);
        }
      }, 1000 / fps);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }

    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, max, step, fps, onChange]);

  const handleStepBack = () => {
    onChange(Math.max(min, value - step));
  };

  const handleStepForward = () => {
    onChange(Math.min(max, value + step));
  };

  const handleReset = () => {
    setIsPlaying(false);
    onChange(min);
  };

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        padding: '12px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.85rem',
        }}
      >
        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
          {label || 'Time Step Scrubber'}
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.85rem',
            backgroundColor: 'var(--surface-card)',
            padding: '2px 8px',
            borderRadius: 'var(--radius-xs)',
            border: '1px solid var(--border-light)',
            color: 'var(--accent-terracotta)',
            fontWeight: 600,
          }}
        >
          {value} {unit} / {max} {unit}
        </span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: '100%',
          accentColor: 'var(--accent-terracotta)',
          cursor: 'pointer',
          height: '6px',
          borderRadius: '3px',
        }}
      />

      {showPlaybackControls && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingTop: '4px',
          }}
        >
          <div style={{ display: 'flex', gap: '6px' }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleStepBack}
              disabled={value <= min}
              style={{ padding: '4px 8px' }}
            >
              <ChevronLeft size={14} />
            </Button>
            <Button
              variant={isPlaying ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
              style={{ padding: '4px 12px' }}
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              <span style={{ marginLeft: '4px' }}>{isPlaying ? 'Pause' : 'Play'}</span>
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleStepForward}
              disabled={value >= max}
              style={{ padding: '4px 8px' }}
            >
              <ChevronRight size={14} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              style={{ padding: '4px 8px' }}
            >
              <RotateCcw size={14} />
            </Button>
          </div>

          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Rate: {fps} steps/s
          </span>
        </div>
      )}
    </div>
  );
};
