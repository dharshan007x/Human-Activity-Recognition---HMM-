import React, { useEffect, useState } from 'react';
import { SlidersHorizontal, Check, Zap, Layers, RefreshCw } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Callout } from '../components/common/Callout';
import { MathBlock } from '../components/common/MathBlock';
import { BodePlot } from '../components/charts/BodePlot';
import { api, FilterResponse } from '../services/api';

export const PreprocessingPage: React.FC = () => {
  const [filterData, setFilterData] = useState<FilterResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getFilterResponse()
      .then((data) => setFilterData(data))
      .catch((err) => console.error('Failed to load filter response:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Signal Preprocessing &amp; Feature Pipeline
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Deterministic, zero-leakage signal conditioning pipeline transforming raw ±16g IMU telemetry
          into calibrated observation vectors O(t) &isin; &reals;⁴.
        </p>
      </div>

      {/* Interactive Bode Plot */}
      <Card
        title="Anti-Aliasing Filter Frequency Response"
        subtitle="4th-order zero-phase Butterworth filter with fc = 10 Hz cutoff"
        badge={<Badge variant="accent">Scipy Signal</Badge>}
      >
        {filterData ? (
          <BodePlot
            frequencies={filterData.frequencies}
            magnitudeDb={filterData.magnitude_db || filterData.gains_db || []}
            cutoffHz={filterData.cutoff_hz}
            height={260}
          />
        ) : (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading filter frequency response...
          </div>
        )}

        <div
          style={{
            marginTop: '16px',
            backgroundColor: 'var(--bg-secondary)',
            padding: '12px 16px',
            borderRadius: 'var(--radius-xs)',
            fontSize: '0.84rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.5,
          }}
        >
          <strong>Why fc = 10 Hz?</strong> Human locomotive biomechanics (gait cycle, heel strikes, arm swings)
          concentrate spectral energy below 5–8 Hz. A 10 Hz low-pass filter preserves all locomotor harmonics while
          attenuating high-frequency mechanical vibration noise and preventing aliasing when decimating from 100 Hz
          to 25 Hz (where Nyquist frequency is 12.5 Hz).
        </div>
      </Card>

      {/* 5-Stage Pipeline Overview */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Deterministic 5-Stage Conditioning Sequence
        </h2>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '16px',
          }}
        >
          {/* Stage 1 */}
          <div
            style={{
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-sand)',
              borderRadius: 'var(--radius-sm)',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-terracotta)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                }}
              >
                1
              </span>
              <strong style={{ fontSize: '0.92rem' }}>Sanitization &amp; Bounds</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Check physical sensor limits (|a| &le; 16g). Filter dropped packets and NaN readings.
              Apply intra-subject linear interpolation on isolated missing ticks. Discard transient code 0.
            </p>
          </div>

          {/* Stage 2 */}
          <div
            style={{
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-sand)',
              borderRadius: 'var(--radius-sm)',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-terracotta)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                }}
              >
                2
              </span>
              <strong style={{ fontSize: '0.92rem' }}>Zero-Phase Butterworth</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Apply 4th-order forward-backward IIR filter (scipy <code>filtfilt</code>) to ensure exactly
              zero phase distortion across acceleration waveforms.
            </p>
          </div>

          {/* Stage 3 */}
          <div
            style={{
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-sand)',
              borderRadius: 'var(--radius-sm)',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-terracotta)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                }}
              >
                3
              </span>
              <strong style={{ fontSize: '0.92rem' }}>Decimation (M = 4)</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Subsample sequence by factor M = 4, downsampling from 100 Hz to target 25 Hz.
              Reduces computational complexity of Forward-Backward recursion by 16&times;.
            </p>
          </div>

          {/* Stage 4 */}
          <div
            style={{
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-sand)',
              borderRadius: 'var(--radius-sm)',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-terracotta)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                }}
              >
                4
              </span>
              <strong style={{ fontSize: '0.92rem' }}>Feature Synthesis</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Synthesize Euclidean magnitude <MathBlock math="a_{\text{mag}} = \sqrt{a_x^2 + a_y^2 + a_z^2}" />
              yielding rotation-invariant observation vectors <MathBlock math="O_t = [a_x, a_y, a_z, a_{\text{mag}}] \in \mathbb{R}^4" />.
            </p>
          </div>

          {/* Stage 5 */}
          <div
            style={{
              backgroundColor: 'var(--surface-card)',
              border: '1px solid var(--border-sand)',
              borderRadius: 'var(--radius-sm)',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span
                style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-terracotta)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                }}
              >
                5
              </span>
              <strong style={{ fontSize: '0.92rem' }}>Window Segmentation</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              Slice continuous streams into non-overlapping temporal segments of T = 50 time steps
              (2.0 seconds at 25 Hz), capturing at least 2 full locomotor gait cycles.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
