import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Slider } from '../components/common/Slider';
import { Accelerometer3D } from '../components/visualizations/Accelerometer3D';
import { ActivityTrajectory3D } from '../components/visualizations/ActivityTrajectory3D';
import { SignalCanvas } from '../components/charts/SignalCanvas';
import { api, SignalStream } from '../services/api';

export const SensorExplorerPage: React.FC = () => {
  const [subjectId, setSubjectId] = useState(101);
  const [activityName, setActivityName] = useState('Walking');
  const [signal, setSignal] = useState<SignalStream | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    api
      .getDatasetSignals(subjectId, activityName, 150)
      .then((data) => {
        setSignal(data);
        setCurrentStep(0);
      })
      .catch((err) => console.error('Failed to load sensor stream:', err));
  }, [subjectId, activityName]);

  const ax = signal?.channels?.acc_x?.[currentStep] ?? 0.85;
  const ay = signal?.channels?.acc_y?.[currentStep] ?? 9.65;
  const az = signal?.channels?.acc_z?.[currentStep] ?? 1.2;
  const totalPoints = signal?.channels?.acc_x?.length ?? 150;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            3D Sensor &amp; Signal Inspector
          </h1>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Interactive synchronized inspection of triaxial physical accelerometer orientation, 3D trajectory
            phase portraits, and continuous waveforms.
          </p>
        </div>

        {/* Filter Selectors */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
              Subject
            </label>
            <select
              value={subjectId}
              onChange={(e) => setSubjectId(Number(e.target.value))}
              style={{
                display: 'block',
                padding: '6px 12px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
                backgroundColor: 'var(--surface-card)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontWeight: 600,
                fontSize: '0.86rem',
                marginTop: '2px',
              }}
            >
              {[101, 102, 103, 104, 105, 106, 107, 108, 109].map((s) => (
                <option key={s} value={s}>
                  Subject {s} {s >= 108 ? '(Test)' : s === 107 ? '(Val)' : '(Train)'}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
              Activity
            </label>
            <select
              value={activityName}
              onChange={(e) => setActivityName(e.target.value)}
              style={{
                display: 'block',
                padding: '6px 12px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
                backgroundColor: 'var(--surface-card)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontWeight: 600,
                fontSize: '0.86rem',
                marginTop: '2px',
              }}
            >
              {['Walking', 'Sitting', 'Running'].map((act) => (
                <option key={act} value={act}>
                  {act}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Synchronized Scrubber */}
      <Slider
        min={0}
        max={Math.max(totalPoints - 1, 1)}
        value={currentStep}
        onChange={setCurrentStep}
        label={`Synchronized Playback Scrubber (Subject ${subjectId} - ${activityName})`}
        unit="ticks"
        fps={12}
      />

      {/* Dual 3D Visualisations */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '20px',
        }}
      >
        <Card
          title="Dynamic 3D IMU Coordinate Frame"
          subtitle={`Instantaneous vector [${ax.toFixed(2)}, ${ay.toFixed(2)}, ${az.toFixed(2)}] at t = ${currentStep}`}
          badge={<Badge variant="accent">Three.js Accelerometer</Badge>}
        >
          <Accelerometer3D ax={ax} ay={ay} az={az} activityName={activityName} height={340} />
        </Card>

        <Card
          title="3D Sensor-Space Phase Portrait"
          subtitle="Contrasting periodic gait loops (Walking/Running) vs static cluster (Sitting)"
          badge={<Badge variant="neutral">Phase Space</Badge>}
        >
          <ActivityTrajectory3D height={340} />
        </Card>
      </div>

      {/* Waveform Canvas */}
      {signal && signal.channels && (
        <Card
          title="Synchronized Triaxial Signals &amp; Magnitude"
          subtitle={`Duration: ${signal.duration_sec.toFixed(2)}s @ ${signal.sampling_rate_hz} Hz. Drag cursor across canvas to scrub.`}
        >
          <SignalCanvas
            accX={signal.channels.acc_x}
            accY={signal.channels.acc_y}
            accZ={signal.channels.acc_z}
            accMag={signal.channels.acc_mag}
            activeStep={currentStep}
            onStepChange={setCurrentStep}
            height={240}
          />
        </Card>
      )}
    </div>
  );
};
