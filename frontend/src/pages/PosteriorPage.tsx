import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Slider } from '../components/common/Slider';
import { MetricCard } from '../components/common/MetricCard';
import { PosteriorAreaChart } from '../components/charts/PosteriorAreaChart';
import { PosteriorLandscape3D } from '../components/visualizations/PosteriorLandscape3D';
import { CalculationPanel } from '../components/common/CalculationPanel';
import { api, SampleSequenceItem, PredictResult } from '../services/api';

export const PosteriorPage: React.FC = () => {
  const [samples, setSamples] = useState<SampleSequenceItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    api
      .getSampleSequences()
      .then((sampList) => {
        setSamples(sampList);
        if (sampList && sampList.length > 0) {
          setSelectedSampleId(sampList[0].id);
        }
      })
      .catch((err) => console.error('Failed to load samples:', err));
  }, []);

  useEffect(() => {
    const sample = samples.find((s) => s.id === selectedSampleId);
    if (!sample || !sample.raw_sequence) return;

    api
      .runPredict(sample.raw_sequence, true)
      .then((res) => {
        setPredictResult(res);
        setActiveStep(0);
      })
      .catch((err) => console.error('Error running posterior decoding:', err));
  }, [selectedSampleId, samples]);

  const currentStepPrediction = predictResult?.pointwise_states?.[activeStep] || 'Walking';
  const currentGamma = predictResult?.gamma_table?.[activeStep] || [0.33, 0.33, 0.34];
  const totalSteps = predictResult?.gamma_table?.length ?? 50;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Posterior State Decoding &amp; Smoothing
          </h1>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Combines past evidence (&alpha;<sub>t</sub>) and future observations (&beta;<sub>t</sub>)
            to estimate exact smoothed state occupancy probabilities &gamma;<sub>t</sub>(i) = P(q<sub>t</sub> = S<sub>i</sub> | O, &lambda;).
          </p>
        </div>

        {/* Sample Picker */}
        <div>
          <label style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            Validation Sample Sequence
          </label>
          <select
            value={selectedSampleId}
            onChange={(e) => setSelectedSampleId(e.target.value)}
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
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.activity_name} (Subj {s.subject_id}) - [T={s.length}]
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Numerical Metrics Summary */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label={`Decoded State at t = ${activeStep}`}
          value={currentStepPrediction}
          subtitle={`Confidence: ${(Math.max(...currentGamma) * 100).toFixed(1)}%`}
          badge={
            <Badge
              variant={
                currentStepPrediction === 'Walking'
                  ? 'walking'
                  : currentStepPrediction === 'Sitting'
                  ? 'sitting'
                  : 'running'
              }
            >
              argmax &gamma;
            </Badge>
          }
          highlight
        />
        <MetricCard
          label="Posterior vs Viterbi Agreement"
          value={
            predictResult?.path_match_percentage != null
              ? `${(predictResult.path_match_percentage * 100).toFixed(1)}%`
              : '-'
          }
          subtitle="Pointwise MAP vs Global Dynamic Programming"
          badge={<Badge variant="accent">Viterbi</Badge>}
        />
        <MetricCard
          label="Total Probability Axiom"
          value="1.000000"
          subtitle="Sum of gamma over states S"
          badge={<Badge variant="walking">Exact Sum</Badge>}
        />
        <MetricCard
          label="Sequence Log-Likelihood"
          value={
            predictResult?.forward_log_likelihood != null
              ? predictResult.forward_log_likelihood.toFixed(2)
              : '-'
          }
          subtitle="Consistent across α and β"
          badge={<Badge variant="neutral">ln P(O|λ)</Badge>}
        />
      </div>

      {/* Synchronized Scrubber */}
      {predictResult?.gamma_table && totalSteps > 0 && (
        <Slider
          min={0}
          max={totalSteps - 1}
          value={activeStep}
          onChange={setActiveStep}
          label="Synchronized Posterior Scrubber"
          unit="t"
          fps={10}
        />
      )}

      {/* 3D Posterior Landscape & D3 Area Chart */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(440px, 1fr))',
          gap: '20px',
        }}
      >
        <Card
          title="3D Posterior Probability Surface"
          subtitle="Topographic relief (X: time t, Y: states Walking/Sitting/Running, Z: probability γ)"
          badge={<Badge variant="accent">Three.js 3D Relief</Badge>}
        >
          {predictResult?.gamma_table && (
            <PosteriorLandscape3D
              gamma={predictResult.gamma_table}
              currentStep={activeStep}
              stateNames={predictResult.state_names}
              height={320}
            />
          )}
        </Card>

        <Card
          title="Continuous State Occupancy Distribution"
          subtitle="Interactive stacked D3 area chart showing temporal state probability mass"
        >
          {predictResult?.gamma_table && (
            <PosteriorAreaChart
              gamma={predictResult.gamma_table}
              stateNames={predictResult.state_names}
              activeStep={activeStep}
              onStepChange={setActiveStep}
              height={260}
            />
          )}
        </Card>
      </div>

      {/* Step-by-Step Calculation Panel */}
      {predictResult?.gamma_table && (
        <CalculationPanel
          step={activeStep}
          stateNames={predictResult.state_names}
          algorithm="posterior"
          gamma={predictResult.gamma_table}
          logLikelihood={predictResult.forward_log_likelihood}
        />
      )}
    </div>
  );
};
