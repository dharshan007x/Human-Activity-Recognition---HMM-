import React, { useEffect, useState } from 'react';
import { Badge } from '../components/common/Badge';
import { Slider } from '../components/common/Slider';
import { MetricCard } from '../components/common/MetricCard';
import { TrellisChart } from '../components/charts/TrellisChart';
import { CalculationPanel } from '../components/common/CalculationPanel';
import { api, SampleSequenceItem, ForwardResult, HMMParameters } from '../services/api';

export const ForwardPage: React.FC = () => {
  const [samples, setSamples] = useState<SampleSequenceItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [forwardResult, setForwardResult] = useState<ForwardResult | null>(null);
  const [params, setParams] = useState<HMMParameters | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    Promise.all([api.getSampleSequences(), api.getModelParameters()])
      .then(([sampList, hmmParams]) => {
        setSamples(sampList);
        setParams(hmmParams);
        if (sampList && sampList.length > 0) {
          setSelectedSampleId(sampList[0].id);
        }
      })
      .catch((err) => console.error('Failed to load initial data:', err));
  }, []);

  useEffect(() => {
    const sample = samples.find((s) => s.id === selectedSampleId);
    if (!sample || !sample.raw_sequence) return;

    api
      .runForward(sample.raw_sequence, true)
      .then((res) => {
        setForwardResult(res);
        setActiveStep(0);
      })
      .catch((err) => console.error('Error running forward algorithm:', err));
  }, [selectedSampleId, samples]);

  const currentSample = samples.find((s) => s.id === selectedSampleId);
  const totalSteps = forwardResult?.scaled_alpha_table?.length ?? 50;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Forward Algorithm (Filtering &amp; Likelihood)
          </h1>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Calculates forward probability variables &alpha;<sub>t</sub>(i) = P(O₁, ..., Oₜ, qₜ = Sᵢ | &lambda;)
            and overall sequence log-likelihood ln P(O | &lambda;) via dynamic programming.
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

      {/* Numerical Results Summary */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label="Sequence Log-Likelihood"
          value={forwardResult?.log_likelihood != null ? forwardResult.log_likelihood.toFixed(2) : '-'}
          subtitle="ln P(O | λ) via log-sum-exp"
          badge={<Badge variant="accent">ln P(O|λ)</Badge>}
          highlight
        />
        <MetricCard
          label="Sequence Length T"
          value={`${totalSteps} steps`}
          subtitle={`${(totalSteps / 25).toFixed(2)}s observation window`}
          badge={<Badge variant="neutral">25 Hz</Badge>}
        />
        <MetricCard
          label="True Activity"
          value={currentSample?.activity_name || 'Walking'}
          subtitle={`Participant ${currentSample?.subject_id || 101}`}
          badge={
            <Badge
              variant={
                currentSample?.activity_name === 'Walking'
                  ? 'walking'
                  : currentSample?.activity_name === 'Sitting'
                  ? 'sitting'
                  : 'running'
              }
            >
              Ground Truth
            </Badge>
          }
        />
        <MetricCard
          label="Numerical Underflow Defense"
          value="Scaled + LogSpace"
          subtitle="Dual validation |Δln P| < 10⁻⁵"
          badge={<Badge variant="walking">Rabiner (1989)</Badge>}
        />
      </div>

      {/* Time Step Scrubber */}
      {forwardResult && totalSteps > 0 && (
        <Slider
          min={0}
          max={totalSteps - 1}
          value={activeStep}
          onChange={setActiveStep}
          label="Interactive Forward Trellis Scrubber"
          unit="t"
          fps={10}
        />
      )}

      {/* Interactive Trellis Diagram */}
      {forwardResult?.scaled_alpha_table && (
        <TrellisChart
          probabilities={forwardResult.scaled_alpha_table}
          stateNames={forwardResult.state_names}
          activeStep={activeStep}
          onSelectStep={setActiveStep}
          height={240}
          label="Forward Probability Lattice Trellis α̂_t(i)"
        />
      )}

      {/* Step-by-Step Arithmetic Calculation Expander */}
      {forwardResult?.scaled_alpha_table && params && (
        <CalculationPanel
          step={activeStep}
          stateNames={forwardResult.state_names}
          algorithm="forward"
          alphaScaled={forwardResult.scaled_alpha_table}
          scalingFactors={forwardResult.scaling_factors}
          transitionMatrix={params.A}
          logLikelihood={forwardResult.log_likelihood}
        />
      )}
    </div>
  );
};
