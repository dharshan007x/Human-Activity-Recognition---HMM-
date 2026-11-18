import React, { useEffect, useState } from 'react';
import { Badge } from '../components/common/Badge';
import { Slider } from '../components/common/Slider';
import { MetricCard } from '../components/common/MetricCard';
import { TrellisChart } from '../components/charts/TrellisChart';
import { CalculationPanel } from '../components/common/CalculationPanel';
import { api, SampleSequenceItem, BackwardResult, ValidationResult } from '../services/api';

export const BackwardPage: React.FC = () => {
  const [samples, setSamples] = useState<SampleSequenceItem[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [backwardResult, setBackwardResult] = useState<BackwardResult | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
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

    Promise.all([
      api.runBackward(sample.raw_sequence, true),
      api.validateLikelihoods(sample.raw_sequence, true),
    ])
      .then(([bwdRes, valRes]) => {
        setBackwardResult(bwdRes);
        setValidation(valRes);
        const steps = bwdRes.scaled_beta_table?.length ?? 50;
        setActiveStep(Math.max(steps - 1, 0));
      })
      .catch((err) => console.error('Error running backward algorithm:', err));
  }, [selectedSampleId, samples]);

  const currentSample = samples.find((s) => s.id === selectedSampleId);
  const totalSteps = backwardResult?.scaled_beta_table?.length ?? 50;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Backward Algorithm (Smoothing &amp; Dual Verification)
          </h1>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Computes backward variables &beta;<sub>t</sub>(i) = P(O<sub>t+1</sub>, ..., O<sub>T</sub> | q<sub>t</sub> = S<sub>i</sub>, &lambda;)
            recursively backwards from t = T - 1 down to t = 0.
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

      {/* Metrics & Dual Identity Validation */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label="Backward Log-Likelihood"
          value={backwardResult?.log_likelihood != null ? backwardResult.log_likelihood.toFixed(2) : '-'}
          subtitle="ln P(O | λ) via reverse recursion"
          badge={<Badge variant="accent">ln P_bwd(O|λ)</Badge>}
          highlight
        />
        <MetricCard
          label="Dual Likelihood Agreement"
          value={
            validation?.abs_difference_fwd_bwd != null
              ? `|Δ| = ${validation.abs_difference_fwd_bwd.toExponential(2)}`
              : '-'
          }
          subtitle="Strict validation |ln P_fwd - ln P_bwd| < 10⁻⁵"
          badge={
            <Badge variant={validation?.tolerance_passed ? 'walking' : 'sitting'}>
              {validation?.tolerance_passed ? '100% Identical' : 'Checking'}
            </Badge>
          }
        />
        <MetricCard
          label="Termination Condition"
          value="β_T(i) ≡ 1.0"
          subtitle="Initial boundary across all states S"
          badge={<Badge variant="neutral">t = T-1</Badge>}
        />
        <MetricCard
          label="Current Activity"
          value={currentSample?.activity_name || 'Walking'}
          subtitle={`Subject ${currentSample?.subject_id || 101}`}
          badge={<Badge variant="walking">Sample</Badge>}
        />
      </div>

      {/* Time Step Scrubber */}
      {backwardResult && totalSteps > 0 && (
        <Slider
          min={0}
          max={totalSteps - 1}
          value={activeStep}
          onChange={setActiveStep}
          label="Interactive Backward Trellis Scrubber (Reversed Time Direction)"
          unit="t"
          fps={10}
        />
      )}

      {/* Backward Trellis Chart */}
      {backwardResult?.scaled_beta_table && (
        <TrellisChart
          probabilities={backwardResult.scaled_beta_table}
          stateNames={backwardResult.state_names}
          activeStep={activeStep}
          onSelectStep={setActiveStep}
          height={240}
          label="Backward Probability Lattice Trellis β̂_t(i)"
        />
      )}

      {/* Step-by-Step Calculation Panel */}
      {backwardResult?.scaled_beta_table && (
        <CalculationPanel
          step={activeStep}
          stateNames={backwardResult.state_names}
          algorithm="backward"
          betaScaled={backwardResult.scaled_beta_table}
          logLikelihood={backwardResult.log_likelihood}
        />
      )}
    </div>
  );
};
