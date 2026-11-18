import React, { useState } from 'react';
import { MathBlock } from './MathBlock';
import { Tabs } from './Tabs';

interface CalculationPanelProps {
  step: number;
  stateNames: string[];
  states?: string[];
  algorithm: 'forward' | 'backward' | 'posterior';
  alphaScaled?: number[][];
  betaScaled?: number[][];
  gamma?: number[][];
  scalingFactors?: number[];
  transitionMatrix?: number[][];
  logLikelihood?: number;
}

export const CalculationPanel: React.FC<CalculationPanelProps> = ({
  step,
  stateNames = ['Walking', 'Sitting', 'Running'],
  algorithm,
  alphaScaled,
  betaScaled,
  gamma,
  scalingFactors,
  transitionMatrix,
  logLikelihood,
}) => {
  const [selectedStateIdx, setSelectedStateIdx] = useState(0);

  const t = Math.max(0, step);
  const stateName = stateNames[selectedStateIdx] || 'Walking';

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1.5px solid var(--border-dark)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px 20px',
        marginTop: '16px',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--border-sand)',
          paddingBottom: '10px',
          marginBottom: '14px',
        }}
      >
        <span
          style={{
            fontWeight: 700,
            fontSize: '0.95rem',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-serif)',
          }}
        >
          Exact Step-by-Step Numerical Arithmetic (t = {t}, State = {stateName})
        </span>

        <Tabs
          tabs={stateNames.map((name, idx) => ({ id: String(idx), label: name }))}
          activeTab={String(selectedStateIdx)}
          onChange={(id) => setSelectedStateIdx(Number(id))}
        />
      </div>

      {algorithm === 'forward' && (
        <div>
          <div style={{ marginBottom: '12px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Theoretical Induction Step:
            </span>
            <MathBlock
              block
              math={`\\alpha_t(j) = \\left[ \\sum_{i=1}^N \\hat{\\alpha}_{t-1}(i) \\cdot A_{ij} \\right] \\cdot b_j(O_t), \\quad \\hat{\\alpha}_t(j) = \\frac{\\alpha_t(j)}{c_t}`}
            />
          </div>

          {t === 0 ? (
            <div style={{ fontSize: '0.88rem', lineHeight: 1.6 }}>
              <p>
                <strong>Initialisation step (t = 0):</strong>
              </p>
              <div
                style={{
                  fontFamily: 'var(--font-mono)',
                  backgroundColor: 'var(--surface-card)',
                  padding: '10px',
                  borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--border-light)',
                  marginTop: '6px',
                }}
              >
                <div>Initial state prior: π({stateName}) ≈ 0.3333</div>
                <div>
                  Scaled forward probability: &alpha;̂<sub>0</sub>({stateName}) ={' '}
                  {alphaScaled?.[0]?.[selectedStateIdx] != null
                    ? alphaScaled[0][selectedStateIdx].toFixed(6)
                    : 'N/A'}
                </div>
                <div>
                  Scaling coefficient: c<sub>0</sub> ={' '}
                  {scalingFactors?.[0] != null ? scalingFactors[0].toFixed(6) : 'N/A'}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.88rem', lineHeight: 1.6 }}>
              <p>
                <strong>1. Sum over all incoming transitions from t - 1:</strong>
              </p>
              <div
                style={{
                  backgroundColor: 'var(--surface-card)',
                  padding: '12px',
                  borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--border-light)',
                  marginTop: '6px',
                  marginBottom: '10px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.82rem',
                }}
              >
                {stateNames.map((fromState, i) => {
                  const prevAlpha = alphaScaled?.[t - 1]?.[i] ?? 0;
                  const trans = transitionMatrix?.[i]?.[selectedStateIdx] ?? 0.3333;
                  const prod = prevAlpha * trans;
                  return (
                    <div key={i} style={{ marginBottom: '4px' }}>
                      • From {fromState}: &alpha;̂<sub>{t - 1}</sub>({fromState}) &times; A(
                      {fromState} &rarr; {stateName}) = {prevAlpha.toFixed(5)} &times;{' '}
                      {trans.toFixed(5)} = <strong>{prod.toExponential(4)}</strong>
                    </div>
                  );
                })}
              </div>

              <div
                style={{
                  display: 'flex',
                  gap: '12px',
                  backgroundColor: 'var(--surface-card)',
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--border-light)',
                  alignItems: 'center',
                }}
              >
                <div>
                  Scaled value &alpha;̂<sub>{t}</sub>({stateName}) ={' '}
                  <strong style={{ color: 'var(--accent-terracotta)' }}>
                    {alphaScaled?.[t]?.[selectedStateIdx] != null
                      ? alphaScaled[t][selectedStateIdx].toFixed(6)
                      : '0.000000'}
                  </strong>
                </div>
                <div>|</div>
                <div>
                  Scale factor c<sub>{t}</sub> ={' '}
                  <strong>
                    {scalingFactors?.[t] != null
                      ? scalingFactors[t].toFixed(6)
                      : '0.000000'}
                  </strong>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {algorithm === 'backward' && (
        <div>
          <div style={{ marginBottom: '12px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Theoretical Induction Step:
            </span>
            <MathBlock
              block
              math={`\\beta_t(i) = \\sum_{j=1}^N A_{ij} \\cdot b_j(O_{t+1}) \\cdot \\beta_{t+1}(j), \\quad \\hat{\\beta}_t(i) = \\frac{\\beta_t(i)}{c_{t+1}}`}
            />
          </div>

          <div style={{ fontSize: '0.88rem', lineHeight: 1.6 }}>
            <div
              style={{
                backgroundColor: 'var(--surface-card)',
                padding: '12px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-light)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.82rem',
              }}
            >
              <div>Current Time Step t = {t}</div>
              <div>State = {stateName}</div>
              <div style={{ marginTop: '8px' }}>
                Scaled backward variable &beta;̂<sub>{t}</sub>({stateName}) ={' '}
                <strong style={{ color: 'var(--accent-teal)' }}>
                  {betaScaled?.[t]?.[selectedStateIdx] != null
                    ? betaScaled[t][selectedStateIdx].toFixed(6)
                    : '1.000000'}
                </strong>
              </div>
              <div style={{ marginTop: '4px', color: 'var(--text-muted)' }}>
                {t === (betaScaled ? betaScaled.length - 1 : 0)
                  ? 'Boundary Condition at t = T - 1: Scaled beta initialized to 1.0'
                  : 'Computed recursively backward from t + 1 observations'}
              </div>
            </div>
          </div>
        </div>
      )}

      {algorithm === 'posterior' && (
        <div>
          <div style={{ marginBottom: '12px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Posterior Smoothing Formula:
            </span>
            <MathBlock
              block
              math={`\\gamma_t(i) = P(q_t = S_i \\mid O, \\lambda) = \\frac{\\hat{\\alpha}_t(i) \\hat{\\beta}_t(i)}{\\sum_{j=1}^N \\hat{\\alpha}_t(j) \\hat{\\beta}_t(j)}`}
            />
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '10px',
              marginTop: '10px',
            }}
          >
            {stateNames.map((name, i) => {
              const a = alphaScaled?.[t]?.[i] ?? 0;
              const b = betaScaled?.[t]?.[i] ?? 0;
              const g = gamma?.[t]?.[i] ?? 0;
              return (
                <div
                  key={i}
                  style={{
                    backgroundColor:
                      i === selectedStateIdx ? 'var(--surface-elevated)' : 'var(--surface-card)',
                    border:
                      i === selectedStateIdx
                        ? '1.5px solid var(--accent-terracotta)'
                        : '1px solid var(--border-light)',
                    padding: '10px',
                    borderRadius: 'var(--radius-xs)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                    {name}
                  </div>
                  <div>&alpha;̂<sub>{t}</sub>: {a.toFixed(5)}</div>
                  <div>&beta;̂<sub>{t}</sub>: {b.toFixed(5)}</div>
                  <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px dashed var(--border-sand)' }}>
                    &gamma;<sub>{t}</sub>: <strong style={{ color: 'var(--accent-terracotta)', fontSize: '0.95rem' }}>{g.toFixed(6)}</strong>
                  </div>
                </div>
              );
            })}
          </div>

          <div
            style={{
              marginTop: '12px',
              padding: '8px 12px',
              backgroundColor: 'var(--surface-card)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--border-light)',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.82rem',
              color: 'var(--text-secondary)',
            }}
          >
            <span>
              Total Probability Axiom Check: &sum;<sub>i</sub> &gamma;<sub>{t}</sub>(i) ={' '}
              <strong>
                {gamma && gamma[t]
                  ? gamma[t].reduce((sum, v) => sum + v, 0).toFixed(6)
                  : '1.000000'}
              </strong>
            </span>
            {logLikelihood !== undefined && (
              <span>Sequence Log-Likelihood ln P(O|&lambda;): <strong>{logLikelihood.toFixed(2)}</strong></span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
