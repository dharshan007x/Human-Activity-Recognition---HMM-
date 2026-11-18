import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { MathBlock } from '../components/common/MathBlock';
import { HMMNetwork3D } from '../components/visualizations/HMMNetwork3D';
import { TransitionHeatmap } from '../components/charts/TransitionHeatmap';
import { api, HMMParameters } from '../services/api';

export const HMMModelExplorerPage: React.FC = () => {
  const [params, setParams] = useState<HMMParameters | null>(null);

  useEffect(() => {
    api
      .getModelParameters()
      .then((data) => setParams(data))
      .catch((err) => console.error('Failed to load HMM parameters:', err));
  }, []);

  const featureNames = ['acc_x', 'acc_y', 'acc_z', 'acc_mag'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          HMM Architecture &amp; Parameter Matrix
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Maximum Likelihood parameter specification for the first-order Gaussian Hidden Markov Model
          &lambda; = (&pi;, A, B) fitted across training subjects 101–106.
        </p>
      </div>

      {/* 3D State Transition Network & Matrix */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '20px',
        }}
      >
        <Card
          title="3D State Transition Topology"
          subtitle="Interactive directed graph: Node size ~ prior π, Arc thickness ~ transition A_ij"
          badge={<Badge variant="accent">Three.js Graph</Badge>}
        >
          {params?.A && (
            <HMMNetwork3D
              transitionMatrix={params.A}
              pi={params.pi}
              stateNames={params.state_names}
              height={340}
            />
          )}
        </Card>

        <Card
          title="Stochastic Parameter Matrices"
          subtitle="Initial state priors π and row-stochastic transition probabilities A"
        >
          {params?.A && (
            <TransitionHeatmap
              transitionMatrix={params.A}
              pi={params.pi}
              stateNames={params.state_names}
            />
          )}

          <div
            style={{
              marginTop: '16px',
              backgroundColor: 'var(--bg-secondary)',
              padding: '10px 14px',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.8rem',
              color: 'var(--text-secondary)',
            }}
          >
            <strong>Laplace Smoothing:</strong> Transition frequencies were regularized using add-one
            smoothing (&alpha; = 1.0) to eliminate zero-probability absorption traps during inference:
            <MathBlock block math="A_{ij} = \frac{C(S_i \to S_j) + 1.0}{\sum_k [C(S_i \to S_k) + 1.0]}" />
          </div>
        </Card>
      </div>

      {/* Emission Distribution Parameters */}
      <Card
        title="Multivariate Gaussian Emission Parameters B = {b_j(O)}"
        subtitle="State-conditioned mean vectors μ_j and regularized covariance matrices Σ_j in ℝ⁴"
      >
        {params?.emissions ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '18px',
            }}
          >
            {params.emissions.map((em, sIdx) => {
              const stateColor =
                em.state === 'Walking' ? '#5B8266' : em.state === 'Sitting' ? '#C98A2C' : '#C4603A';
              return (
                <div
                  key={em.state}
                  style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1.5px solid ${stateColor}`,
                    borderRadius: 'var(--radius-sm)',
                    padding: '16px',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '12px',
                    }}
                  >
                    <h3 style={{ fontSize: '1.1rem', color: stateColor }}>
                      State {sIdx + 1}: {em.state}
                    </h3>
                    <Badge variant={em.state === 'Walking' ? 'walking' : em.state === 'Sitting' ? 'sitting' : 'running'}>
                      N({em.state})
                    </Badge>
                  </div>

                  {/* Mean Vector */}
                  <div style={{ marginBottom: '14px' }}>
                    <div
                      style={{
                        fontSize: '0.76rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        color: 'var(--text-secondary)',
                        marginBottom: '4px',
                      }}
                    >
                      Mean Vector &mu;<sub>{em.state}</sub> &isin; &reals;⁴
                    </div>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: '4px',
                        textAlign: 'center',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.78rem',
                      }}
                    >
                      {em.mean?.map((val, fIdx) => (
                        <div
                          key={fIdx}
                          style={{
                            backgroundColor: 'var(--surface-card)',
                            padding: '6px 2px',
                            borderRadius: 'var(--radius-xs)',
                            border: '1px solid var(--border-sand)',
                          }}
                        >
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                            {featureNames[fIdx]}
                          </div>
                          <strong style={{ color: 'var(--text-primary)' }}>{val.toFixed(2)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Covariance Diagonal / Full Matrix */}
                  <div>
                    <div
                      style={{
                        fontSize: '0.76rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        color: 'var(--text-secondary)',
                        marginBottom: '4px',
                      }}
                    >
                      Covariance Diagonal diag(&Sigma;<sub>{em.state}</sub>)
                    </div>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: '4px',
                        textAlign: 'center',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.78rem',
                      }}
                    >
                      {em.variance_diagonal?.map((val, fIdx) => (
                        <div
                          key={fIdx}
                          style={{
                            backgroundColor: 'var(--surface-card)',
                            padding: '6px 2px',
                            borderRadius: 'var(--radius-xs)',
                            border: '1px solid var(--border-sand)',
                          }}
                        >
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                            &sigma;²({featureNames[fIdx]})
                          </div>
                          <strong style={{ color: 'var(--accent-teal)' }}>{val.toFixed(3)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading Gaussian parameters...
          </div>
        )}
      </Card>
    </div>
  );
};
