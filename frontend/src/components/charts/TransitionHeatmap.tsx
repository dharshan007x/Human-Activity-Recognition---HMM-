import React from 'react';

interface TransitionHeatmapProps {
  transitionMatrix: number[][]; // [3, 3]
  pi?: number[];
  stateNames?: string[];
}

export const TransitionHeatmap: React.FC<TransitionHeatmapProps> = ({
  transitionMatrix,
  pi = [0.333, 0.333, 0.334],
  stateNames = ['Walking', 'Sitting', 'Running'],
}) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--surface-card)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px',
        maxWidth: '520px',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '14px',
        }}
      >
        <span
          style={{
            fontSize: '0.92rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-serif)',
          }}
        >
          Transition Matrix A &amp; Priors &pi;
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Row Stochastic (&sum;<sub>j</sub> A<sub>ij</sub> = 1.0)
        </span>
      </div>

      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
        {/* Initial Prior Pi Column */}
        <div>
          <div
            style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              color: 'var(--text-secondary)',
              textAlign: 'center',
              marginBottom: '6px',
            }}
          >
            &pi;
          </div>
          {pi.map((prior, idx) => (
            <div
              key={idx}
              style={{
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-sand)',
                borderRadius: 'var(--radius-xs)',
                padding: '10px 8px',
                textAlign: 'center',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.82rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginBottom: '6px',
                minWidth: '55px',
              }}
              title={`Initial State Prior: ${stateNames[idx]}`}
            >
              {prior.toFixed(3)}
            </div>
          ))}
        </div>

        {/* Transition Matrix Grid */}
        <div style={{ flex: 1 }}>
          {/* Header Row: Destination states */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '70px repeat(3, 1fr)',
              gap: '6px',
              textAlign: 'center',
              fontSize: '0.74rem',
              fontWeight: 700,
              color: 'var(--text-secondary)',
              marginBottom: '6px',
            }}
          >
            <div></div>
            {stateNames.map((name) => (
              <div key={name} style={{ textTransform: 'uppercase' }}>
                &rarr; {name}
              </div>
            ))}
          </div>

          {/* Body Rows: Source states */}
          {stateNames.map((fromState, rIdx) => (
            <div
              key={fromState}
              style={{
                display: 'grid',
                gridTemplateColumns: '70px repeat(3, 1fr)',
                gap: '6px',
                marginBottom: '6px',
                alignItems: 'center',
              }}
            >
              <div
                style={{
                  fontSize: '0.74rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  color: 'var(--text-secondary)',
                  textAlign: 'right',
                  paddingRight: '6px',
                }}
              >
                {fromState}
              </div>

              {stateNames.map((_, cIdx) => {
                const prob =
                  transitionMatrix && transitionMatrix[rIdx] ? transitionMatrix[rIdx][cIdx] : 0;
                const isSelf = rIdx === cIdx;

                // Color based on probability weight
                const bg = isSelf
                  ? 'rgba(196, 96, 58, 0.22)'
                  : prob > 0.05
                  ? 'rgba(196, 96, 58, 0.1)'
                  : 'var(--surface-elevated)';

                return (
                  <div
                    key={cIdx}
                    style={{
                      backgroundColor: bg,
                      border: isSelf
                        ? '1.5px solid var(--accent-terracotta)'
                        : '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-xs)',
                      padding: '10px 4px',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <div
                      style={{
                        fontWeight: isSelf ? 700 : 500,
                        fontSize: '0.86rem',
                        color: isSelf ? 'var(--accent-terracotta)' : 'var(--text-primary)',
                      }}
                    >
                      {prob.toFixed(4)}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
