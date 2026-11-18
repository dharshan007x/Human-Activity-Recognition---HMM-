import React from 'react';

interface ConfusionMatrixD3Props {
  classes?: string[];
  normalized: number[][]; // [3, 3]
  raw?: number[][]; // [3, 3]
}

export const ConfusionMatrixD3: React.FC<ConfusionMatrixD3Props> = ({
  classes = ['Walking', 'Sitting', 'Running'],
  normalized,
  raw,
}) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--surface-card)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px',
        maxWidth: '480px',
        margin: '0 auto',
      }}
    >
      <div
        style={{
          fontSize: '0.88rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-serif)',
          textAlign: 'center',
          marginBottom: '14px',
        }}
      >
        Held-Out Test Confusion Matrix (Normalized & Counts)
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {/* Y-axis label */}
        <div
          style={{
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
            fontSize: '0.78rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-secondary)',
            marginRight: '8px',
          }}
        >
          Ground Truth Label
        </div>

        <div>
          {/* X-axis header */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '70px repeat(3, 85px)',
              gap: '6px',
              textAlign: 'center',
              fontSize: '0.74rem',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              marginBottom: '6px',
            }}
          >
            <div></div>
            {classes.map((cls) => (
              <div key={cls} style={{ textTransform: 'uppercase' }}>
                {cls}
              </div>
            ))}
          </div>

          {/* Matrix Rows */}
          {classes.map((rowCls, rIdx) => (
            <div
              key={rowCls}
              style={{
                display: 'grid',
                gridTemplateColumns: '70px repeat(3, 85px)',
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
                  paddingRight: '8px',
                }}
              >
                {rowCls}
              </div>

              {classes.map((_, cIdx) => {
                const normVal = normalized && normalized[rIdx] ? normalized[rIdx][cIdx] : 0;
                const countVal = raw && raw[rIdx] ? raw[rIdx][cIdx] : 0;
                const isDiagonal = rIdx === cIdx;

                // Color interpolation: cream to terracotta
                const bgColor =
                  normVal > 0.001
                    ? `rgba(196, 96, 58, ${Math.max(normVal * 0.85, 0.12)})`
                    : 'var(--bg-secondary)';

                const textColor = normVal > 0.6 ? '#FFFFFF' : 'var(--text-primary)';

                return (
                  <div
                    key={cIdx}
                    style={{
                      backgroundColor: bgColor,
                      border: isDiagonal
                        ? '1.5px solid var(--accent-terracotta)'
                        : '1px solid var(--border-light)',
                      borderRadius: 'var(--radius-xs)',
                      padding: '10px 4px',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                      transition: 'transform 0.1s ease',
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 700,
                        fontSize: '0.98rem',
                        color: textColor,
                      }}
                    >
                      {(normVal * 100).toFixed(1)}%
                    </div>
                    {raw && (
                      <div
                        style={{
                          fontSize: '0.68rem',
                          color: normVal > 0.6 ? '#FBF6EE' : 'var(--text-muted)',
                          marginTop: '2px',
                        }}
                      >
                        n = {countVal}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}

          {/* X-axis title */}
          <div
            style={{
              textAlign: 'center',
              fontSize: '0.78rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--text-secondary)',
              marginTop: '10px',
              marginLeft: '70px',
            }}
          >
            Predicted Model Label
          </div>
        </div>
      </div>
    </div>
  );
};
