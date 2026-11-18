import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  badge?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  title,
  subtitle,
  action,
  badge,
}) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--surface-card)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-card)',
        overflow: 'hidden',
        transition: 'all 0.2s ease',
      }}
      className={`card-container ${className}`}
    >
      {(title || subtitle || action || badge) && (
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--border-light)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {title && (
                <h3
                  style={{
                    fontSize: '1.15rem',
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    letterSpacing: '-0.01em',
                  }}
                >
                  {title}
                </h3>
              )}
              {badge}
            </div>
            {subtitle && (
              <p
                style={{
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)',
                  marginTop: '2px',
                }}
              >
                {subtitle}
              </p>
            )}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div style={{ padding: '20px' }}>{children}</div>
    </div>
  );
};
