import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtitle,
  badge,
  icon,
  highlight = false,
}) => {
  return (
    <div
      style={{
        backgroundColor: highlight ? 'var(--surface-elevated)' : 'var(--surface-card)',
        border: highlight
          ? '1.5px solid var(--accent-terracotta)'
          : '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-md)',
        padding: '18px 20px',
        boxShadow: highlight ? 'var(--shadow-hover)' : 'var(--shadow-card)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        transition: 'all 0.2s ease',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '8px',
        }}
      >
        <span
          style={{
            fontSize: '0.82rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
            color: 'var(--text-secondary)',
          }}
        >
          {label}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {badge}
          {icon && (
            <span style={{ color: 'var(--accent-terracotta)', opacity: 0.85 }}>{icon}</span>
          )}
        </div>
      </div>

      <div
        style={{
          fontSize: '2.1rem',
          fontWeight: 700,
          fontFamily: 'var(--font-serif)',
          color: highlight ? 'var(--accent-terracotta)' : 'var(--text-primary)',
          lineHeight: 1.15,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </div>

      {subtitle && (
        <div
          style={{
            fontSize: '0.78rem',
            color: 'var(--text-muted)',
            marginTop: '8px',
          }}
        >
          {subtitle}
        </div>
      )}
    </div>
  );
};
