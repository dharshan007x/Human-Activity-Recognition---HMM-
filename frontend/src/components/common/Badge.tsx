import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'walking' | 'sitting' | 'running' | 'neutral' | 'accent' | 'outline';
  size?: 'sm' | 'md';
  style?: React.CSSProperties;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  size = 'md',
  style,
}) => {
  const getBadgeStyle = (): React.CSSProperties => {
    const base: React.CSSProperties = {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '5px',
      borderRadius: 'var(--radius-full)',
      fontWeight: 600,
      letterSpacing: '0.02em',
      textTransform: 'uppercase',
      lineHeight: 1,
    };

    const sizeStyles = {
      sm: { padding: '3px 8px', fontSize: '0.68rem' },
      md: { padding: '5px 11px', fontSize: '0.75rem' },
    };

    const variantStyles: Record<string, React.CSSProperties> = {
      walking: {
        backgroundColor: 'var(--state-walking-soft)',
        color: 'var(--state-walking)',
        border: '1px solid var(--state-walking-border)',
      },
      sitting: {
        backgroundColor: 'var(--state-sitting-soft)',
        color: 'var(--state-sitting)',
        border: '1px solid var(--state-sitting-border)',
      },
      running: {
        backgroundColor: 'var(--state-running-soft)',
        color: 'var(--state-running)',
        border: '1px solid var(--state-running-border)',
      },
      accent: {
        backgroundColor: 'var(--accent-terracotta-soft)',
        color: 'var(--accent-terracotta)',
        border: '1px solid var(--accent-terracotta-border)',
      },
      neutral: {
        backgroundColor: 'var(--bg-secondary)',
        color: 'var(--text-secondary)',
        border: '1px solid var(--border-sand)',
      },
      outline: {
        backgroundColor: 'transparent',
        color: 'var(--text-secondary)',
        border: '1px solid var(--border-dark)',
      },
    };

    return {
      ...base,
      ...sizeStyles[size],
      ...variantStyles[variant],
      ...style,
    };
  };

  return <span style={getBadgeStyle()}>{children}</span>;
};
