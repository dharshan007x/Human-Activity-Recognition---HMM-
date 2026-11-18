import React from 'react';
import { Info, AlertTriangle, CheckCircle, BookOpen, Lightbulb } from 'lucide-react';

interface CalloutProps {
  children: React.ReactNode;
  title?: string;
  type?: 'info' | 'formula' | 'warning' | 'success' | 'note';
  style?: React.CSSProperties;
}

export const Callout: React.FC<CalloutProps> = ({
  children,
  title,
  type = 'info',
  style,
}) => {
  const config = {
    info: {
      bg: 'var(--accent-teal-soft)',
      border: 'var(--accent-teal-border)',
      color: 'var(--accent-teal)',
      icon: <Info size={18} />,
    },
    formula: {
      bg: 'var(--bg-secondary)',
      border: 'var(--border-dark)',
      color: 'var(--text-primary)',
      icon: <BookOpen size={18} />,
    },
    warning: {
      bg: 'var(--state-sitting-soft)',
      border: 'var(--state-sitting-border)',
      color: 'var(--state-sitting)',
      icon: <AlertTriangle size={18} />,
    },
    success: {
      bg: 'var(--state-walking-soft)',
      border: 'var(--state-walking-border)',
      color: 'var(--state-walking)',
      icon: <CheckCircle size={18} />,
    },
    note: {
      bg: 'var(--accent-terracotta-soft)',
      border: 'var(--accent-terracotta-border)',
      color: 'var(--accent-terracotta)',
      icon: <Lightbulb size={18} />,
    },
  }[type];

  return (
    <div
      style={{
        backgroundColor: config.bg,
        borderLeft: `4px solid ${config.color}`,
        borderTop: `1px solid ${config.border}`,
        borderRight: `1px solid ${config.border}`,
        borderBottom: `1px solid ${config.border}`,
        borderRadius: 'var(--radius-xs)',
        padding: '14px 18px',
        margin: '12px 0',
        ...style,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: config.color,
          fontWeight: 600,
          fontSize: '0.92rem',
          marginBottom: title ? '6px' : '0',
        }}
      >
        <span>{config.icon}</span>
        {title && <span>{title}</span>}
      </div>
      <div
        style={{
          fontSize: '0.88rem',
          color: 'var(--text-primary)',
          lineHeight: 1.55,
        }}
      >
        {children}
      </div>
    </div>
  );
};
