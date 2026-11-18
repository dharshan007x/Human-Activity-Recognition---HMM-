import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  icon,
  className = '',
  disabled,
  style,
  ...props
}) => {
  const getStyles = (): React.CSSProperties => {
    const base: React.CSSProperties = {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'var(--font-sans)',
      fontWeight: 600,
      borderRadius: 'var(--radius-sm)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.55 : 1,
      transition: 'all 0.15s ease',
      border: '1px solid transparent',
      textDecoration: 'none',
      whiteSpace: 'nowrap',
    };

    const sizeStyles: Record<string, React.CSSProperties> = {
      sm: { padding: '6px 12px', fontSize: '0.82rem' },
      md: { padding: '9px 18px', fontSize: '0.92rem' },
      lg: { padding: '12px 24px', fontSize: '1.05rem' },
    };

    const variantStyles: Record<string, React.CSSProperties> = {
      primary: {
        backgroundColor: 'var(--accent-terracotta)',
        color: '#FFFFFF',
        borderColor: 'var(--accent-terracotta)',
        boxShadow: '0 1px 3px rgba(196, 96, 58, 0.3)',
      },
      secondary: {
        backgroundColor: 'var(--bg-secondary)',
        color: 'var(--text-primary)',
        borderColor: 'var(--border-sand)',
      },
      outline: {
        backgroundColor: 'transparent',
        color: 'var(--accent-terracotta)',
        borderColor: 'var(--accent-terracotta-border)',
      },
      ghost: {
        backgroundColor: 'transparent',
        color: 'var(--text-secondary)',
        borderColor: 'transparent',
      },
    };

    return {
      ...base,
      ...sizeStyles[size],
      ...variantStyles[variant],
      ...style,
    };
  };

  return (
    <button
      disabled={disabled}
      style={getStyles()}
      className={`btn-${variant} ${className}`}
      {...props}
    >
      {icon && <span style={{ display: 'inline-flex', fontSize: '1.1em' }}>{icon}</span>}
      {children}
    </button>
  );
};
