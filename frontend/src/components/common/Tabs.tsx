import React from 'react';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  count?: number;
}

interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onChange, className = '' }) => {
  return (
    <div
      style={{
        display: 'flex',
        gap: '4px',
        backgroundColor: 'var(--bg-secondary)',
        padding: '4px',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-sand)',
        overflowX: 'auto',
      }}
      className={`tabs-nav ${className}`}
    >
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.88rem',
              fontWeight: 600,
              fontFamily: 'var(--font-sans)',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              backgroundColor: isActive ? 'var(--surface-card)' : 'transparent',
              color: isActive ? 'var(--accent-terracotta)' : 'var(--text-secondary)',
              boxShadow: isActive ? '0 1px 4px rgba(46, 33, 24, 0.08)' : 'none',
              whiteSpace: 'nowrap',
            }}
          >
            {tab.icon && <span>{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                style={{
                  fontSize: '0.72rem',
                  backgroundColor: isActive ? 'var(--accent-terracotta-soft)' : 'var(--border-sand)',
                  color: isActive ? 'var(--accent-terracotta)' : 'var(--text-secondary)',
                  padding: '1px 6px',
                  borderRadius: 'var(--radius-full)',
                }}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
