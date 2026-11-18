import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Compass,
  Database,
  SlidersHorizontal,
  Activity,
  Layers,
  ArrowRight,
  ArrowLeft,
  PieChart,
  UploadCloud,
  CheckCircle,
  HelpCircle,
  Menu,
  X,
  Server,
  BookOpen,
} from 'lucide-react';
import { api } from '../../services/api';

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactNode;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'Foundations',
    items: [
      { path: '/', label: 'Overview & Monograph', icon: <Compass size={17} /> },
      { path: '/dataset', label: 'Dataset Explorer', icon: <Database size={17} /> },
      { path: '/preprocessing', label: 'Signal Preprocessing', icon: <SlidersHorizontal size={17} /> },
    ],
  },
  {
    title: 'Signals & Models',
    items: [
      { path: '/sensor-explorer', label: '3D Sensor Explorer', icon: <Activity size={17} /> },
      { path: '/model-parameters', label: 'HMM Parameters', icon: <Layers size={17} /> },
    ],
  },
  {
    title: 'Inference Algorithms',
    items: [
      { path: '/forward', label: 'Forward Algorithm', icon: <ArrowRight size={17} /> },
      { path: '/backward', label: 'Backward Algorithm', icon: <ArrowLeft size={17} /> },
      { path: '/posterior', label: 'Posterior Decoding', icon: <PieChart size={17} /> },
    ],
  },
  {
    title: 'Applications & Benchmarks',
    items: [
      { path: '/recognition', label: 'Live Recognition', icon: <UploadCloud size={17} /> },
      { path: '/evaluation', label: 'Model Evaluation', icon: <CheckCircle size={17} /> },
      { path: '/about', label: 'Viva Defense & Theory', icon: <BookOpen size={17} /> },
    ],
  },
];

export const AppShell: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState<{
    online: boolean;
    version?: string;
  }>({ online: false });
  const location = useLocation();

  useEffect(() => {
    // Check backend health periodically
    const checkHealth = async () => {
      try {
        const res = await api.getHealth();
        setBackendStatus({ online: res.status === 'healthy', version: res.version });
      } catch {
        setBackendStatus({ online: false });
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Close mobile sidebar on route change
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Find active item title
  let currentTitle = 'Human Activity Recognition';
  for (const section of NAV_SECTIONS) {
    for (const item of section.items) {
      if (item.path === location.pathname) {
        currentTitle = item.label;
        break;
      }
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        minHeight: '100vh',
        backgroundColor: 'var(--bg-canvas)',
        color: 'var(--text-primary)',
      }}
    >
      {/* Mobile Backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(46, 33, 24, 0.4)',
            backdropFilter: 'blur(3px)',
            zIndex: 40,
          }}
        />
      )}

      {/* Sidebar Navigation */}
      <aside
        style={{
          width: '280px',
          backgroundColor: 'var(--bg-secondary)',
          borderRight: '1px solid var(--border-sand)',
          display: 'flex',
          flexDirection: 'column',
          position: 'fixed',
          top: 0,
          bottom: 0,
          left: sidebarOpen ? 0 : '-280px',
          zIndex: 50,
          transition: 'left 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        className="sidebar-container"
      >
        {/* Sidebar Header */}
        <div
          style={{
            padding: '24px 20px 18px',
            borderBottom: '1px solid var(--border-sand)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span
              style={{
                fontFamily: 'var(--font-serif)',
                fontWeight: 700,
                fontSize: '1.25rem',
                color: 'var(--accent-terracotta)',
                letterSpacing: '-0.02em',
              }}
            >
              HMM &middot; HAR
            </span>
            <button
              onClick={() => setSidebarOpen(false)}
              style={{
                display: 'none',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-secondary)',
              }}
              className="mobile-close-btn"
            >
              <X size={20} />
            </button>
          </div>
          <p
            style={{
              fontSize: '0.78rem',
              color: 'var(--text-secondary)',
              marginTop: '4px',
              lineHeight: 1.35,
            }}
          >
            Human Activity Recognition via Forward &amp; Backward Algorithms
          </p>
        </div>

        {/* Navigation Sections */}
        <nav style={{ flex: 1, overflowY: 'auto', padding: '16px 12px' }}>
          {NAV_SECTIONS.map((sec) => (
            <div key={sec.title} style={{ marginBottom: '20px' }}>
              <div
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  color: 'var(--text-muted)',
                  padding: '4px 12px 6px',
                }}
              >
                {sec.title}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                {sec.items.map((item) => {
                  const isActive = location.pathname === item.path;
                  return (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        padding: '9px 12px',
                        borderRadius: 'var(--radius-xs)',
                        textDecoration: 'none',
                        fontSize: '0.86rem',
                        fontWeight: isActive ? 600 : 500,
                        backgroundColor: isActive ? 'var(--surface-card)' : 'transparent',
                        color: isActive ? 'var(--accent-terracotta)' : 'var(--text-primary)',
                        border: isActive ? '1px solid var(--border-sand)' : '1px solid transparent',
                        boxShadow: isActive ? 'var(--shadow-sm)' : 'none',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <span
                        style={{
                          color: isActive ? 'var(--accent-terracotta)' : 'var(--text-secondary)',
                        }}
                      >
                        {item.icon}
                      </span>
                      <span>{item.label}</span>
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Sidebar Footer with Health Status */}
        <div
          style={{
            padding: '14px 18px',
            borderTop: '1px solid var(--border-sand)',
            backgroundColor: 'var(--surface-card)',
            fontSize: '0.78rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: backendStatus.online ? '#5B8266' : '#CC3333',
                  display: 'inline-block',
                }}
              />
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                {backendStatus.online ? 'FastAPI Connected' : 'FastAPI Offline'}
              </span>
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
              v{backendStatus.version || '1.0'}
            </span>
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
            Backend: 127.0.0.1:8000
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
        }}
        className="main-viewport"
      >
        {/* Top Header */}
        <header
          style={{
            height: '60px',
            backgroundColor: 'var(--surface-card)',
            borderBottom: '1px solid var(--border-sand)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            position: 'sticky',
            top: 0,
            zIndex: 30,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <button
              onClick={() => setSidebarOpen(true)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-primary)',
                display: 'none',
              }}
              className="mobile-hamburger-btn"
            >
              <Menu size={22} />
            </button>
            <h1
              style={{
                fontSize: '1.18rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-serif)',
              }}
            >
              {currentTitle}
            </h1>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.78rem',
                backgroundColor: 'var(--bg-secondary)',
                padding: '4px 10px',
                borderRadius: 'var(--radius-full)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <Server size={13} style={{ color: 'var(--accent-terracotta)' }} />
              <span style={{ fontFamily: 'var(--font-mono)' }}>Pure NumPy Core</span>
            </div>

            <NavLink
              to="/about"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                textDecoration: 'none',
                color: 'var(--text-secondary)',
                fontSize: '0.82rem',
                fontWeight: 600,
              }}
            >
              <HelpCircle size={15} />
              <span>Viva Guide</span>
            </NavLink>
          </div>
        </header>

        {/* Page Content Viewport */}
        <main
          style={{
            flex: 1,
            padding: '28px 32px 60px',
            maxWidth: '1380px',
            width: '100%',
            margin: '0 auto',
          }}
        >
          <Outlet />
        </main>
      </div>

      <style>{`
        @media (min-width: 1024px) {
          .sidebar-container {
            left: 0 !important;
          }
          .main-viewport {
            margin-left: 280px;
          }
        }
        @media (max-width: 1023px) {
          .mobile-hamburger-btn {
            display: block !important;
          }
          .mobile-close-btn {
            display: block !important;
          }
        }
      `}</style>
    </div>
  );
};
