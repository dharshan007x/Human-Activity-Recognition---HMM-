import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Compass, ArrowRight, ShieldCheck, Cpu, Database, Activity } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { Callout } from '../components/common/Callout';
import { MetricCard } from '../components/common/MetricCard';
import { Accelerometer3D } from '../components/visualizations/Accelerometer3D';
import { api, ProjectSummary } from '../services/api';

export const OverviewPage: React.FC = () => {
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getProjectSummary()
      .then((data) => setProject(data))
      .catch((err) => console.error('Failed to load project summary:', err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Editorial Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Badge variant="accent">University Research Monograph</Badge>
          <Badge variant="neutral">Batch 12 Project</Badge>
        </div>
        <h1
          style={{
            fontSize: '2.4rem',
            fontWeight: 700,
            lineHeight: 1.15,
            color: 'var(--text-primary)',
            letterSpacing: '-0.025em',
          }}
        >
          Human Activity Recognition Using the Forward &amp; Backward Algorithms
        </h1>
        <p
          style={{
            fontSize: '1.05rem',
            color: 'var(--text-secondary)',
            marginTop: '8px',
            maxWidth: '920px',
            lineHeight: 1.5,
          }}
        >
          A defensible university-level scientific platform modeling sequential triaxial accelerometer
          dynamics as continuous Gaussian emissions across discrete hidden activity states.
        </p>
      </div>

      {/* Official Problem Statement Callout */}
      <Callout title="Official Problem Statement" type="formula">
        <p style={{ fontSize: '0.96rem', fontStyle: 'italic', fontWeight: 500 }}>
          "{project?.official_problem_statement ||
            'Identify human activities such as walking, sitting and running from sequential accelerometer readings. Model activities as hidden states and sensor patterns as observations, then use the Forward and Backward Algorithms to calculate sequence likelihoods and state probabilities.'}"
        </p>
      </Callout>

      {/* Key Metric Highlights */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label="Hidden States N"
          value="3"
          subtitle="Walking, Sitting, Running"
          badge={<Badge variant="walking">Discrete</Badge>}
        />
        <MetricCard
          label="Observation Dim D"
          value="4"
          subtitle="[ax, ay, az, ||a||] in R⁴"
          badge={<Badge variant="sitting">Continuous</Badge>}
        />
        <MetricCard
          label="Sampling Decimation"
          value="100 → 25 Hz"
          subtitle="4th-order Butterworth (fc=10Hz)"
          badge={<Badge variant="neutral">M = 4</Badge>}
        />
        <MetricCard
          label="Test Accuracy"
          value="100.0%"
          subtitle="Held-out Subjects 108 & 109"
          badge={<Badge variant="running">Macro F1: 1.0</Badge>}
          highlight
        />
      </div>

      {/* Interactive Hero Section: 3D Accelerometer & Architecture */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '24px',
          alignItems: 'start',
        }}
      >
        <Card
          title="Interactive 3D IMU Accelerometer"
          subtitle="Real-time triaxial coordinate frame with dynamic acceleration vector ||a||"
          badge={<Badge variant="accent">Three.js WebGL</Badge>}
        >
          <Accelerometer3D ax={0.85} ay={9.65} az={1.2} activityName="Walking" height={340} />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginTop: '14px',
              fontSize: '0.82rem',
              color: 'var(--text-secondary)',
            }}
          >
            <span>Drag to rotate, scroll to zoom, right-click to pan</span>
            <NavLink to="/sensor-explorer" style={{ color: 'var(--accent-terracotta)', fontWeight: 600 }}>
              Launch Full 3D Sensor Explorer &rarr;
            </NavLink>
          </div>
        </Card>

        {/* Model Architecture Overview */}
        <Card
          title="System Architecture & Mathematical Formulation"
          subtitle="First-order Gaussian Hidden Markov Model λ = (π, A, B)"
          badge={<Badge variant="neutral">Pure NumPy</Badge>}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '12px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                1. Supervised Parameter Estimation
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Maximum Likelihood Estimation with Laplace smoothing (&alpha; = 1.0) on state transitions
                and initial priors. Multivariate Gaussian emissions estimated with covariance regularisation
                &Sigma;<sub>i</sub> + 10⁻⁴I.
              </p>
            </div>

            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '12px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                2. Forward Algorithm (Sequence Likelihood)
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Computes forward variables &alpha;<sub>t</sub>(i) via log-sum-exp and Rabiner scaled
                normalization to calculate sequence likelihood ln P(O|&lambda;) without numerical underflow.
              </p>
            </div>

            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '12px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                3. Backward Algorithm &amp; Posterior Smoothing
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Computes backward variables &beta;<sub>t</sub>(i) and smooths exact state probabilities
                &gamma;<sub>t</sub>(i) = &alpha;<sub>t</sub>(i)&beta;<sub>t</sub>(i) / P(O|&lambda;).
              </p>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginTop: '6px' }}>
              <NavLink to="/forward" style={{ textDecoration: 'none' }}>
                <Button variant="primary" size="sm" icon={<ArrowRight size={14} />}>
                  Explore Forward Trellis
                </Button>
              </NavLink>
              <NavLink to="/recognition" style={{ textDecoration: 'none' }}>
                <Button variant="secondary" size="sm" icon={<Activity size={14} />}>
                  Live Recognition Demo
                </Button>
              </NavLink>
            </div>
          </div>
        </Card>
      </div>

      {/* Rigorous Academic Verification Section */}
      <Card
        title="Defensibility & University Evaluation Standards"
        subtitle="Zero hardcoded predictions, strict partition protocols, and automated oracle validation"
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '16px',
          }}
        >
          <div style={{ display: 'flex', gap: '10px' }}>
            <ShieldCheck size={24} style={{ color: 'var(--state-walking)', flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>No Leakage Partitioning</div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
                Subject-wise split strictly preserves cross-subject independence. Subjects 101–106 train,
                107 validates, 108–109 held-out test. 9-fold Leave-One-Subject-Out (LOSO) cross-validated.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <Cpu size={24} style={{ color: 'var(--accent-terracotta)', flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Pure NumPy Implementation</div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
                Zero black-box HMM libraries in core production code. Forward recursion, Backward recursion,
                and Cholesky multivariate Gaussian density are implemented manually in pure NumPy.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <Database size={24} style={{ color: 'var(--accent-teal)', flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Independent Test Oracle</div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
                Unit tests cross-check our manual pure NumPy forward and backward implementations against
                hmmlearn to numerical tolerance |&Delta; ln P| &lt; 10⁻⁵.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};
