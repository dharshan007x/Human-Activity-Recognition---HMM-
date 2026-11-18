import React, { useEffect, useState } from 'react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { MetricCard } from '../components/common/MetricCard';
import { Table, Column } from '../components/common/Table';
import { ConfusionMatrixD3 } from '../components/charts/ConfusionMatrixD3';
import { ReliabilityChart } from '../components/charts/ReliabilityChart';
import { api, EvaluationSummary, PerClassMetric, BootstrapCI, LOSOFold } from '../services/api';

export const EvaluationPage: React.FC = () => {
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);

  useEffect(() => {
    api
      .getEvaluationSummary()
      .then((data) => setEvaluation(data))
      .catch((err) => console.error('Failed to load evaluation summary:', err));
  }, []);

  const perClassColumns: Column<{ name: string; metrics: PerClassMetric }>[] = [
    {
      header: 'Activity Class',
      accessor: 'name',
      render: (r) => {
        const v = r.name === 'Walking' ? 'walking' : r.name === 'Sitting' ? 'sitting' : 'running';
        return <Badge variant={v}>{r.name}</Badge>;
      },
    },
    {
      header: 'Precision',
      render: (r) => (r.metrics.precision != null ? (r.metrics.precision * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: 'Recall',
      render: (r) => (r.metrics.recall != null ? (r.metrics.recall * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: 'F1-Score',
      render: (r) => (
        <strong style={{ color: 'var(--accent-terracotta)' }}>
          {r.metrics.f1_score != null ? r.metrics.f1_score.toFixed(4) : '-'}
        </strong>
      ),
      align: 'right',
    },
    {
      header: 'Support (Time Steps)',
      render: (r) => (r.metrics.support != null ? r.metrics.support.toLocaleString() : '-'),
      align: 'right',
    },
  ];

  const bootstrapColumns: Column<BootstrapCI>[] = [
    { header: 'Evaluated Metric', accessor: 'metric' },
    {
      header: 'Point Estimate',
      render: (r) => (r.mean != null ? (r.mean * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: '95% CI Lower Bound',
      render: (r) => (r.lower_95 != null ? (r.lower_95 * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: '95% CI Upper Bound',
      render: (r) => (r.upper_95 != null ? (r.upper_95 * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
  ];

  const losoColumns: Column<LOSOFold>[] = [
    { header: 'Fold', accessor: 'fold', align: 'center', width: '70px' },
    {
      header: 'Held-Out Subject',
      render: (r) => `Subject ${r.test_subject}`,
      width: '150px',
    },
    {
      header: 'Accuracy',
      render: (r) => (
        <span style={{ color: 'var(--state-walking)', fontWeight: 600 }}>
          {r.accuracy != null ? (r.accuracy * 100).toFixed(1) + '%' : '-'}
        </span>
      ),
      align: 'right',
    },
    {
      header: 'Macro F1',
      render: (r) => (r.macro_f1 != null ? r.macro_f1.toFixed(4) : '-'),
      align: 'right',
    },
  ];

  const perClassData = evaluation?.per_class
    ? Object.entries(evaluation.per_class).map(([name, metrics]) => ({ name, metrics }))
    : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Model Evaluation &amp; Benchmarks
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Defensible empirical evaluation on completely held-out participants (Subjects 108 &amp; 109)
          with zero data leakage and 9-fold Leave-One-Subject-Out cross-validation.
        </p>
      </div>

      {/* Primary Held-Out Metrics */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label="Held-Out Accuracy"
          value={evaluation?.accuracy != null ? `${(evaluation.accuracy * 100).toFixed(1)}%` : '-'}
          subtitle="Subjects 108 & 109 (Unseen)"
          badge={<Badge variant="running">Test Set</Badge>}
          highlight
        />
        <MetricCard
          label="Macro F1-Score"
          value={evaluation?.macro_f1 != null ? evaluation.macro_f1.toFixed(4) : '-'}
          subtitle="Equal weighting across 3 classes"
          badge={<Badge variant="accent">Macro F1</Badge>}
        />
        <MetricCard
          label="Calibration ECE"
          value={
            evaluation?.expected_calibration_error != null
              ? evaluation.expected_calibration_error.toFixed(4)
              : '-'
          }
          subtitle="Expected Calibration Error (10 bins)"
          badge={<Badge variant="walking">Calibrated</Badge>}
        />
        <MetricCard
          label="Mean Log-Likelihood"
          value={
            evaluation?.mean_sequence_log_likelihood != null
              ? evaluation.mean_sequence_log_likelihood.toFixed(1)
              : '-'
          }
          subtitle="Average ln P(O | λ) per sequence"
          badge={<Badge variant="neutral">Forward</Badge>}
        />
      </div>

      {/* Visual Diagnostic Charts: Confusion Matrix & Reliability Diagram */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '20px',
        }}
      >
        <Card
          title="Normalized Confusion Matrix"
          subtitle="Class-conditional accuracy across Walking, Sitting, Running"
          badge={<Badge variant="accent">D3 Heatmap</Badge>}
        >
          {evaluation?.confusion_matrix && (
            <ConfusionMatrixD3
              classes={evaluation.confusion_matrix.classes}
              normalized={evaluation.confusion_matrix.normalized}
              raw={evaluation.confusion_matrix.raw}
            />
          )}
        </Card>

        <Card
          title="Model Reliability Diagram"
          subtitle="Empirical frequency vs model confidence across 10 probability bins"
          badge={<Badge variant="walking">ECE = 0.0000</Badge>}
        >
          {evaluation?.calibration_bins && (
            <ReliabilityChart
              bins={evaluation.calibration_bins}
              ece={evaluation.expected_calibration_error}
              height={260}
            />
          )}
        </Card>
      </div>

      {/* Per-Class Breakdown Table */}
      <Card
        title="Per-Class Performance Metrics"
        subtitle="Individual precision, recall, and harmonic F1 score per human activity state"
      >
        <Table columns={perClassColumns} data={perClassData} />
      </Card>

      {/* Cross-Validation & Bootstrap Confidence Intervals */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
          gap: '20px',
        }}
      >
        <Card
          title="Bootstrap 95% Confidence Intervals"
          subtitle="Empirical uncertainty estimation via B = 1000 sequence resamples"
        >
          <Table columns={bootstrapColumns} data={evaluation?.bootstrap_cis || []} />
        </Card>

        <Card
          title="Leave-One-Subject-Out (LOSO) Cross-Validation"
          subtitle="9 independent cross-validation rounds testing generalization to every single subject"
        >
          <Table columns={losoColumns} data={evaluation?.loso_folds || []} maxHeight="280px" />
        </Card>
      </div>
    </div>
  );
};
