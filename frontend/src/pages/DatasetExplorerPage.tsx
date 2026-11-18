import React, { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { MetricCard } from '../components/common/MetricCard';
import { Table, Column } from '../components/common/Table';
import { Button } from '../components/common/Button';
import { api, DatasetMetadata, DatasetSummary, DatasetPreview } from '../services/api';

export const DatasetExplorerPage: React.FC = () => {
  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [limit, setLimit] = useState(25);
  const [offset, setOffset] = useState(0);

  const loadData = () => {
    Promise.all([
      api.getDatasetMetadata(),
      api.getDatasetSummary(),
      api.getDatasetPreview(limit, offset),
    ])
      .then(([meta, summ, prev]) => {
        setMetadata(meta);
        setSummary(summ);
        setPreview(prev);
      })
      .catch((err) => console.error('Error fetching dataset info:', err));
  };

  useEffect(() => {
    loadData();
  }, [offset, limit]);

  const columns: Column<Record<string, number | string>>[] = [
    { header: 'Subject', accessor: 'subject_id', align: 'center', width: '80px' },
    {
      header: 'Activity',
      accessor: 'activity_name',
      render: (row) => {
        const act = String(row['activity_name'] ?? 'Walking');
        const variant =
          act === 'Walking' ? 'walking' : act === 'Sitting' ? 'sitting' : 'running';
        return <Badge variant={variant}>{act}</Badge>;
      },
      width: '120px',
    },
    {
      header: 'Timestamp (s)',
      accessor: 'timestamp_s',
      render: (row) => {
        const ts = row['timestamp_s'] ?? row['timestamp'];
        return ts != null ? Number(ts).toFixed(3) : '-';
      },
      align: 'right',
      width: '120px',
    },
    {
      header: 'acc_x (m/s²)',
      accessor: 'acc_x',
      render: (row) => (row['acc_x'] != null ? Number(row['acc_x']).toFixed(3) : '-'),
      align: 'right',
    },
    {
      header: 'acc_y (m/s²)',
      accessor: 'acc_y',
      render: (row) => (row['acc_y'] != null ? Number(row['acc_y']).toFixed(3) : '-'),
      align: 'right',
    },
    {
      header: 'acc_z (m/s²)',
      accessor: 'acc_z',
      render: (row) => (row['acc_z'] != null ? Number(row['acc_z']).toFixed(3) : '-'),
      align: 'right',
    },
    {
      header: '||acc|| (m/s²)',
      accessor: 'acc_mag',
      render: (row) => (
        <strong style={{ color: 'var(--accent-terracotta)' }}>
          {row['acc_mag'] != null ? Number(row['acc_mag']).toFixed(3) : '-'}
        </strong>
      ),
      align: 'right',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          PAMAP2 Dataset Explorer
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Physical Activity Monitoring Dataset (UCI Machine Learning Repository). Triaxial IMU telemetry
          from 9 participants across Walking, Sitting, and Running activities.
        </p>
      </div>

      {/* Dataset Metrics */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: '16px',
        }}
      >
        <MetricCard
          label="Total Records"
          value={(summary?.total_processed_samples ?? summary?.total_records ?? 40500).toLocaleString()}
          subtitle="Decimated at 25 Hz"
          badge={<Badge variant="neutral">25 Hz</Badge>}
        />
        <MetricCard
          label="Sequences (T=50)"
          value={(
            summary?.sequence_count ?? Math.floor((summary?.total_records ?? 40500) / 50)
          ).toLocaleString()}
          subtitle="2.0s observation windows"
          badge={<Badge variant="accent">Non-overlapping</Badge>}
        />
        <MetricCard
          label="Subjects"
          value="9 Participants"
          subtitle="Train: 101–106 | Val: 107 | Test: 108–109"
          badge={<Badge variant="walking">No Leakage</Badge>}
        />
        <MetricCard
          label="IMU Placement"
          value="Chest IMU ±16g"
          subtitle="Thoracic center of mass"
          badge={<Badge variant="neutral">Cols 21–23</Badge>}
        />
      </div>

      {/* Subject Distribution & Class Balance */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
          gap: '20px',
        }}
      >
        <Card title="Activity Class Distribution" subtitle="Balanced representation across target motion states">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {(summary?.activity_distribution || summary?.records_per_activity) &&
              Object.entries(summary.activity_distribution || summary.records_per_activity || {}).map(([act, count]) => {
                const total = summary?.total_processed_samples ?? summary?.total_records ?? 1;
                const pct = ((count / total) * 100).toFixed(1);
                const color =
                  act === 'Walking' ? '#5B8266' : act === 'Sitting' ? '#C98A2C' : '#C4603A';
                return (
                  <div key={act}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.85rem',
                        marginBottom: '4px',
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>{act}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                        {count.toLocaleString()} samples ({pct}%)
                      </span>
                    </div>
                    <div
                      style={{
                        height: '8px',
                        backgroundColor: 'var(--bg-secondary)',
                        borderRadius: 'var(--radius-full)',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${pct}%`,
                          backgroundColor: color,
                          borderRadius: 'var(--radius-full)',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </Card>

        <Card title="Experimental Subject Partition" subtitle="Strict subject-independent generalization setup">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.86rem' }}>
            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '10px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Training Partition (6 Subjects)</strong>
                <Badge variant="walking">Subjects 101–106</Badge>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Used for Maximum Likelihood parameter estimation (transition matrix A, priors π, Gaussian emissions μ, Σ).
              </p>
            </div>

            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '10px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Validation Partition (1 Subject)</strong>
                <Badge variant="sitting">Subject 107</Badge>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Tuning of covariance regularisation diagonal damping factor and window segmentation hyperparameter T=50.
              </p>
            </div>

            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                padding: '10px 14px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-sand)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Held-Out Test Partition (2 Subjects)</strong>
                <Badge variant="running">Subjects 108–109</Badge>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Completely unseen participants evaluating zero-leakage generalized activity recognition accuracy.
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Dataset Preview Table */}
      <Card
        title="Decimated Sensor Records Preview"
        subtitle={`Showing records ${offset + 1} to ${Math.min(
          offset + limit,
          preview?.total_records ?? preview?.total_rows ?? 0
        )} of ${(preview?.total_records ?? preview?.total_rows ?? 0).toLocaleString()}`}
        action={
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button
              variant="secondary"
              size="sm"
              disabled={offset <= 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              Previous {limit}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={offset + limit >= (preview?.total_records ?? preview?.total_rows ?? 0)}
              onClick={() => setOffset(offset + limit)}
            >
              Next {limit}
            </Button>
            <Button variant="ghost" size="sm" onClick={loadData}>
              <RefreshCw size={14} />
            </Button>
          </div>
        }
      >
        <Table
          columns={columns}
          data={preview?.records ?? preview?.rows ?? []}
          emptyMessage="No accelerometer records loaded."
        />
      </Card>
    </div>
  );
};
