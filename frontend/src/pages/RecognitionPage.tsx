import React, { useState } from 'react';
import { Download } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { Button } from '../components/common/Button';
import { MetricCard } from '../components/common/MetricCard';
import { Table, Column } from '../components/common/Table';
import { Slider } from '../components/common/Slider';
import { UploadZone } from '../components/common/UploadZone';
import { Accelerometer3D } from '../components/visualizations/Accelerometer3D';
import { api, RecognitionResult, StepPrediction } from '../services/api';

export const RecognitionPage: React.FC = () => {
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  const handleFile = async (file: File) => {
    setLoading(true);
    try {
      const res = await api.recognizeCsv(file);
      setResult(res);
      setActiveStep(0);
    } catch (err: any) {
      console.error('Recognition error:', err);
      alert(`Error recognizing activity: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUseSample = async () => {
    // Generate a synthetic test CSV in memory and upload
    const csvContent =
      'acc_x,acc_y,acc_z\n' +
      Array.from({ length: 60 })
        .map((_, i) => {
          const t = i * 0.04;
          const ax = (1.5 * Math.sin(t * 8) + 0.2 * (Math.random() - 0.5)).toFixed(3);
          const ay = (9.8 + 2.0 * Math.cos(t * 8) + 0.2 * (Math.random() - 0.5)).toFixed(3);
          const az = (1.1 * Math.sin(t * 16) + 0.2 * (Math.random() - 0.5)).toFixed(3);
          return `${ax},${ay},${az}`;
        })
        .join('\n');

    const file = new File([csvContent], 'pamap2_walking_sample.csv', { type: 'text/csv' });
    handleFile(file);
  };

  const downloadResultsCsv = () => {
    if (!result) return;
    const header =
      'step,t_sec,acc_x,acc_y,acc_z,acc_mag,predicted_activity,confidence,walking_prob,sitting_prob,running_prob\n';
    const rows = result.steps
      .map(
        (s) =>
          `${s.step},${s.t_sec},${s.acc_x},${s.acc_y},${s.acc_z},${s.acc_mag},${s.predicted_activity},${s.confidence},${s.walking_prob},${s.sitting_prob},${s.running_prob}`
      )
      .join('\n');

    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recognition_results_${result.filename}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const currentStep = result?.steps?.[activeStep];
  const totalSteps = result?.total_timesteps ?? result?.steps?.length ?? 0;

  const columns: Column<StepPrediction>[] = [
    { header: 'Step t', accessor: 'step', align: 'center', width: '70px' },
    {
      header: 'Time (s)',
      accessor: 't_sec',
      render: (r) => (r.t_sec != null ? r.t_sec.toFixed(3) : '-'),
      align: 'right',
      width: '90px',
    },
    {
      header: 'acc_x',
      accessor: 'acc_x',
      render: (r) => (r.acc_x != null ? r.acc_x.toFixed(2) : '-'),
      align: 'right',
    },
    {
      header: 'acc_y',
      accessor: 'acc_y',
      render: (r) => (r.acc_y != null ? r.acc_y.toFixed(2) : '-'),
      align: 'right',
    },
    {
      header: 'acc_z',
      accessor: 'acc_z',
      render: (r) => (r.acc_z != null ? r.acc_z.toFixed(2) : '-'),
      align: 'right',
    },
    {
      header: 'Predicted Activity',
      accessor: 'predicted_activity',
      render: (r) => {
        const v =
          r.predicted_activity === 'Walking'
            ? 'walking'
            : r.predicted_activity === 'Sitting'
            ? 'sitting'
            : 'running';
        return <Badge variant={v}>{r.predicted_activity}</Badge>;
      },
      width: '130px',
    },
    {
      header: 'Confidence',
      accessor: 'confidence',
      render: (r) => (
        <strong style={{ color: 'var(--accent-terracotta)' }}>
          {r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '-'}
        </strong>
      ),
      align: 'right',
      width: '100px',
    },
    {
      header: 'P(Walking)',
      accessor: 'walking_prob',
      render: (r) => (r.walking_prob != null ? (r.walking_prob * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: 'P(Sitting)',
      accessor: 'sitting_prob',
      render: (r) => (r.sitting_prob != null ? (r.sitting_prob * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
    {
      header: 'P(Running)',
      accessor: 'running_prob',
      render: (r) => (r.running_prob != null ? (r.running_prob * 100).toFixed(1) + '%' : '-'),
      align: 'right',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '26px' }}>
      <div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Live Human Activity Recognition
        </h1>
        <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Upload custom triaxial accelerometer CSV records to execute the pure NumPy Forward-Backward
          posterior inference engine and classify human motion states.
        </p>
      </div>

      {/* CSV Uploader */}
      <Card title="Upload Accelerometer CSV Stream" subtitle="Drop your sensor recording or load a sample sequence">
        <UploadZone
          onFileSelected={handleFile}
          isLoading={loading}
          onUseSample={handleUseSample}
        />
      </Card>

      {/* Recognition Results */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Key Metrics */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
              gap: '16px',
            }}
          >
            <MetricCard
              label="Dominant Activity"
              value={result.overall_predicted_activity || 'Unknown'}
              subtitle="Majority state across sequence"
              badge={
                <Badge
                  variant={
                    result.overall_predicted_activity === 'Walking'
                      ? 'walking'
                      : result.overall_predicted_activity === 'Sitting'
                      ? 'sitting'
                      : 'running'
                  }
                >
                  Classified
                </Badge>
              }
              highlight
            />
            <MetricCard
              label="Average Confidence"
              value={
                result.mean_confidence != null
                  ? `${(result.mean_confidence * 100).toFixed(1)}%`
                  : '-'
              }
              subtitle="Mean maximum posterior probability"
              badge={<Badge variant="accent">Mean max γ</Badge>}
            />
            <MetricCard
              label="Duration (Seconds)"
              value={
                result.duration_seconds != null
                  ? `${result.duration_seconds.toFixed(2)}s`
                  : '-'
              }
              subtitle={`Total: ${totalSteps} time steps`}
              badge={<Badge variant="neutral">25 Hz</Badge>}
            />
            <MetricCard
              label="Input File"
              value={result.filename}
              subtitle="Processed by FastAPI Backend"
              badge={<Badge variant="walking">Verified</Badge>}
            />
          </div>

          {/* Synchronized 3D Accelerometer Playback */}
          {currentStep && (
            <Card
              title="Synchronized 3D IMU Playback of Uploaded Motion"
              subtitle={`Instantaneous vector at step t = ${activeStep}: [${currentStep.acc_x.toFixed(
                2
              )}, ${currentStep.acc_y.toFixed(2)}, ${currentStep.acc_z.toFixed(
                2
              )}] m/s² (Class: ${currentStep.predicted_activity})`}
              badge={<Badge variant="accent">Three.js Playback</Badge>}
            >
              <Accelerometer3D
                ax={currentStep.acc_x}
                ay={currentStep.acc_y}
                az={currentStep.acc_z}
                activityName={currentStep.predicted_activity}
                height={320}
              />
              <div style={{ marginTop: '14px' }}>
                <Slider
                  min={0}
                  max={Math.max(totalSteps - 1, 1)}
                  value={activeStep}
                  onChange={setActiveStep}
                  label="Trajectory Playback Scrubber"
                  unit="t"
                  fps={12}
                />
              </div>
            </Card>
          )}

          {/* Detailed Predictions Table */}
          <Card
            title="Step-by-Step State Classification Log"
            subtitle={`Showing all ${totalSteps} decoded time steps with posterior probability breakdown`}
            action={
              <Button
                variant="secondary"
                size="sm"
                onClick={downloadResultsCsv}
                icon={<Download size={14} />}
              >
                Export CSV
              </Button>
            }
          >
            <Table
              columns={columns}
              data={result.steps || []}
              maxHeight="380px"
              keyExtractor={(r) => r.step}
            />
          </Card>
        </div>
      )}
    </div>
  );
};
