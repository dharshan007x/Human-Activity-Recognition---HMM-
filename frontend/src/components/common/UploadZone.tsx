import React, { useRef, useState } from 'react';
import { Upload, FileText, AlertCircle, Download, CheckCircle2 } from 'lucide-react';
import { Button } from './Button';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  isLoading?: boolean;
  onUseSample?: () => void;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileSelected,
  isLoading = false,
  onUseSample,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const validateAndPassFile = (file: File) => {
    setError(null);
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
      setError('Invalid file type. Please upload a valid CSV file (.csv).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large (exceeds 10 MB limit).');
      return;
    }
    setSelectedFile(file);
    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndPassFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndPassFile(e.target.files[0]);
    }
  };

  const downloadSampleTemplate = () => {
    const csvContent =
      'timestamp,acc_x,acc_y,acc_z\n' +
      '0.00,0.85,9.65,1.20\n' +
      '0.04,1.12,9.45,1.45\n' +
      '0.08,0.92,9.80,1.10\n' +
      '0.12,0.78,9.90,0.95\n' +
      '0.16,1.05,9.50,1.30\n' +
      '0.20,1.25,9.35,1.55\n' +
      '0.24,0.80,9.75,1.15\n' +
      '0.28,0.85,9.60,1.22\n' +
      '0.32,1.10,9.40,1.40\n' +
      '0.36,0.95,9.70,1.18\n';

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_accelerometer_sequence.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${
            isDragOver
              ? 'var(--accent-terracotta)'
              : selectedFile
              ? 'var(--state-walking)'
              : 'var(--border-dark)'
          }`,
          backgroundColor: isDragOver
            ? 'var(--accent-terracotta-soft)'
            : selectedFile
            ? 'var(--state-walking-soft)'
            : 'var(--surface-card)',
          borderRadius: 'var(--radius-md)',
          padding: '36px 24px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt"
          style={{ display: 'none' }}
          onChange={handleInputChange}
        />

        <div
          style={{
            width: '52px',
            height: '52px',
            borderRadius: 'var(--radius-full)',
            backgroundColor: selectedFile ? 'var(--state-walking)' : 'var(--bg-secondary)',
            color: selectedFile ? '#FFFFFF' : 'var(--accent-terracotta)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {selectedFile ? <CheckCircle2 size={28} /> : <Upload size={28} />}
        </div>

        <div>
          <div
            style={{
              fontWeight: 600,
              fontSize: '1.05rem',
              color: 'var(--text-primary)',
              marginBottom: '4px',
            }}
          >
            {selectedFile
              ? `Selected: ${selectedFile.name}`
              : 'Drag & Drop Accelerometer CSV here, or Browse'}
          </div>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
            Requires triaxial columns: <code>acc_x, acc_y, acc_z</code> (or <code>ax, ay, az</code>). Max 10MB.
          </div>
        </div>

        {selectedFile && (
          <div
            style={{
              fontSize: '0.78rem',
              color: 'var(--state-walking)',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <FileText size={14} /> {(selectedFile.size / 1024).toFixed(1)} KB — Click to change file
          </div>
        )}
      </div>

      {error && (
        <div
          style={{
            backgroundColor: 'var(--state-sitting-soft)',
            border: '1px solid var(--state-sitting-border)',
            color: 'var(--state-sitting)',
            padding: '10px 14px',
            borderRadius: 'var(--radius-xs)',
            fontSize: '0.84rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <AlertCircle size={16} /> {error}
        </div>
      )}

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.85rem',
        }}
      >
        <button
          onClick={downloadSampleTemplate}
          type="button"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--accent-teal)',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontWeight: 600,
            textDecoration: 'underline',
          }}
        >
          <Download size={14} /> Download Sample CSV Template
        </button>

        {onUseSample && (
          <Button variant="secondary" size="sm" onClick={onUseSample} disabled={isLoading}>
            Load Built-in PAMAP2 Validation Sample
          </Button>
        )}
      </div>
    </div>
  );
};
