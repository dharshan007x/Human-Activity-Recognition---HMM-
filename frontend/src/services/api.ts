/**
 * API client for the Human Activity Recognition HMM FastAPI Backend.
 */

export interface ProjectSummary {
  title: string;
  official_problem_statement: string;
  hidden_states: string[];
  observation_features: string[];
  model_architecture: string;
  inference_engine: string;
  design_theme: {
    palette: Record<string, string>;
    activity_colors: Record<string, string>;
  };
}

export interface DatasetMetadata {
  dataset_name: string;
  repository: string;
  sampling_rate_hz: number;
  target_sampling_rate_hz: number;
  total_subjects: number;
  subjects_train: number[];
  subjects_val: number[];
  subjects_test: number[];
  imu_placements: string[];
  chosen_imu_placement: string;
  activities: Record<string, number>;
  total_raw_records: number;
}

export interface DatasetSummary {
  total_processed_samples?: number;
  total_records?: number;
  activity_distribution?: Record<string, number>;
  records_per_activity?: Record<string, number>;
  subject_distribution?: Record<string, number>;
  sequence_count?: number;
  feature_statistics?: Record<
    string,
    { mean: number; std: number; min: number; max: number }
  >;
}

export interface DatasetPreview {
  total_records?: number;
  total_rows?: number;
  limit: number;
  offset: number;
  columns?: string[];
  records?: Array<Record<string, number | string>>;
  rows?: Array<Record<string, number | string>>;
}

export interface SignalStream {
  subject_id: number;
  activity_name: string;
  sampling_rate_hz: number;
  duration_sec: number;
  time_stamps: number[];
  channels: {
    acc_x: number[];
    acc_y: number[];
    acc_z: number[];
    acc_mag: number[];
  };
}

export interface FilterResponse {
  cutoff_hz: number;
  order: number;
  nyquist_hz?: number;
  frequencies: number[];
  magnitude_db?: number[];
  gains_db?: number[];
  phase_deg?: number[];
}

export interface SampleSequenceItem {
  id: string;
  subject_id: number;
  activity_name: string;
  length: number;
  raw_sequence: number[][]; // [T, 4]
  scaled_sequence: number[][];
}

export interface StateEmission {
  state: string;
  mean: number[];
  covariance: number[][];
  variance_diagonal: number[];
}

export interface HMMParameters {
  state_names: string[];
  n_states: number;
  n_features: number;
  pi: number[];
  A: number[][];
  emissions: StateEmission[];
}

export interface ForwardResult {
  log_likelihood: number;
  log_alpha_table: number[][];
  scaled_alpha_table: number[][];
  scaling_factors: number[];
  state_names: string[];
  time_ms: number;
}

export interface BackwardResult {
  log_likelihood: number;
  log_beta_table: number[][];
  scaled_beta_table: number[][];
  state_names: string[];
  time_ms: number;
}

export interface PosteriorResult {
  log_likelihood: number;
  gamma_table: number[][];
  predicted_states: string[];
  confidences: number[];
  row_sums_verified: boolean;
  state_names: string[];
  time_ms: number;
}

export interface PredictResult {
  pointwise_states: string[];
  viterbi_states: string[];
  viterbi_log_likelihood: number;
  forward_log_likelihood: number;
  backward_log_likelihood: number;
  path_match_percentage: number;
  gamma_table: number[][];
  confidences: number[];
  state_names: string[];
  time_ms: number;
}

export interface ValidationResult {
  forward_log_likelihood: number;
  backward_log_likelihood: number;
  scaled_forward_log_likelihood: number;
  abs_difference_fwd_bwd: number;
  abs_difference_log_scaled: number;
  tolerance_passed: boolean;
  status: string;
  message: string;
}

export interface StepPrediction {
  step: number;
  t_sec: number;
  acc_x: number;
  acc_y: number;
  acc_z: number;
  acc_mag: number;
  predicted_activity: string;
  confidence: number;
  walking_prob: number;
  sitting_prob: number;
  running_prob: number;
}

export interface RecognitionResult {
  filename: string;
  total_timesteps: number;
  duration_seconds: number;
  overall_predicted_activity: string;
  activity_distribution: Record<string, number>;
  mean_confidence: number;
  forward_log_likelihood: number;
  viterbi_log_likelihood: number;
  steps: StepPrediction[];
  status: string;
  message: string;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ConfusionMatrixData {
  classes: string[];
  raw: number[][];
  normalized: number[][];
}

export interface CalibrationBin {
  bin_index: number;
  confidence: number;
  accuracy: number;
  count: number;
}

export interface BootstrapCI {
  metric: string;
  mean: number;
  lower_95: number;
  upper_95: number;
}

export interface LOSOFold {
  fold: number;
  test_subject: number;
  accuracy: number;
  macro_f1: number;
}

export interface EvaluationSummary {
  test_participants: number[];
  total_test_sequences: number;
  total_time_steps: number;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  expected_calibration_error: number;
  mean_sequence_log_likelihood: number;
  per_class: Record<string, PerClassMetric>;
  confusion_matrix: ConfusionMatrixData;
  calibration_bins: CalibrationBin[];
  bootstrap_cis: BootstrapCI[];
  loso_folds: LOSOFold[];
}

const API_BASE = '/api/v1';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errorDetail = `Request failed: ${res.status} ${res.statusText}`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      } else if (errorJson.message) {
        errorDetail = errorJson.message;
      }
    } catch {
      // Keep default errorDetail
    }
    throw new Error(errorDetail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<{ status: string; version: string; model_loaded: boolean }> {
    return fetchJson('/api/v1/health');
  },

  async getProjectSummary(): Promise<ProjectSummary> {
    return fetchJson(`${API_BASE}/project/summary`);
  },

  async getDatasetMetadata(): Promise<DatasetMetadata> {
    return fetchJson(`${API_BASE}/dataset/metadata`);
  },

  async getDatasetSummary(): Promise<DatasetSummary> {
    return fetchJson(`${API_BASE}/dataset/summary`);
  },

  async getDatasetPreview(limit = 50, offset = 0): Promise<DatasetPreview> {
    return fetchJson(`${API_BASE}/dataset/preview?limit=${limit}&offset=${offset}`);
  },

  async getDatasetSignals(subjectId = 101, activityName = 'Walking', maxPoints = 150): Promise<SignalStream> {
    return fetchJson(
      `${API_BASE}/dataset/signals?subject_id=${subjectId}&activity_name=${encodeURIComponent(activityName)}&max_points=${maxPoints}`
    );
  },

  async getFilterResponse(): Promise<FilterResponse> {
    return fetchJson(`${API_BASE}/dataset/filter-response`);
  },

  async getSampleSequences(): Promise<SampleSequenceItem[]> {
    return fetchJson(`${API_BASE}/dataset/samples`);
  },

  async getModelParameters(): Promise<HMMParameters> {
    return fetchJson(`${API_BASE}/model/parameters`);
  },

  async runForward(sequence: number[][], scaleInput = false): Promise<ForwardResult> {
    return fetchJson(`${API_BASE}/inference/forward`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, scale_input: scaleInput }),
    });
  },

  async runBackward(sequence: number[][], scaleInput = false): Promise<BackwardResult> {
    return fetchJson(`${API_BASE}/inference/backward`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, scale_input: scaleInput }),
    });
  },

  async runPosterior(sequence: number[][], scaleInput = false): Promise<PosteriorResult> {
    return fetchJson(`${API_BASE}/inference/posterior`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, scale_input: scaleInput }),
    });
  },

  async runPredict(sequence: number[][], scaleInput = false): Promise<PredictResult> {
    return fetchJson(`${API_BASE}/inference/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, scale_input: scaleInput }),
    });
  },

  async validateLikelihoods(sequence: number[][], scaleInput = false): Promise<ValidationResult> {
    return fetchJson(`${API_BASE}/inference/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, scale_input: scaleInput }),
    });
  },

  async recognizeCsv(file: File): Promise<RecognitionResult> {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson(`${API_BASE}/recognize`, {
      method: 'POST',
      body: formData,
    });
  },

  async getEvaluationSummary(): Promise<EvaluationSummary> {
    return fetchJson(`${API_BASE}/evaluation/summary`);
  },
};
