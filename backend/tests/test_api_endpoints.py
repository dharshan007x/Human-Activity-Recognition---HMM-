"""Integration tests for all FastAPI API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_project_summary():
    response = client.get("/api/v1/project/summary")
    assert response.status_code == 200
    data = response.json()
    assert "Walking" in data["hidden_states"]
    assert "Sitting" in data["hidden_states"]
    assert "Running" in data["hidden_states"]


def test_dataset_metadata():
    response = client.get("/api/v1/dataset/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["original_sampling_rate_hz"] == 100.0
    assert data["target_sampling_rate_hz"] == 25.0
    assert data["decimation_factor"] == 4


def test_dataset_summary():
    response = client.get("/api/v1/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] > 0
    assert data["missing_values_count"] == 0


def test_dataset_preview():
    response = client.get("/api/v1/dataset/preview?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 10
    assert "acc_mag" in data["rows"][0]


def test_dataset_signals():
    response = client.get("/api/v1/dataset/signals?subject_id=101&activity_name=Walking&max_points=50")
    assert response.status_code == 200
    data = response.json()
    assert data["activity_name"] == "Walking"
    assert len(data["points"]) <= 50


def test_dataset_samples():
    response = client.get("/api/v1/dataset/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3  # walking, sitting, running
    assert len(data[0]["raw_sequence"]) == 50


def test_model_parameters():
    response = client.get("/api/v1/model/parameters")
    assert response.status_code == 200
    data = response.json()
    assert data["n_states"] == 3
    assert len(data["pi"]) == 3
    assert len(data["A"]) == 3
    assert len(data["means"]) == 3


def test_inference_forward_backward_posterior():
    samples_resp = client.get("/api/v1/dataset/samples")
    sample_seq = samples_resp.json()[0]["scaled_sequence"]

    # 1. Forward
    fwd_resp = client.post("/api/v1/inference/forward", json={"sequence": sample_seq})
    assert fwd_resp.status_code == 200
    fwd_data = fwd_resp.json()
    assert "log_likelihood" in fwd_data
    assert len(fwd_data["log_alpha_table"]) == len(sample_seq)

    # 2. Backward
    bwd_resp = client.post("/api/v1/inference/backward", json={"sequence": sample_seq})
    assert bwd_resp.status_code == 200
    bwd_data = bwd_resp.json()
    assert "log_likelihood" in bwd_data

    # 3. Posterior
    post_resp = client.post("/api/v1/inference/posterior", json={"sequence": sample_seq})
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["row_sums_verified"] is True
    assert len(post_data["predicted_states"]) == len(sample_seq)


def test_inference_validation():
    samples_resp = client.get("/api/v1/dataset/samples")
    sample_seq = samples_resp.json()[0]["scaled_sequence"]

    val_resp = client.post("/api/v1/inference/validate", json={"sequence": sample_seq})
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["tolerance_passed"] is True
    assert abs(val_data["abs_difference_fwd_bwd"]) < 1e-8


def test_inference_predict():
    samples_resp = client.get("/api/v1/dataset/samples")
    sample_seq = samples_resp.json()[0]["scaled_sequence"]

    pred_resp = client.post("/api/v1/inference/predict", json={"sequence": sample_seq})
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert len(pred_data["pointwise_states"]) == len(sample_seq)
    assert len(pred_data["viterbi_states"]) == len(sample_seq)


def test_csv_upload_recognition():
    csv_content = "acc_x,acc_y,acc_z\n"
    # Create 30 rows of simulated walking
    for i in range(30):
        csv_content += f"{0.1 * i},{9.8 + 0.5 * (i % 2)},{0.2}\n"

    files = {"file": ("test_gait.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    rec_resp = client.post("/api/v1/recognize", files=files)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["status"] == "SUCCESS"
    assert rec_data["total_timesteps"] == 30
    assert len(rec_data["steps"]) == 30


def test_evaluation_summary():
    response = client.get("/api/v1/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] > 0.90
    assert data["macro_f1"] > 0.90
    assert len(data["loso_folds"]) == 9
