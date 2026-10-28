
import joblib
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

artifact = joblib.load("models/fraud_detection_model.joblib")
feature_names = artifact["feature_names"]


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_valid_prediction():
    features = {name: 0.0 for name in feature_names}
    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1


def test_missing_feature():
    features = {name: 0.0 for name in feature_names}
    features.pop(feature_names[0])

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422
