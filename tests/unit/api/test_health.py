from fastapi.testclient import TestClient

from sales_agent.api.app import app


def test_liveness_health() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_health_does_not_expose_api_key() -> None:
    response = TestClient(app).get("/health/model")

    assert response.status_code == 200
    assert set(response.json()) == {"status", "configured", "model"}
    assert "api_key" not in response.text.lower()
