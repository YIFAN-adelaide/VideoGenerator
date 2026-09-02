from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_video():
    response = client.post(
        "/v1/videos",
        json={
            "prompt": "A cinematic tiger walking through a forest",
            "duration_seconds": 4,
            "fps": 24,
            "resolution": "480p",
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]
