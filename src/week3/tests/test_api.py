from api.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_ask_reject_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422
    body = response.json()
    detail_text = str(body.get("detail", ""))
    assert "question" in detail_text.lower()


def test_health_return_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
