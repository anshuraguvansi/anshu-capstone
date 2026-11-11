import json
import sqlite3
from unittest.mock import Mock

from src.week7.api import main
from fastapi.testclient import TestClient


client = TestClient(main.app)


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


def test_ask_batched_uses_qdrant_and_saves_sources(monkeypatch, tmp_path):
    db_path = tmp_path / "results.db"
    monkeypatch.setattr(main, "_db_path", db_path)
    ask_rag = Mock(return_value={
        "answer": "Treatment was successful [case_2481].",
        "sources": ["case_2481"],
        "tokens_in": 100,
        "tokens_out": 20,
    })
    monkeypatch.setattr(main.qdrant_rag, "ask_rag", ask_rag)

    response = client.post("/ask_batched", json={"question": "What was the outcome?"})

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "Treatment was successful [case_2481]."
    assert body["sources"] == ["case_2481"]
    assert body["cost_usd"] == main.compute_cost_usd(main._settings.model, 100, 20)
    ask_rag.assert_called_once_with("What was the outcome?", settings=main._settings)
    with sqlite3.connect(db_path) as conn:
        content, sources, model = conn.execute(
            "SELECT content, sources_json, model FROM answers"
        ).fetchone()
    assert content == body["content"]
    assert json.loads(sources) == body["sources"]
    assert model == main._settings.model


def test_ask_batched_rejects_empty_question():
    assert client.post("/ask_batched", json={"question": ""}).status_code == 422
