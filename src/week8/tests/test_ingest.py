import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from src.week8.ingest import pipeline
from src.week8.rag.qdrant_store import QdrantStore, ensure_collection, upsert_chunks


def write_cases(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_case_boundaries_and_scrubbed_payloads(tmp_path, monkeypatch):
    path = write_cases(
        tmp_path,
        [
            {"id": "case_1", "text": "Contact test@example.com. " * 30},
            {"id": "case_2", "text": "BP 140/70 mmHg; pulse 130/min; SLC7A7."},
        ],
    )
    analyzer = Mock()
    analyzer.analyze.return_value = []
    monkeypatch.setattr(pipeline, "_get_analyzer", lambda: analyzer)
    embeddings = Mock()
    embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[
            SimpleNamespace(index=i, embedding=[1.0] + [0.0] * 1535)
            for i in range(len(kwargs["input"]))
        ]
    )
    monkeypatch.setattr(
        pipeline, "_get_openai", lambda: SimpleNamespace(embeddings=embeddings)
    )
    client = QdrantClient(":memory:")
    monkeypatch.setattr(client, "close", Mock())
    monkeypatch.setattr(pipeline, "_get_client", lambda: client)
    try:
        stats = pipeline.ingest_corpus(path)
        points, _ = client.scroll("capstone_chunks_v2", limit=100)
        chunks = [point.payload for point in points]
        assert stats["n_documents"] == 2
        assert stats["n_pii_flags"] == 30
        assert {c["source_id"] for c in chunks} == {"case_1", "case_2"}
        assert all(len(c["text"]) <= 800 for c in chunks)
        assert "test@example.com" not in str(chunks)
        assert (
            next(c["text"] for c in chunks if c["source_id"] == "case_2")
            == "BP 140/70 mmHg; pulse 130/min; SLC7A7."
        )
    finally:
        QdrantClient.close(client)


def test_selected_dataset_uses_full_note(tmp_path):
    path = write_cases(
        tmp_path,
        [
            {
                "idx": 2481,
                "note": "Truncated",
                "full_note": "Complete evidence.",
                "conversation": "Generated dialogue",
                "summary": "Generated summary",
            }
        ],
    )
    assert list(pipeline.read_corpus(path)) == [
        (path, "case_2481", "Complete evidence.")
    ]


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"id": "case_1", "text": "ok"}] * 2, "duplicate"),
        ([{"id": "case_1", "text": ""}], "expected"),
        ([[]], "JSON object"),
    ],
)
def test_invalid_rows_fail_before_connecting(tmp_path, rows, match, monkeypatch):
    connect = Mock()
    monkeypatch.setattr(pipeline, "_get_client", connect)
    analyzer = Mock()
    analyzer.analyze.return_value = []
    monkeypatch.setattr(pipeline, "_get_analyzer", lambda: analyzer)
    with pytest.raises(ValueError, match=match):
        pipeline.ingest_corpus(write_cases(tmp_path, rows))
    connect.assert_not_called()


def test_chunk_limits_and_clinical_numbers():
    assert max(map(len, pipeline.chunk_document("x" * 3000))) <= 800
    for size, overlap in [(0, 0), (10, 10), (10, -1)]:
        with pytest.raises(ValueError):
            pipeline.chunk_document("text", size, overlap)
    text = "2722 grams; WBC 6900; CFTR; +1 (212) 555-1234; EMP-12345"
    clean, flags = pipeline.scrub_pii(text, pii_mode="regex")
    assert clean == "2722 grams; WBC 6900; CFTR; [PHONE]; [EMP_ID]"
    assert len(flags) == 2
    assert all("matched" not in f for f in flags)


def test_embedding_upsert_and_reingest(tmp_path, monkeypatch):
    path = write_cases(tmp_path, [{"id": "case_1", "text": "Patient recovered."}])
    vector = [1.0] + [0.0] * 1535
    embeddings = Mock()
    embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=vector)]
    )
    monkeypatch.setattr(
        pipeline, "_get_openai", lambda: SimpleNamespace(embeddings=embeddings)
    )
    client = QdrantClient(":memory:")
    close = Mock()
    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(pipeline, "_get_client", lambda: client)
    analyzer = Mock()
    analyzer.analyze.return_value = []
    monkeypatch.setattr(pipeline, "_get_analyzer", lambda: analyzer)
    pipeline.ingest_corpus(path)
    close.assert_called_once()
    from src.week8.rag import qdrant_rag

    monkeypatch.setattr(qdrant_rag, "embed_query", lambda _: vector)
    hits = qdrant_rag.retrieve(QdrantStore(client, "capstone_chunks_v2"), "Outcome?")
    assert hits[0]["source_id"] == "case_1"
    assert hits[0]["text"] == "Patient recovered."
    QdrantClient.close(client)

    # Reordered/incremental uploads must not overwrite a different case.
    client = QdrantClient(":memory:")
    store = QdrantStore(client, "test", dim=2)
    ensure_collection(store)
    first = {"chunk_id": "case_1#0", "source_id": "case_1", "text": "One"}
    second = {"chunk_id": "case_2#0", "source_id": "case_2", "text": "Two"}
    upsert_chunks(store, [first, second], [[1.0, 0.0], [0.0, 1.0]])
    upsert_chunks(store, [second], [[0.0, 1.0]])
    assert client.get_collection("test").points_count == 2
    with pytest.raises(ValueError, match="embedding"):
        upsert_chunks(store, [first], [])
    client.close()


def test_hybrid_scrubber_keeps_clinical_context_and_hides_raw_pii(monkeypatch):
    analyzer, anonymizer = Mock(), Mock()
    analyzer.analyze.return_value = [SimpleNamespace(entity_type="PERSON", score=0.9)]
    anonymizer.anonymize.return_value = SimpleNamespace(
        text="<PERSON>, age 76; email [EMAIL]"
    )
    monkeypatch.setattr(pipeline, "_get_analyzer", lambda: analyzer)
    monkeypatch.setattr(pipeline, "_get_anonymizer", lambda: anonymizer)
    clean, flags = pipeline.scrub_pii("Jane Doe, age 76; email jane@example.com")
    assert clean == "<PERSON>, age 76; email [EMAIL]"
    assert (
        analyzer.analyze.call_args.kwargs["text"] == "Jane Doe, age 76; email [EMAIL]"
    )
    assert "DATE_TIME" not in analyzer.analyze.call_args.kwargs["entities"]
    assert len(flags) == 2
    assert "Jane" not in str(flags) and "jane@example.com" not in str(flags)
