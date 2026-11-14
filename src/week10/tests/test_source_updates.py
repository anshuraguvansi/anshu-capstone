from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from src.week10.ingest import pipeline
from src.week10.rag import retrieval
from src.week10.rag.qdrant_store import (
    COLLECTION_NAME,
    QdrantStore,
    ensure_collection,
    upsert_chunks,
)


@pytest.fixture
def store(monkeypatch):
    client = QdrantClient(":memory:")
    store = QdrantStore(client, COLLECTION_NAME)
    ensure_collection(store)
    upsert_chunks(
        store,
        [
            {"chunk_id": "case_1#0", "source_id": "case_1", "source": "cases.jsonl", "text": "Old clinical evidence."},
            {"chunk_id": "case_1#1", "source_id": "case_1", "source": "cases.jsonl", "text": "More old evidence.", "deleted_at": None},
            {"chunk_id": "case_2#0", "source_id": "case_2", "source": "cases.jsonl", "text": "Another patient's evidence."},
        ],
        [[1.0] + [0.0] * 1535] * 3,
    )
    monkeypatch.setattr(pipeline, "scrub_pii", lambda text: (text, []))
    embeddings = Mock()
    embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=[1.0] + [0.0] * 1535) for i in range(len(kwargs["input"]))]
    )
    monkeypatch.setattr(pipeline, "_get_openai", lambda: SimpleNamespace(embeddings=embeddings))
    monkeypatch.setattr(pipeline.cache, "clear", Mock(return_value=4))
    retrieval._load_bm25.cache_clear()
    yield store
    retrieval._load_bm25.cache_clear()
    client.close()


def test_tombstones_missing_and_null_fields_without_touching_other_cases(store):
    assert pipeline.tombstone_source(store, "case_1") == 2
    assert pipeline.tombstone_source(store, "case_1") == 0
    points, _ = store.client.scroll(store.collection)
    assert all(p.payload.get("deleted_at") for p in points if p.payload["source_id"] == "case_1")
    assert all("deleted_at" not in p.payload for p in points if p.payload["source_id"] == "case_2")


def test_update_preserves_tombstones_and_week8_chunk_metadata(store, monkeypatch):
    bm25_clear = Mock()
    monkeypatch.setattr(retrieval._load_bm25, "cache_clear", bm25_clear)
    assert pipeline.ingest_or_update_source(store, "case_1", "Updated evidence.") == {
        "tombstoned": 2, "ingested": 1, "cache_cleared": 4,
    }
    assert pipeline.ingest_or_update_source(store, "case_1", "Latest evidence.", clear_cache_after=False) == {
        "tombstoned": 1, "ingested": 1, "cache_cleared": 0,
    }
    pipeline.cache.clear.assert_called_once()
    assert bm25_clear.called
    points, _ = store.client.scroll(store.collection)
    live = [p.payload for p in points if not p.payload.get("deleted_at")]
    assert len(points) == 5
    deleted = [p for p in points if p.payload.get("deleted_at")]
    assert len(deleted) == 3
    assert {p.payload["text"] for p in deleted} == {
        "Old clinical evidence.", "More old evidence.", "Updated evidence.",
    }
    assert {p["text"] for p in live} == {"Latest evidence.", "Another patient's evidence."}
    updated = next(p for p in points if p.payload["text"] == "Latest evidence.")
    from uuid import UUID

    assert UUID(updated.id).version == 4
    assert updated.payload["chunk_id"] == "case_1#0"
    expected = pipeline.enrich_metadata(
        "Latest evidence.", pipeline.Path("data/corpus/cases.jsonl"), 0, 0,
        source_id="case_1",
    )
    assert updated.payload.keys() == expected.keys()
    for key in expected.keys() - {"ingested_at"}:
        assert updated.payload[key] == expected[key]


def test_embedding_failure_leaves_existing_case_unchanged(store, monkeypatch):
    embeddings = Mock()
    embeddings.create.side_effect = RuntimeError("API unavailable")
    monkeypatch.setattr(pipeline, "_get_openai", lambda: SimpleNamespace(embeddings=embeddings))
    with pytest.raises(RuntimeError):
        pipeline.ingest_or_update_source(store, "case_1", "Updated evidence.")
    points, _ = store.client.scroll(store.collection)
    assert len(points) == 3
    assert all(not p.payload.get("deleted_at") for p in points)
    pipeline.cache.clear.assert_not_called()


def test_retrieval_excludes_tombstones(store, monkeypatch):
    pipeline.tombstone_source(store, "case_1")
    monkeypatch.setattr(store.client, "close", Mock())
    monkeypatch.setattr(retrieval, "_get_client", lambda: store.client)
    monkeypatch.setattr(retrieval, "_get_openai", pipeline._get_openai)
    for search in (retrieval.bm25_search, retrieval.dense_search):
        hits = search("clinical evidence")
        assert hits
        assert {hit["doc"]["source_id"] for hit in hits} == {"case_2"}
    pipeline.tombstone_source(store, "case_2")
    assert retrieval.bm25_search("clinical evidence") == []
    assert retrieval.dense_search("clinical evidence") == []
    assert retrieval.rerank("clinical evidence", []) == []
