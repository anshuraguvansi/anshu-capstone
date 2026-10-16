import json
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from scripts import eval_qdrant, run_rag_eval
from src.week7.rag.qdrant_store import QdrantStore, ensure_collection, upsert_chunks


@pytest.mark.parametrize("local,dim", [(False, 1536), (True, 384)])
def test_qdrant_matches_json_metrics_and_preserves_old_results(monkeypatch, tmp_path, local, dim):
    model = "all-MiniLM-L6-v2" if local else "text-embedding-3-small"
    vector = [1.0] + [0.0] * (dim - 1)
    chunk = {"chunk_id": "case_1#0", "source_id": "case_1", "text": "Patient recovered.", "embedding": vector}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"metadata": {
        "embedding_model": model, "embedding_dimension": dim,
        "chunk_size": 500, "chunk_overlap": 50,
    }, "chunks": [chunk]}))
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({
        "id": "outcome", "question": "What was the outcome?",
        "expected_source": "case_1", "evidence": "Patient recovered.",
    }) + "\n")
    output = tmp_path / "results.jsonl"
    output.write_text('{"label": "wk6-existing"}\n')
    embed = Mock(return_value=[vector])
    monkeypatch.setattr(run_rag_eval, "get_embedder", lambda _: (model, embed))
    monkeypatch.setattr(eval_qdrant, "get_embedder", lambda _: (model, embed))
    # Tokenizer initialization can download vocabulary; keep this test offline.
    monkeypatch.setattr(run_rag_eval, "count_tokens", lambda *args: 6)
    monkeypatch.setattr(eval_qdrant, "count_tokens", lambda *args: 6)
    common = ["--index", str(index), "--golden", str(golden), "--results", str(output), "--local", str(local)]
    baseline = run_rag_eval.run(run_rag_eval.parse_args(["--label", "json", *common]))
    old_results = output.read_text()
    qdrant_output = tmp_path / "eval_qdrant_results.jsonl"
    qdrant_output.write_text('{"label": "other-qdrant-run"}\n')

    client = QdrantClient(":memory:")
    store = QdrantStore(client, "test_collection", dim=dim)
    ensure_collection(store)
    upsert_chunks(store, [chunk], [vector])
    monkeypatch.setattr(eval_qdrant, "load_store", lambda *a, **kw: store)
    close = Mock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    args = eval_qdrant.parse_args(["--label", "qdrant", "--collection", store.collection, *common])
    args.results = qdrant_output
    results = eval_qdrant.run(args)

    for field in ("source_hit", "evidence_hit", "evidence_coverage", "cost_usd", "embedding_model"):
        assert results[0][field] == baseline[0][field]
    assert results[0]["source_hit"] is True
    assert results[0]["collection"] == store.collection
    assert results[0]["retrieved"][0]["source_id"] == "case_1"
    assert output.read_text() == old_results
    assert {r["label"] for r in run_rag_eval.load_jsonl(qdrant_output)} == {"other-qdrant-run", "qdrant"}
    eval_qdrant.save_results(qdrant_output, args.label, results)
    assert len(run_rag_eval.load_jsonl(qdrant_output)) == 2
    close.assert_called_once()


def test_model_mismatch_rejected_before_connecting(monkeypatch, tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"metadata": {"embedding_model": "all-MiniLM-L6-v2"}}))
    monkeypatch.setattr(eval_qdrant, "get_embedder", lambda _: ("text-embedding-3-small", Mock()))
    connect = Mock()
    monkeypatch.setattr(eval_qdrant, "load_store", connect)
    args = eval_qdrant.parse_args(["--label", "test", "--index", str(index), "--collection", "test"])
    with pytest.raises(ValueError, match="does not match"):
        eval_qdrant.run(args)
    connect.assert_not_called()
