"""Evaluate Qdrant retrieval using the same metrics as the Week 6 evaluator."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_rag_eval import (  # noqa: E402
    DEFAULT_EMBEDDING_COST_PER_MILLION,
    DEFAULT_GOLDEN_SET,
    count_tokens,
    get_embedder,
    load_jsonl,
    measure_evidence,
    parse_bool,
    save_results,
    validate_golden,
)
from src.week7.rag.qdrant_store import load_store  # noqa: E402

DEFAULT_RESULTS = Path("data/eval_qdrant_results.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--index", type=Path, required=True,
        help="Original cache for model and chunking metadata; retrieval uses Qdrant.",
    )
    parser.add_argument("--local", type=parse_bool, default=False, metavar="true|false")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--evidence-threshold", type=float, default=0.70)
    parser.add_argument(
        "--embedding-cost-per-million", type=float,
        default=DEFAULT_EMBEDDING_COST_PER_MILLION,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict]:
    if args.top_k <= 0 or not 0 <= args.evidence_threshold <= 1:
        raise ValueError("Use a positive --top-k and an evidence threshold between 0 and 1.")
    if args.embedding_cost_per_million < 0:
        raise ValueError("Embedding cost cannot be negative.")

    metadata = json.loads(args.index.read_text(encoding="utf-8"))["metadata"]
    embedding_model, embed_many = get_embedder(args.local)
    if metadata["embedding_model"] != embedding_model:
        raise ValueError("The cache model does not match --local. Select the matching cache and model.")
    golden = load_jsonl(args.golden)
    if not golden:
        raise ValueError("The golden set is empty.")
    for position, entry in enumerate(golden, start=1):
        validate_golden(entry, position)

    results = []
    store = load_store(args.collection, dim=metadata["embedding_dimension"])
    try:
        # Check the existing collection without creating or modifying it.
        from qdrant_client.models import Distance, VectorParams

        config = store.client.get_collection(store.collection).config.params.vectors
        if (
            not isinstance(config, VectorParams)
            or config.size != store.dim
            or config.distance != Distance.COSINE
        ):
            raise ValueError("Collection dimensions or distance metric do not match the cache.")

        for entry in golden:
            question = entry["question"]
            started = time.perf_counter()
            query_vector = embed_many([question])[0]
            hits = store.client.query_points(
                collection_name=store.collection,
                query=query_vector,
                limit=args.top_k,
                with_payload=True,
            ).points
            latency = time.perf_counter() - started
            retrieved = [{**hit.payload, "score": hit.score} for hit in hits]
            source_hit, coverage = measure_evidence(
                entry["evidence"], entry["expected_source"], retrieved
            )
            tokens = 0 if args.local else count_tokens(question, embedding_model)
            results.append({
                "label": args.label,
                "golden_id": entry["id"],
                "question": question,
                "expected_source": entry["expected_source"],
                "source_hit": source_hit,
                "evidence_hit": source_hit and coverage >= args.evidence_threshold,
                "evidence_coverage": round(coverage, 6),
                "top_k": args.top_k,
                "latency_seconds": round(latency, 6),
                "embedding_tokens": tokens,
                "embedding_model": embedding_model,
                "local": args.local,
                "cost_usd": tokens / 1_000_000 * args.embedding_cost_per_million,
                "index": str(args.index),
                "collection": args.collection,
                "backend": "qdrant",
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": hit["chunk_id"],
                        "source_id": hit["source_id"],
                        "score": round(hit["score"], 6),
                    }
                    for rank, hit in enumerate(retrieved, start=1)
                ],
            })
            print(f"{entry['id']}: source_hit={source_hit}, evidence_hit={results[-1]['evidence_hit']}")
    finally:
        store.client.close()

    save_results(args.results, args.label, results)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run(args)
    total = len(results)
    print(f"\nCollection: {args.collection}")
    print(f"Source hit rate@{args.top_k}: {sum(r['source_hit'] for r in results)}/{total}")
    print(f"Evidence hit rate@{args.top_k}: {sum(r['evidence_hit'] for r in results)}/{total}")
    print(f"Median embedding + retrieval latency: {statistics.median(r['latency_seconds'] for r in results):.3f}s")
    print(f"Total query embedding cost: ${sum(r['cost_usd'] for r in results):.8f}")
    print(f"Results: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
