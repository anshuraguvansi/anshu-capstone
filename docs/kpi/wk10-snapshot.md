# W10 Snapshot — Caching + lifecycle baseline

**Date:** 2026-11-14  
**Stack:** W9 hybrid retrieval + reranking, with a SQLite semantic cache at a 0.95 threshold.  
**Update flow:** tombstone old chunks → insert fresh points → clear answer and BM25 caches.

## Results

| Metric | W9 (no cache) | W10 cold | W10 warm | Warm change vs W9 |
|---|---:|---:|---:|---|
| Average cost/query | $0.000124399 | $0.000125003 | $0.000000374 | 99.7% lower |
| Latency p50 | 4015 ms | 4296 ms | 707 ms | 82.4% lower |
| Latency p95 | 5563 ms | 5699 ms | 797 ms | 85.7% lower |
| Cache hits | n/a | 0/30 (0%) | 30/30 (100%) | All repeated queries hit |
| Precision@3 | 0.289 | Not remeasured | Not remeasured | — |
| Grounded response rate | 96.7% | Not remeasured | Not remeasured | — |

Total cost for 30 queries: **$0.0037501 cold** and **$0.0000112 warm**.

## Method

I ran the same 30 clinical golden-set questions twice. I cleared the cache before the cold pass, then immediately repeated the questions for the warm pass.

Costs use GPT-4o-mini at $0.15/M input, $0.075/M cached input and $0.60/M output, plus text-embedding-3-small at $0.02/M tokens. A cache miss currently embeds the question twice; a hit embeds it once. These estimates exclude Qdrant and local compute costs.

The first cold query includes BM25 and reranker initialization. W9 excluded initialization, so the latency comparison is approximate. I did not rerun precision or groundedness evaluation.

Run: `uv run --env-file .env python -m scripts.measure_wk10_cache`

## Interpretation

The warm pass saved **99.7%** compared with the cold pass. This is above the expected 60–70% range because all 30 questions were repeated and every warm query hit the cache, avoiding retrieval and answer generation. Production savings will depend on how often similar questions repeat.

I also verified the single-case update flow with `case_2481`: each run tombstoned 7 live chunks and inserted 7 fresh points. The old points remain visible with `deleted_at`, and retrieval filters them out.

## Known risks

- Similar questions could incorrectly reuse an answer. A 100% hit rate does not prove answer quality; I still need to sample hits and compare them with full RAG answers.
- Every document update path must clear stale answers. The single-case update clears both caches; bulk ingestion still needs cache invalidation.
