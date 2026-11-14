# W9 Snapshot — Hybrid retrieval + reranking baseline

**Date:** 2026-11-13  
**Collection:** `capstone_chunks_v2` (75 points)  
**Retrieval stack:** BM25 + dense (Qdrant cosine, text-embedding-3-small) via RRF (k=60) → cross-encoder rerank (ms-marco-MiniLM-L-6-v2).

## Headline numbers

| Metric | W8 (dense-only) | W9 (hybrid+rerank) | Delta |
|---|---:|---:|---:|
| Precision@3 | 0.289 | 0.289 | +0.000 |
| Precision@5 | 0.187 | 0.193 | +0.007 |
| Chunk hit rate@3 | 86.7% | 86.7% | +0.0 pp |
| Chunk hit rate@5 | 93.3% | 96.7% | +3.3 pp |
| Grounded response rate | 93.3% | 96.7% | +3.3 pp |
| Estimated cost/query (embedding only) | $0.000000374 | $0.000000374 | +0.000000000 USD |
| Estimated cost/query (answer generation) | $0.000123610 | $0.000124025 | +0.000000415 USD |
| Estimated cost/query (retrieval + answer) | $0.000123984 | $0.000124399 | +0.000000415 USD |
| Retrieval latency p50 (ms) | 1856.9 | 2142.4 | +285.5 |
| Answer generation latency p50 (ms) | 1843.8 | 1853.6 | +9.8 |
| Retrieval + answer latency p50 (ms) | 3793.4 | 4015.2 | +221.8 |
| Abstentions | 0/30 | 0/30 | — |
| Mean accuracy (1–4) | 3.57 | 3.57 | +0.00 |
| Estimated judge cost (entire run) | $0.1049 | $0.1053 | — |

## Method

- I used the 30 golden questions and ran both retrieval methods against the same collection.
- Dense-only represents the W8 approach. Both columns are from this run; historical W8 results are in [wk8-snapshot.md](wk8-snapshot.md).
- Precision and hit rate use the chunk labels in `EXPECTED_CHUNKS`. I retrieved five chunks per question and used only the top three to generate each answer.
- Hybrid retrieves 10 candidates per retriever, fuses to 10 with RRF, and reranks to five.
- Answers use `gpt-4o-mini` at temperature 0 with the same context-only prompt. The `gpt-4o` judge scores accuracy, groundedness, and format from 1 to 4.
- Grounded response rate counts non-abstaining answers with groundedness 4/4, divided by all 30 questions. Reference answers are used for accuracy; grounding uses only the retrieved context.
- BM25 and reranker initialization took 16.8 seconds and is excluded from query latency. Retrieval latency covers query embedding and retrieval to five results; answer latency covers generation. Judging is excluded from both.
- Costs are token API estimates: embeddings at $0.02/million tokens, and generation/judging using API token usage and repository rates before caching discounts. Local compute and Qdrant charges are excluded.
- The same LLM judge and rubric are used for both methods. Judgments have not been manually audited.

## What changed structurally

W9 adds BM25, RRF, and a local cross-encoder to dense retrieval. Both methods use one query embedding; reranking adds no embedding API call.

## Interpretation

P@3 changed by +0.00 percentage points: 1 questions improved and 1 regressed.
Improved: `case_99863_treatment`.
Regressed: `case_133814_presentation`.

Grounded responses: 28/30 for dense and 29/30 for hybrid+rerank. This measures support in the retrieved context, separately from reference-answer accuracy.
Most questions have one relevant chunk, giving a maximum P@3 of 0.333 for those questions. These results compare the full stacks; separate runs are needed to attribute changes to BM25/RRF or reranking.

Per-question precision: [wk9-precision-delta.md](../wk9-precision-delta.md).
Answers, context, scores, and judge reasoning: [wk9_results.jsonl](../../data/wk9_results.jsonl).
Summary: [wk9_summary.json](../../data/wk9_summary.json).

Run: `uv run --env-file .env python scripts/measure_wk9_precision.py`.
