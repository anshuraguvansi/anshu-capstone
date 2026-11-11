# W8 Snapshot — Post-ingestion baseline

**Date:** 2026-11-11  
**Vector store:** Qdrant Cloud  
**Collection:** capstone_chunks_v2 (75 points, 1536 dimensions, cosine)  
**Chunking strategy:** 800-character windows with 80-character overlap  
**PII scrubbing:** regex + Presidio hybrid

## Headline numbers

| Metric | W7 (capstone_chunks) | W8 (capstone_chunks_v2) | Delta |
|---|---|---|---|
| Cost per query (embedding only) | $0.000000374 | $0.000000374 | No change |
| Latency p50 (ms) | 1047.6 | 1038.4 | −9.2 |
| Retrieval hit rate (evidence@3) | 90% (27/30) | 90% (27/30) | 0 pp |
| Grounded response rate | Not measured | Not measured | — |
| Confirmed PII items scrubbed | 0 | 0; all 28 flags were false positives | No confirmed PII removed |

## Operational deltas

| Property | W7 | W8 |
|---|---|---|
| Payload fields per chunk, including text | 3 | 12 |
| Filtered retrieval possible | Yes, by source or chunk ID | Yes, with additional metadata |
| PII handling | No scrubbing | Scrubbed at ingest, but medical terms were wrongly redacted |

## Method

- I used the same 30 golden questions as W7, with top-k 3 and at least 70% evidence-word coverage for a hit.
- W7 baseline: the 800/80 OpenAI results in [wk7-snapshot.md](wk7-snapshot.md).
- W8: a fresh run against `capstone_chunks_v2`, saved as `wk8-qdrant-openai` in `data/eval_qdrant_results.jsonl`. Cost and latency cover query embedding and retrieval; generated answers were not evaluated.

## Interpretation

Retrieval stayed the same, including the three evidence misses. Chunking stayed
the same, and I did not use metadata filters. The 9.2 ms latency difference is too
small to claim an improvement. All 28 PII flags were false positives, so I need to
fix the scrubber and re-ingest the cases. Examples are in [the PII audit](../wk8-pii-audit.md).
