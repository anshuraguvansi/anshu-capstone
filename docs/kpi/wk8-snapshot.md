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
| Grounded response rate | Not measured | 96.7% (29/30; follow-up 2026-11-13) | — |
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
- W8: a fresh run against `capstone_chunks_v2`, saved as `wk8-qdrant-openai` in `data/eval_qdrant_results.jsonl`. Cost and latency cover query embedding and retrieval; generated answers were not evaluated in that run.

## Interpretation

Retrieval stayed the same, including the three evidence misses. Chunking stayed
the same, and I did not use metadata filters. The 9.2 ms latency difference is too
small to claim an improvement. All 28 PII flags were false positives, so I need to
fix the scrubber and re-ingest the cases. Examples are in [the PII audit](../wk8-pii-audit.md).

## Grounded response evaluation — 2026-11-13

I re-ran the 30 golden questions using Week 8 dense retrieval from `capstone_chunks_v2`
(75 points), top-3 context, and `gpt-4o-mini` at temperature 0.
The `gpt-4o` judge scores accuracy, groundedness, and format from 1 to 4.
A grounded response scores 4/4 for groundedness and is not an abstention.
The denominator includes all 30 questions. The judge checks support against the
retrieved context; the reference answer is used for accuracy only.

| Metric | Result |
|---|---:|
| Grounded response rate | 96.7% (29/30) |
| Abstentions | 0/30 |
| Mean accuracy (1–4) | 3.57 |
| Retrieval latency p50 (ms) | 1005.7 |
| Answer generation latency p50 (ms) | 1861.9 |
| Retrieval + answer latency p50 (ms) | 3087.6 |
| Estimated embedding cost/query | $0.000000374 |
| Estimated answer generation cost/query | $0.000123 |
| Estimated retrieval + answer cost/query | $0.000124 |
| Estimated judge cost (entire run) | $0.1042 |

Latency excludes judging; retrieval timing includes query embedding and Qdrant search.
Costs cover token API charges only. Embedding cost uses tokenized queries at
$0.02/million tokens; generation and judge estimates use API token usage and the
repository's model rates, before caching discounts. Qdrant and local compute costs
are excluded. LLM judgments have not been manually audited.

Answers, retrieved context, token usage, and judge reasoning:
[`data/wk8_grounded_results.jsonl`](../../data/wk8_grounded_results.jsonl).
Summary: [`data/wk8_grounded_summary.json`](../../data/wk8_grounded_summary.json).
Run: `uv run --env-file .env python scripts/measure_wk8_grounding.py`.
