# Week 7 KPI Snapshot

**Date:** 2026-10-16

## Qdrant migration

Reused saved Week 6 embeddings without new document embedding calls. `/ask_batched` now uses Qdrant retrieval and saves answers and sources to SQLite.

| Collection | Model | Chunk size / overlap | Points | Dimensions |
|---|---|---|---:|---:|
| `capstone_chunks` | OpenAI 3-small | 800 / 80 | 75 | 1536 |
| `capstone_chunks_openai_medium` | OpenAI 3-small | 500 / 50 | 116 | 1536 |
| `capstone_chunks_local` | MiniLM-L6-v2 | 500 / 50 | 116 | 384 |

Evaluated the same 30 golden questions with top-k 3 and cosine distance. An evidence hit requires the expected source and at least 70% evidence-word coverage. Latency includes query embedding and retrieval; cost includes query embedding API charges only.

| Configuration | W6 evidence hits → W7 | W6 source hits → W7 | W6 median latency → W7 |
|---|---|---|---|
| OpenAI, 800/80 | 27/30 → 27/30 | 30/30 → 30/30 | 678.7 → 1047.6 ms |
| MiniLM, 500/50 | 22/30 → 22/30 | 29/30 → 29/30 | 10.6 → 276.0 ms |

Migration preserved aggregate hit rates. Qdrant was slower on this small corpus. One OpenAI top-three chunk list changed without changing the hit/miss result; all local lists matched Week 6.

## Activity — hosted vs local embeddings

Compared identical 500/50 chunks in `capstone_chunks_openai_medium` and `capstone_chunks_local`.

| KPI | OpenAI 3-small | MiniLM-L6-v2 |
|---|---:|---:|
| Source hit rate@3 | 30/30 (100%) | 29/30 (96.7%) |
| Evidence hit rate@3 | 24/30 (80%) | 22/30 (73.3%) |
| Median embedding + retrieval latency | 966.5 ms | 276.0 ms |
| Mean embedding API cost / query | $0.000000374 | $0 |
| Raw float32 vector storage, 116 chunks | 712,704 bytes | 178,176 bytes |
| First-run setup time | Not measured | Not measured |

MiniLM uses normalized embeddings. Its raw vectors need 75% less storage, excluding payloads and indexes. At 10,000 queries/day, the measured OpenAI query embedding cost is approximately $0.00374/day, excluding generation and hosting.

### Per-question findings

| Question ID | Finding |
|---|---|
| `case_2481_outcome` | MiniLM ranks the wrong case first, but both models find the expected evidence within the top three. |
| `case_83092_treatment` | Evidence coverage: OpenAI 100%, MiniLM 48%. |
| `case_52085_diagnosis` | OpenAI finds the expected case; MiniLM misses it entirely. |
| `case_181392_presentation` | Evidence coverage: OpenAI 69.0%, MiniLM 100%. |
| `case_101886_outcome` | Both rank the correct case first; evidence coverage is OpenAI 100%, MiniLM 27.3%. |

### Verdict

Keep OpenAI 3-small as the default. On identical chunks it achieves two more evidence hits and one more source hit than MiniLM. MiniLM is faster and avoids embedding API charges, but the savings are small at this project scale. Retain 800/80 chunking for the default pipeline, which achieves 90% evidence hit rate.

Local embedding computation avoids sending text to an embedding provider. The current application still sends chunk text to Qdrant and retrieved context to OpenAI for generation.

## Results and remaining work

Results: `data/eval_qdrant_results.jsonl`, under `wk7-qdrant-openai`, `wk7-qdrant-openai-medium`, and `wk7-qdrant-local`. Week 6 baselines: `data/rag_eval_results.jsonl`.

The current choice is recorded in `docs/adr/0001-capstone-framing.md`. Remaining: compare OpenAI 3-small versus 3-large on identical chunks and revisit the decision using those results.
