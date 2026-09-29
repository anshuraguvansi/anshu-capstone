# Week 6 KPI Snapshot

## Activity A — Chunk-size sweep

Tested three chunking profiles against the same 30 questions in the golden set. Retrieval hit rate is based on whether the top three chunks covered at least 70% of the expected evidence. All three profiles found the correct source for 30 out of 30 questions, so evidence coverage was the useful measure for comparison.

| Profile | Size | Overlap | Hit rate | Cost / q | p95 latency |
|---------|-----:|--------:|----------|----------|-------------|
| Small   | 300  | 30      | 21 / 30 (70%) | $0.00000037 | 1.2s |
| Medium  | 500  | 50      | 24 / 30 (80%) | $0.00000037 | 1.3s |
| Large   | 800  | 80      | 27 / 30 (90%) | $0.00000037 | 0.8s |

Tested `(300/30)`, `(500/50)`, and `(800/80)`. Winner: `800/80` because it achieved the highest evidence hit rate at 90%, compared with 80% for the baseline, without increasing the cost per query. I will keep this as the Week 7 starting point and use section-aware chunking to see whether the remaining misses come from information crossing section boundaries.

The results are stored under the labels `wk6-chunk-small`, `wk6-chunk-medium`, and `wk6-chunk-large` in `data/rag_eval_results.jsonl`.

## Activity B — Local embeddings

Swapped OpenAI `text-embedding-3-small` for `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, local, and free). Both indexes used the same 500-character chunks with a 50-character overlap, and both were evaluated against all 30 golden questions.

| KPI | OpenAI baseline | Local | Difference |
|-----|----------------:|------:|------------|
| Evidence hit rate | 24 / 30 (80.0%) | 22 / 30 (73.3%) | Local was 2 hits lower (-6.7 percentage points) |
| Source hit rate | 30 / 30 (100%) | 29 / 30 (96.7%) | Local was 1 hit lower |
| Cost / query | $0.00000037 | $0 after setup | Local removed the API query cost |
| Median latency | 0.686s | 0.011s | Local was about 0.675s faster |
| Setup | API key and network access | One-time model download and load | Initial setup time was not measured |

Five question-level comparisons show that a similar cosine score does not necessarily mean the models retrieved the same case:

| Question | OpenAI top source | Local top source | Same? | Cosine OpenAI top | Cosine local top |
|----------|-------------------|------------------|:-----:|------------------:|-----------------:|
| `case_2481_outcome` | `case_2481` | `case_101886` | No | 0.655 | 0.614 |
| `case_83092_treatment` | `case_83092` | `case_2050` | No | 0.619 | 0.455 |
| `case_181392_presentation` | `case_181392` | `case_181392` | Yes | 0.644 | 0.612 |
| `case_52085_diagnosis` | `case_52085` | `case_75567` | No | 0.509 | 0.508 |
| `case_152245_outcome` | `case_152245` | `case_152245` | Yes | 0.539 | 0.528 |

For `case_181392_presentation` and `case_152245_outcome`, both models ranked the correct case first, but the local model retrieved better supporting chunks within that case. In contrast, the local model ranked the wrong case first for three of the five examples above, and it missed the expected source entirely for `case_83092_treatment`.

If the corpus were ten times larger, I would still ship the OpenAI embedding baseline for this clinical application because it met the 80% evidence-hit target while the local model was 6.7 percentage points lower. The local model is much faster and removes API cost, so I would keep it for development and revisit it after testing the winning 800/80 chunk configuration or section-aware chunking. If those changes close the quality gap, local embeddings would become the better scaling choice.

The local results are stored under `wk6-local-embeds` in `data/rag_eval_results.jsonl`.
