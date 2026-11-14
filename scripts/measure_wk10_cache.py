"""Print cold and warm golden-set cost, latency, and cache hit counts.

Run: uv run --env-file .env python -m scripts.measure_wk10_cache
"""

import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

import tiktoken

from src.week10.rag import cache
from src.week10.rag.qdrant_rag import EMBED_MODEL, ask_rag

GOLDEN = Path("data/golden_set.jsonl")

# USD per million tokens; official model rates checked 2026-11-14.
PRICING = {"input": 0.15, "output": 0.60, "input_cached": 0.075, "embed": 0.02}


def cost_of(res: dict, q_len_tokens: int) -> float:
    if res["cache_hit"]:
        return q_len_tokens * PRICING["embed"] / 1_000_000
    cached = res["tokens_cached"]
    fresh = res["tokens_in"] - cached
    return (
        cached * PRICING["input_cached"]
        + fresh * PRICING["input"]
        + res["tokens_out"] * PRICING["output"]
        # A miss embeds once for cache lookup and again for dense retrieval.
        + 2 * q_len_tokens * PRICING["embed"]
    ) / 1_000_000



def run_pass(label, golden, tokenizer):
    latencies, cost = [], 0.0
    hits = 0
    for question in golden:
        started = perf_counter()
        res = ask_rag(question["question"])
        latencies.append((perf_counter() - started) * 1000)
        cost += cost_of(res, len(tokenizer.encode(question["question"])))
        hits += res["cache_hit"]
    latencies.sort()
    return {
        "label": label,
        "cost": cost,
        "p50_ms": median(latencies),
        "p95_ms": latencies[ceil(len(latencies) * 0.95) - 1],
        "hits": hits,
        "misses": len(golden) - hits,
    }


def main():
    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    tokenizer = tiktoken.encoding_for_model(EMBED_MODEL)

    cache.clear()
    cold = run_pass("cold", golden, tokenizer)
    warm = run_pass("warm", golden, tokenizer)

    print("\n══ W10 cost + latency ══")
    print(f"  {'Pass':<8} {'hits':>5} {'misses':>6} {'p50':>9} {'p95':>9} {'total cost':>12} {'cost/query':>12}")
    for p in (cold, warm):
        print(
            f"  {p['label']:<8} {p['hits']:>5} {p['misses']:>6} "
            f"{p['p50_ms']:>7.0f}ms {p['p95_ms']:>7.0f}ms "
            f"${p['cost']:>11.7f} ${p['cost']/len(golden):>11.9f}"
        )
    if cold["cost"]:
        saved_pct = 100 * (cold["cost"] - warm["cost"]) / cold["cost"]
        print(f"\nWarm-pass cost saved: {saved_pct:.1f}%")


if __name__ == "__main__":
    main()
