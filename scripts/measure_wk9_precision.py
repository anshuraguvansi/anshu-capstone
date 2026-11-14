"""Compare dense and hybrid retrieval using Week 9 answer-bearing chunk labels."""

import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from src.week5.eval.judge import JUDGE_MODEL, score_llm_judge  # noqa: E402
from src.week9.pipeline.cost import compute_cost_usd  # noqa: E402
from src.week9.rag import retrieval  # noqa: E402
from src.week9.rag.qdrant_rag import SYSTEM  # noqa: E402

GOLDEN_SET = ROOT / "data/golden_set.jsonl"
COLLECTION_NAME = "capstone_chunks_v2"
OUTPUT = ROOT / "docs/wk9-precision-delta.md"
RESULTS = ROOT / "data/wk9_results.jsonl"
SUMMARY = ROOT / "data/wk9_summary.json"
SNAPSHOT = ROOT / "docs/kpi/wk9-snapshot.md"
ANSWER_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBEDDING_RATE_PER_MILLION = 0.02

# Answer-bearing chunks after PII scrubbing and 800/80 chunking.
EXPECTED_CHUNKS = {
    "case_2481_presentation": ["case_2481#0"],
    "case_2481_diagnosis": ["case_2481#2"],
    "case_2481_outcome": ["case_2481#5"],
    "case_99863_presentation": ["case_99863#0"],
    "case_99863_diagnosis": ["case_99863#3"],
    "case_99863_treatment": ["case_99863#4"],
    "case_2050_presentation": ["case_2050#0"],
    "case_2050_diagnosis": ["case_2050#4"],
    "case_2050_outcome": ["case_2050#6"],
    "case_75567_presentation": ["case_75567#0"],
    "case_75567_diagnosis": ["case_75567#1"],
    "case_75567_outcome": ["case_75567#7", "case_75567#8"],
    "case_83092_presentation": ["case_83092#0"],
    "case_83092_diagnosis": ["case_83092#4"],
    "case_83092_treatment": ["case_83092#6"],
    "case_181392_presentation": ["case_181392#0"],
    "case_181392_diagnosis": ["case_181392#4"],
    "case_181392_treatment": ["case_181392#5"],
    "case_52085_presentation": ["case_52085#0"],
    "case_52085_diagnosis": ["case_52085#4"],
    "case_52085_outcome": ["case_52085#5"],
    "case_101886_presentation": ["case_101886#0"],
    "case_101886_diagnosis": ["case_101886#1"],
    "case_101886_outcome": ["case_101886#5"],
    "case_133814_presentation": ["case_133814#1"],
    "case_133814_diagnosis": ["case_133814#9"],
    "case_133814_outcome": ["case_133814#9"],
    "case_152245_presentation": ["case_152245#1"],
    "case_152245_diagnosis": ["case_152245#0"],
    "case_152245_outcome": ["case_152245#7"],
}


def load_golden(path):
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        for field in (
            "id",
            "question",
            "expected_answer",
            "expected_source",
            "evidence",
        ):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{path}:{line_number}: missing or invalid {field}")
        expected = EXPECTED_CHUNKS.get(record["id"])
        if not expected:
            raise ValueError(f"Missing Week 9 chunk labels for {record['id']}")
        if any(
            not chunk_id.startswith(record["expected_source"] + "#")
            for chunk_id in expected
        ):
            raise ValueError(
                f"Week 9 chunk labels have the wrong source for {record['id']}"
            )
        records.append(record)
    if not records:
        raise ValueError(f"Golden set is empty: {path}")
    if len({q["id"] for q in records}) != len(records):
        raise ValueError("Golden question IDs must be unique")
    return records


def precision_at_k(hits, expected_chunk_ids, k):
    """Relevant unique chunks in the top k / k; missing results count as misses."""
    retrieved_ids = {hit["doc"]["chunk_id"] for hit in hits[:k]}
    return len(retrieved_ids & set(expected_chunk_ids)) / k


def hit_at_k(hits, expected_chunk_ids, k):
    """Whether at least one relevant chunk appears in the top k."""
    retrieved_ids = {hit["doc"]["chunk_id"] for hit in hits[:k]}
    return float(bool(retrieved_ids & set(expected_chunk_ids)))


async def run_and_score(retriever_fn, label, golden, client, tokenizer, output):
    rows = []
    for index, question in enumerate(golden, 1):
        expected = EXPECTED_CHUNKS[question["id"]]
        started = perf_counter()
        hits = retriever_fn(question["question"], k=5)
        retrieval_ms = (perf_counter() - started) * 1000
        context = "\n\n".join(
            f"[{hit['doc']['chunk_id']}] (source: {hit['doc']['source_id']})\n{hit['doc']['text']}"
            for hit in hits[:3]
        )
        if not context.strip():
            raise ValueError(f"No retrieved context for {question['id']}")
        started = perf_counter()
        response = await client.chat.completions.create(
            model=ANSWER_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nQuestion: {question['question']}",
                },
            ],
        )
        answer_ms = (perf_counter() - started) * 1000
        answer = response.choices[0].message.content
        if not answer or response.usage is None:
            raise ValueError(f"Missing answer or token usage for {question['id']}")
        judgment = await score_llm_judge(
            client,
            question=question["question"],
            candidate_answer=answer,
            ideal_answer=question["expected_answer"],
            context=context,
            format_requirements="Cite the source_id in square brackets after factual claims.",
        )
        tokens = len(tokenizer.encode(question["question"]))
        row = {
            "label": label,
            "id": question["id"],
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "expected_chunk_ids": expected,
            "chunk_ids": [hit["doc"]["chunk_id"] for hit in hits],
            "retrieved": hits,
            "context": context,
            "answer": answer,
            "answer_model": ANSWER_MODEL,
            "retrieval_latency_ms": retrieval_ms,
            "answer_latency_ms": answer_ms,
            "embedding_tokens": tokens,
            "embedding_cost_usd": tokens / 1_000_000 * EMBEDDING_RATE_PER_MILLION,
            "answer_prompt_tokens": response.usage.prompt_tokens,
            "answer_completion_tokens": response.usage.completion_tokens,
            "answer_cost_usd": compute_cost_usd(
                ANSWER_MODEL,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            ),
            "judge": {
                **asdict(judgment),
                "evaluation": judgment.evaluation.model_dump(),
            },
            "grounded_response": judgment.grounded_response,
        }
        for k in (3, 5):
            row[f"p_at_{k}"] = precision_at_k(hits, expected, k)
            row[f"hit_at_{k}"] = hit_at_k(hits, expected, k)
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        rows.append(row)
        print(
            f"{label}: {index}/{len(golden)} {question['id']}, grounded={row['grounded_response']}",
            flush=True,
        )
    return {
        "label": label,
        "per_question": rows,
        **{
            metric: mean(row[metric] for row in rows)
            for metric in ("p_at_3", "p_at_5", "hit_at_3", "hit_at_5")
        },
        "grounded_responses": sum(row["grounded_response"] for row in rows),
        "grounded_response_rate": mean(row["grounded_response"] for row in rows),
        "abstentions": sum(row["judge"]["evaluation"]["abstained"] for row in rows),
        "mean_accuracy": mean(row["judge"]["evaluation"]["accuracy"] for row in rows),
        "retrieval_latency_p50_ms": median(row["retrieval_latency_ms"] for row in rows),
        "answer_latency_p50_ms": median(row["answer_latency_ms"] for row in rows),
        "retrieval_and_answer_latency_p50_ms": median(
            row["retrieval_latency_ms"] + row["answer_latency_ms"] for row in rows
        ),
        "embedding_cost_per_query_usd": mean(row["embedding_cost_usd"] for row in rows),
        "answer_cost_per_query_usd": mean(row["answer_cost_usd"] for row in rows),
        "retrieval_and_answer_cost_per_query_usd": mean(
            row["embedding_cost_usd"] + row["answer_cost_usd"] for row in rows
        ),
        "judge_cost_total_usd": sum(row["judge"]["estimated_cost_usd"] for row in rows),
    }


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_snapshot(summary, baseline, hybrid):
    count = summary["questions"]
    lines = [
        "# W9 Snapshot — Hybrid retrieval + reranking baseline",
        "",
        f"**Date:** {summary['date']}  ",
        f"**Collection:** `{COLLECTION_NAME}` ({summary['points']} points)  ",
        "**Retrieval stack:** BM25 + dense (Qdrant cosine, text-embedding-3-small) via RRF (k=60) → cross-encoder rerank (ms-marco-MiniLM-L-6-v2).",
        "",
        "## Headline numbers",
        "",
        "| Metric | W8 (dense-only) | W9 (hybrid+rerank) | Delta |",
        "|---|---:|---:|---:|",
    ]
    for title, metric in (("Precision@3", "p_at_3"), ("Precision@5", "p_at_5")):
        delta = hybrid[metric] - baseline[metric]
        lines.append(
            f"| {title} | {baseline[metric]:.3f} | {hybrid[metric]:.3f} | {delta:+.3f} |"
        )
    for title, metric in (
        ("Chunk hit rate@3", "hit_at_3"),
        ("Chunk hit rate@5", "hit_at_5"),
        ("Grounded response rate", "grounded_response_rate"),
    ):
        delta = (hybrid[metric] - baseline[metric]) * 100
        lines.append(
            f"| {title} | {baseline[metric]:.1%} | {hybrid[metric]:.1%} | {delta:+.1f} pp |"
        )
    for title, metric in (
        ("Estimated cost/query (embedding only)", "embedding_cost_per_query_usd"),
        ("Estimated cost/query (answer generation)", "answer_cost_per_query_usd"),
        (
            "Estimated cost/query (retrieval + answer)",
            "retrieval_and_answer_cost_per_query_usd",
        ),
    ):
        delta = hybrid[metric] - baseline[metric]
        lines.append(
            f"| {title} | ${baseline[metric]:.9f} | ${hybrid[metric]:.9f} | {delta:+.9f} USD |"
        )
    for title, metric in (
        ("Retrieval latency p50 (ms)", "retrieval_latency_p50_ms"),
        ("Answer generation latency p50 (ms)", "answer_latency_p50_ms"),
        ("Retrieval + answer latency p50 (ms)", "retrieval_and_answer_latency_p50_ms"),
    ):
        delta = hybrid[metric] - baseline[metric]
        lines.append(
            f"| {title} | {baseline[metric]:.1f} | {hybrid[metric]:.1f} | {delta:+.1f} |"
        )
    lines.extend(
        [
            f"| Abstentions | {baseline['abstentions']}/{count} | {hybrid['abstentions']}/{count} | — |",
            f"| Mean accuracy (1–4) | {baseline['mean_accuracy']:.2f} | {hybrid['mean_accuracy']:.2f} | {hybrid['mean_accuracy'] - baseline['mean_accuracy']:+.2f} |",
            f"| Estimated judge cost (entire run) | ${baseline['judge_cost_total_usd']:.4f} | ${hybrid['judge_cost_total_usd']:.4f} | — |",
            "",
            "## Method",
            "",
            f"- I used the {count} golden questions and ran both retrieval methods against the same collection.",
            "- Dense-only represents the W8 approach. Both columns are from this run; historical W8 results are in [wk8-snapshot.md](wk8-snapshot.md).",
            "- Precision and hit rate use the chunk labels in `EXPECTED_CHUNKS`. I retrieved five chunks per question and used only the top three to generate each answer.",
            "- Hybrid retrieves 10 candidates per retriever, fuses to 10 with RRF, and reranks to five.",
            f"- Answers use `{ANSWER_MODEL}` at temperature 0 with the same context-only prompt. The `{JUDGE_MODEL}` judge scores accuracy, groundedness, and format from 1 to 4.",
            f"- Grounded response rate counts non-abstaining answers with groundedness 4/4, divided by all {count} questions. Reference answers are used for accuracy; grounding uses only the retrieved context.",
            f"- BM25 and reranker initialization took {summary['hybrid_initialization_s']:.1f} seconds and is excluded from query latency. Retrieval latency covers query embedding and retrieval to five results; answer latency covers generation. Judging is excluded from both.",
            "- Costs are token API estimates: embeddings at $0.02/million tokens, and generation/judging using API token usage and repository rates before caching discounts. Local compute and Qdrant charges are excluded.",
            "- The same LLM judge and rubric are used for both methods. Judgments have not been manually audited.",
            "",
            "## What changed structurally",
            "",
            "W9 adds BM25, RRF, and a local cross-encoder to dense retrieval. Both methods use one query embedding; reranking adds no embedding API call.",
            "",
            "## Interpretation",
            "",
        ]
    )
    improved, regressed = [], []
    for dense, new in zip(baseline["per_question"], hybrid["per_question"]):
        if new["p_at_3"] > dense["p_at_3"]:
            improved.append(new["id"])
        elif new["p_at_3"] < dense["p_at_3"]:
            regressed.append(new["id"])
    lines.append(
        f"P@3 changed by {(hybrid['p_at_3'] - baseline['p_at_3']) * 100:+.2f} percentage points: {len(improved)} questions improved and {len(regressed)} regressed."
    )
    if improved:
        lines.append("Improved: " + ", ".join(f"`{qid}`" for qid in improved) + ".")
    if regressed:
        lines.append("Regressed: " + ", ".join(f"`{qid}`" for qid in regressed) + ".")
    lines.extend(
        [
            "",
            f"Grounded responses: {baseline['grounded_responses']}/{count} for dense and {hybrid['grounded_responses']}/{count} for hybrid+rerank. This measures support in the retrieved context, separately from reference-answer accuracy.",
            "Most questions have one relevant chunk, giving a maximum P@3 of 0.333 for those questions. These results compare the full stacks; separate runs are needed to attribute changes to BM25/RRF or reranking.",
            "",
            "Per-question precision: [wk9-precision-delta.md](../wk9-precision-delta.md).",
            "Answers, context, scores, and judge reasoning: [wk9_results.jsonl](../../data/wk9_results.jsonl).",
            "Summary: [wk9_summary.json](../../data/wk9_summary.json).",
            "",
            "Run: `uv run --env-file .env python scripts/measure_wk9_precision.py`.",
        ]
    )
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text("\n".join(lines) + "\n")


async def main():
    golden = load_golden(GOLDEN_SET)

    retrieval.COLLECTION_NAME = COLLECTION_NAME
    retrieval._load_bm25.cache_clear()
    tokenizer = tiktoken.encoding_for_model(EMBED_MODEL)
    run_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    print(f"Loaded {len(golden)} questions; collection={COLLECTION_NAME}", flush=True)
    print("Initializing BM25 and reranker...", flush=True)
    started = perf_counter()
    retrieval._load_bm25()
    retrieval._get_reranker().predict([("warmup", "warmup")])
    initialization_s = perf_counter() - started
    client = retrieval._get_client()
    try:
        point_count = client.get_collection(COLLECTION_NAME).points_count
    finally:
        client.close()
    partial = RESULTS.with_suffix(".jsonl.tmp")
    try:
        async with AsyncOpenAI() as api_client:
            with partial.open("w", encoding="utf-8") as output:
                baseline = await run_and_score(
                    retrieval.dense_search,
                    "Dense-only baseline",
                    golden,
                    api_client,
                    tokenizer,
                    output,
                )
                hybrid = await run_and_score(
                    retrieval.retrieve,
                    "W9 hybrid+rerank",
                    golden,
                    api_client,
                    tokenizer,
                    output,
                )
    finally:
        if retrieval._openai is not None:
            retrieval._openai.close()
            retrieval._openai = None
    partial.replace(RESULTS)

    print("\n══ Precision comparison ══")
    print(
        f"  {'Metric':<10s} {'W8 baseline':>12s} {'W9 hybrid+rerank':>18s} {'Delta':>10s}"
    )
    print(
        f"  {'------':<10s} {'-----------':>12s} {'-----------------':>18s} {'-----':>10s}"
    )
    for metric in ("p_at_3", "p_at_5"):
        delta = hybrid[metric] - baseline[metric]
        print(
            f"  {metric:<10s} {baseline[metric]:>12.3f} {hybrid[metric]:>18.3f} {delta:>+10.3f}"
        )

    # Save per-question precision deltas.
    lines = [
        "# W9 Precision Delta — per-question breakdown",
        "",
        "| Q# | Question | W8 P@3 | W9 P@3 | Change |",
        "|---|---|---|---|---|",
    ]
    for index, (question, dense, new) in enumerate(
        zip(golden, baseline["per_question"], hybrid["per_question"]), 1
    ):
        p8_3, p9_3 = dense["p_at_3"], new["p_at_3"]
        change = "→ FIXED" if p9_3 > p8_3 else ("→ BROKE" if p9_3 < p8_3 else "same")
        question_text = question["question"]
        if len(question_text) > 60:
            question_text = question_text[:60] + "..."
        lines.append(
            f"| {index} | {markdown_cell(question_text)} | {p8_3:.3f} | {p9_3:.3f} | {change} |"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved per-question delta to {OUTPUT}")
    summary = {
        "date": run_date,
        "collection": COLLECTION_NAME,
        "points": point_count,
        "questions": len(golden),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "hybrid_initialization_s": initialization_s,
        "dense": {
            key: value for key, value in baseline.items() if key != "per_question"
        },
        "hybrid": {
            key: value for key, value in hybrid.items() if key != "per_question"
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    write_snapshot(summary, baseline, hybrid)
    print(f"Saved measurements to {SUMMARY}")
    print(f"Updated {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
