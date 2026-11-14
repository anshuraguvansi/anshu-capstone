"""Measure grounded responses from Week 8 dense retrieval."""

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
from src.week8.pipeline.cost import compute_cost_usd  # noqa: E402
from src.week8.rag import qdrant_rag  # noqa: E402

GOLDEN_SET = ROOT / "data/golden_set.jsonl"
RESULTS = ROOT / "data/wk8_grounded_results.jsonl"
SUMMARY = ROOT / "data/wk8_grounded_summary.json"
SNAPSHOT = ROOT / "docs/kpi/wk8-snapshot.md"
ANSWER_MODEL = "gpt-4o-mini"
COLLECTION = "capstone_chunks_v2"
EMBEDDING_RATE_PER_MILLION = 0.02


def load_golden():
    questions = [json.loads(line) for line in GOLDEN_SET.read_text().splitlines() if line.strip()]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Golden set must be nonempty with unique question IDs")
    for question in questions:
        for field in ("id", "question", "expected_answer", "expected_source", "evidence"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"Missing golden field: {field}")
    return questions


def summarize(rows, run_date, point_count):
    count = len(rows)
    return {
        "date": run_date,
        "collection": COLLECTION,
        "points": point_count,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "questions": count,
        "grounded_responses": sum(row["grounded_response"] for row in rows),
        "grounded_response_rate": mean(row["grounded_response"] for row in rows),
        "abstentions": sum(row["judge"]["evaluation"]["abstained"] for row in rows),
        "mean_accuracy": mean(row["judge"]["evaluation"]["accuracy"] for row in rows),
        "retrieval_latency_p50_ms": median(row["retrieval_latency_ms"] for row in rows),
        "answer_latency_p50_ms": median(row["answer_latency_ms"] for row in rows),
        "retrieval_and_answer_latency_p50_ms": median(row["retrieval_latency_ms"] + row["answer_latency_ms"] for row in rows),
        "embedding_cost_per_query_usd": mean(row["embedding_cost_usd"] for row in rows),
        "answer_cost_per_query_usd": mean(row["answer_cost_usd"] for row in rows),
        "retrieval_and_answer_cost_per_query_usd": mean(row["embedding_cost_usd"] + row["answer_cost_usd"] for row in rows),
        "judge_cost_total_usd": sum(row["judge"]["estimated_cost_usd"] for row in rows),
    }


def update_snapshot(summary):
    text = SNAPSHOT.read_text()
    count = summary["questions"]
    grounded = summary["grounded_responses"]
    rate = f"{summary['grounded_response_rate']:.1%} ({grounded}/{count}; follow-up {summary['date']})"
    new_row = f"| Grounded response rate | Not measured | {rate} | — |"
    marker = "\n## Grounded response evaluation"
    text = text.split(marker)[0]
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| Grounded response rate |"):
            lines[index] = new_row
            break
    else:
        raise ValueError("Grounded response row missing from Week 8 snapshot")
    text = "\n".join(lines).rstrip() + "\n"
    text = text.replace("generated answers were not evaluated.", "generated answers were not evaluated in that run.")
    section = f"""
## Grounded response evaluation — {summary['date']}

I re-ran the 30 golden questions using Week 8 dense retrieval from `{COLLECTION}`
({summary['points']} points), top-3 context, and `{ANSWER_MODEL}` at temperature 0.
The `{JUDGE_MODEL}` judge scores accuracy, groundedness, and format from 1 to 4.
A grounded response scores 4/4 for groundedness and is not an abstention.
The denominator includes all {count} questions. The judge checks support against the
retrieved context; the reference answer is used for accuracy only.

| Metric | Result |
|---|---:|
| Grounded response rate | {summary['grounded_response_rate']:.1%} ({grounded}/{count}) |
| Abstentions | {summary['abstentions']}/{count} |
| Mean accuracy (1–4) | {summary['mean_accuracy']:.2f} |
| Retrieval latency p50 (ms) | {summary['retrieval_latency_p50_ms']:.1f} |
| Answer generation latency p50 (ms) | {summary['answer_latency_p50_ms']:.1f} |
| Retrieval + answer latency p50 (ms) | {summary['retrieval_and_answer_latency_p50_ms']:.1f} |
| Estimated embedding cost/query | ${summary['embedding_cost_per_query_usd']:.9f} |
| Estimated answer generation cost/query | ${summary['answer_cost_per_query_usd']:.6f} |
| Estimated retrieval + answer cost/query | ${summary['retrieval_and_answer_cost_per_query_usd']:.6f} |
| Estimated judge cost (entire run) | ${summary['judge_cost_total_usd']:.4f} |

Latency excludes judging; retrieval timing includes query embedding and Qdrant search.
Costs cover token API charges only. Embedding cost uses tokenized queries at
$0.02/million tokens; generation and judge estimates use API token usage and the
repository's model rates, before caching discounts. Qdrant and local compute costs
are excluded. LLM judgments have not been manually audited.

Answers, retrieved context, token usage, and judge reasoning:
[`data/wk8_grounded_results.jsonl`](../../data/wk8_grounded_results.jsonl).
Summary: [`data/wk8_grounded_summary.json`](../../data/wk8_grounded_summary.json).
Run: `uv run --env-file .env python scripts/measure_wk8_grounding.py`.
"""
    SNAPSHOT.write_text(text + section)


async def main():
    questions = load_golden()
    run_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    tokenizer = tiktoken.encoding_for_model(qdrant_rag.EMBED_MODEL)
    store = qdrant_rag.load_store()
    rows = []
    try:
        point_count = store.client.get_collection(COLLECTION).points_count
        async with AsyncOpenAI() as judge_client:
            with RESULTS.open("w", encoding="utf-8") as output:
                for index, question in enumerate(questions, 1):
                    started = perf_counter()
                    hits = qdrant_rag.retrieve(store, question["question"], k=3)
                    retrieval_ms = (perf_counter() - started) * 1000
                    context = "\n\n".join(
                        f"[{hit['chunk_id']}] (source: {hit['source_id']})\n{hit['text']}"
                        for hit in hits
                    )
                    if not context.strip():
                        raise ValueError(f"No retrieved context for {question['id']}")
                    started = perf_counter()
                    response = qdrant_rag._openai().chat.completions.create(
                        model=ANSWER_MODEL,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": qdrant_rag.SYSTEM},
                            {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {question['question']}"},
                        ],
                    )
                    answer_ms = (perf_counter() - started) * 1000
                    answer = response.choices[0].message.content
                    if not answer or response.usage is None:
                        raise ValueError(f"Missing answer or token usage for {question['id']}")
                    judgment = await score_llm_judge(
                        judge_client,
                        question=question["question"],
                        candidate_answer=answer,
                        ideal_answer=question["expected_answer"],
                        context=context,
                        format_requirements="Cite the source_id in square brackets after factual claims.",
                    )
                    tokens = len(tokenizer.encode(question["question"]))
                    row = {
                        "date": run_date,
                        "collection": COLLECTION,
                        "golden_id": question["id"],
                        "question": question["question"],
                        "expected_answer": question["expected_answer"],
                        "answer_model": ANSWER_MODEL,
                        "answer": answer,
                        "retrieved": hits,
                        "context": context,
                        "retrieval_latency_ms": retrieval_ms,
                        "answer_latency_ms": answer_ms,
                        "embedding_tokens": tokens,
                        "embedding_cost_usd": tokens / 1_000_000 * EMBEDDING_RATE_PER_MILLION,
                        "answer_prompt_tokens": response.usage.prompt_tokens,
                        "answer_completion_tokens": response.usage.completion_tokens,
                        "answer_cost_usd": compute_cost_usd(ANSWER_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens),
                        "judge": {
                            **asdict(judgment),
                            "evaluation": judgment.evaluation.model_dump(),
                        },
                        "grounded_response": judgment.grounded_response,
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    rows.append(row)
                    print(f"{index}/{len(questions)} {question['id']}: grounded={row['grounded_response']}, accuracy={judgment.evaluation.accuracy}", flush=True)
    finally:
        store.client.close()
        if qdrant_rag._client is not None:
            qdrant_rag._client.close()
            qdrant_rag._client = None
    summary = summarize(rows, run_date, point_count)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    update_snapshot(summary)
    print(json.dumps(summary, indent=2))
    print(f"Updated {SNAPSHOT}")


if __name__ == "__main__":
    asyncio.run(main())
