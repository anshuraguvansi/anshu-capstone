import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_RESULTS = Path("data/rag_eval_results.jsonl")
METRICS = {
    "retrieval_hit_rate",
    "source_hit_rate",
    "evidence_hit_rate",
    "cost_per_query",
    "p50_latency",
    "p95_latency",
    "all",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KPIs for a RAG eval run.")
    parser.add_argument("--label", required=True, help="Evaluation run label")
    parser.add_argument(
        "--metric", required=True, choices=sorted(METRICS), help="KPI to print"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Evaluation results JSONL (default: {DEFAULT_RESULTS})",
    )
    return parser.parse_args(argv)


def load_run(path: Path, label: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if isinstance(record, dict) and record.get("label") == label:
                records.append(record)
    if not records:
        raise ValueError(f"No results found for label '{label}' in {path}")
    return records


def hit_rate(records: list[dict[str, Any]], field: str) -> tuple[int, int, float]:
    hits = sum(bool(record.get(field)) for record in records)
    total = len(records)
    return hits, total, hits / total


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentage * len(ordered)) - 1)
    return ordered[index]


def calculate(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_hits, total, source_rate = hit_rate(records, "source_hit")
    evidence_hits, _, evidence_rate = hit_rate(records, "evidence_hit")
    costs = [float(record.get("cost_usd", 0.0)) for record in records]
    latencies = [float(record["latency_seconds"]) for record in records]
    return {
        "questions": total,
        "source_hits": source_hits,
        "source_hit_rate": source_rate,
        "evidence_hits": evidence_hits,
        "evidence_hit_rate": evidence_rate,
        "retrieval_hit_rate": evidence_rate,
        "cost_per_query": statistics.fmean(costs),
        "p50_latency": statistics.median(latencies),
        "p95_latency": percentile(latencies, 0.95),
    }


def print_metric(metric: str, kpis: dict[str, Any]) -> None:
    if metric == "retrieval_hit_rate" or metric == "evidence_hit_rate":
        print(
            f"{metric}: {kpis['evidence_hits']} / {kpis['questions']} "
            f"({kpis['evidence_hit_rate']:.2%})"
        )
    elif metric == "source_hit_rate":
        print(
            f"source_hit_rate: {kpis['source_hits']} / {kpis['questions']} "
            f"({kpis['source_hit_rate']:.2%})"
        )
    elif metric == "cost_per_query":
        print(f"cost_per_query: ${kpis['cost_per_query']:.8f}")
    elif metric == "p50_latency":
        print(f"p50_latency: {kpis['p50_latency']:.3f}s")
    elif metric == "p95_latency":
        print(f"p95_latency: {kpis['p95_latency']:.3f}s")
    else:
        print(
            json.dumps(
                {
                    **kpis,
                    "source_hit_rate": round(kpis["source_hit_rate"], 6),
                    "evidence_hit_rate": round(kpis["evidence_hit_rate"], 6),
                    "retrieval_hit_rate": round(kpis["retrieval_hit_rate"], 6),
                    "cost_per_query": round(kpis["cost_per_query"], 10),
                    "p50_latency": round(kpis["p50_latency"], 6),
                    "p95_latency": round(kpis["p95_latency"], 6),
                },
                indent=2,
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = load_run(args.results, args.label)
        kpis = calculate(records)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print_metric(args.metric, kpis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
