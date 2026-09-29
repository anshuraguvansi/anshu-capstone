import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:  # A lightweight estimate keeps the evaluator usable without it.
    tiktoken = None

# Make the project package importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.week6.rag.retriever import retrieve

DEFAULT_GOLDEN_SET = Path("data/golden_set.jsonl")
DEFAULT_RESULTS = Path("data/rag_eval_results.jsonl")
DEFAULT_EMBEDDING_COST_PER_MILLION = 0.02
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def parse_bool(value: str) -> bool:
    normalised = value.strip().casefold()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def get_embedder(local: bool) -> tuple[str, Callable[[list[str]], list[list[float]]]]:
    if local:
        from src.week6.rag.embeddings_local import EMBED_MODEL, embed

        return EMBED_MODEL, lambda texts: [embed(text) for text in texts]

    from src.week6.rag.embeddings import EMBED_MODEL, embed_batch

    return EMBED_MODEL, embed_batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval against the clinical golden set."
    )
    parser.add_argument("--label", required=True, help="Name for this evaluation run")
    parser.add_argument("--index", required=True, type=Path, help="Index JSON file")
    parser.add_argument(
        "--local",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help="Use local MiniLM embeddings instead of OpenAI (default: false)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_SET,
        help=f"Golden-set JSONL file (default: {DEFAULT_GOLDEN_SET})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Result JSONL file (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--evidence-threshold",
        type=float,
        default=0.70,
        help="Required gold-evidence word coverage between 0 and 1",
    )
    parser.add_argument(
        "--embedding-cost-per-million",
        type=float,
        default=DEFAULT_EMBEDDING_COST_PER_MILLION,
        help="Embedding input price in USD per million tokens",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(  # noqa: TRY004
                    f"Expected a JSON object at {path}:{line_number}"
                )
            records.append(record)
    return records


def load_index(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate an embedding index."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise ValueError(f"{path} must contain a 'chunks' list")  # noqa: TRY004

    chunks = []
    for position, chunk in enumerate(payload["chunks"], start=1):
        if not isinstance(chunk, dict):
            raise ValueError(  # noqa: TRY004
                f"Invalid chunk at position {position} in {path}"
            )

        source_id = chunk.get("source_id", chunk.get("source"))
        embedding = chunk.get("embedding", chunk.get("vector"))
        if not isinstance(source_id, str) or not isinstance(chunk.get("text"), str):
            raise ValueError(  # noqa: TRY004
                f"Chunk {position} is missing source or text in {path}"
            )
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Chunk {position} is missing its embedding in {path}")

        chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", f"{source_id}#{position - 1}"),
                "source_id": source_id,
                "text": chunk["text"],
                "embedding": [float(value) for value in embedding],
            }
        )

    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return payload.get("metadata", {}), chunks


def _normalise_words(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.casefold()))


def measure_evidence(
    evidence: str, expected_source: str, retrieved: list[dict[str, Any]]
) -> tuple[bool, float]:
    """Measure gold-evidence word coverage in retrieved source chunks."""
    relevant_text = " ".join(
        hit["text"] for hit in retrieved if hit["source_id"] == expected_source
    )
    evidence_words = _normalise_words(evidence)
    if not evidence_words:
        return False, 0.0
    coverage = len(evidence_words & _normalise_words(relevant_text)) / len(
        evidence_words
    )
    return bool(relevant_text), coverage


def count_tokens(text: str, embedding_model: str) -> int:
    if tiktoken is not None:
        encoding = tiktoken.encoding_for_model(embedding_model)
        return len(encoding.encode(text))
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def validate_golden(record: dict[str, Any], position: int) -> None:
    required = {"id", "question", "expected_source", "evidence"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Golden record {position} is missing {sorted(missing)}")
    for field in required:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Golden record {position} has an invalid '{field}'")


def save_results(path: Path, label: str, new_records: list[dict[str, Any]]) -> None:
    existing = load_jsonl(path) if path.exists() else []
    retained = [record for record in existing if record.get("label") != label]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in [*retained, *new_records]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if not 0 <= args.evidence_threshold <= 1:
        raise ValueError("--evidence-threshold must be between 0 and 1")
    if args.embedding_cost_per_million < 0:
        raise ValueError("--embedding-cost-per-million cannot be negative")

    embedding_model, embed_many = get_embedder(args.local)
    metadata, chunks = load_index(args.index)
    index_model = metadata.get("embedding_model")
    if index_model and index_model != embedding_model:
        raise ValueError(
            f"Index uses '{index_model}', but the selected evaluator uses "
            f"'{embedding_model}'. Set --local to match the index."
        )
    golden_records = load_jsonl(args.golden)
    results = []

    for position, golden in enumerate(golden_records, start=1):
        validate_golden(golden, position)
        question = golden["question"]
        started = time.perf_counter()
        query_embedding = embed_many([question])[0]
        retrieved = retrieve(query_embedding, chunks, args.top_k)
        latency_seconds = time.perf_counter() - started

        source_hit, evidence_coverage = measure_evidence(
            golden["evidence"], golden["expected_source"], retrieved
        )
        token_count = 0 if args.local else count_tokens(question, embedding_model)
        cost_usd = (
            0.0
            if args.local
            else token_count / 1_000_000 * args.embedding_cost_per_million
        )
        results.append(
            {
                "label": args.label,
                "golden_id": golden["id"],
                "question": question,
                "expected_source": golden["expected_source"],
                "source_hit": source_hit,
                "evidence_hit": source_hit
                and evidence_coverage >= args.evidence_threshold,
                "evidence_coverage": round(evidence_coverage, 6),
                "top_k": args.top_k,
                "latency_seconds": round(latency_seconds, 6),
                "embedding_tokens": token_count,
                "embedding_model": embedding_model,
                "local": args.local,
                "cost_usd": cost_usd,
                "index": str(args.index),
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": hit["chunk_id"],
                        "source_id": hit["source_id"],
                        "score": round(hit["score"], 6),
                    }
                    for rank, hit in enumerate(retrieved, start=1)
                ],
            }
        )

    save_results(args.results, args.label, results)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = run(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    source_hits = sum(record["source_hit"] for record in results)
    evidence_hits = sum(record["evidence_hit"] for record in results)
    total = len(results)
    print(f"label: {args.label}")
    print(f"questions: {total}")
    print(f"source hit rate@{args.top_k}: {source_hits}/{total}")
    print(f"evidence hit rate@{args.top_k}: {evidence_hits}/{total}")
    print(f"results: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
