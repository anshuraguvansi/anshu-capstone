import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the project package importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.week6.rag.chunker import chunk_corpus
from src.week6.rag.embeddings import EMBED_MODEL, embed_batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an embedding index.")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/cases.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/embeddings_medium.json"))
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args(argv)


def build_payload(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    *,
    corpus: Path,
    size: int,
    overlap: int,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("The corpus produced no chunks")
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("All embeddings must have the same non-zero dimension")

    return {
        "metadata": {
            "schema_version": "1.0",
            "corpus": str(corpus),
            "embedding_model": EMBED_MODEL,
            "embedding_dimension": dimensions.pop(),
            "chunk_size": size,
            "chunk_overlap": overlap,
            "document_count": len({chunk["source_id"] for chunk in chunks}),
            "chunk_count": len(chunks),
        },
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"],
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Chunking {args.corpus} with size={args.size} overlap={args.overlap}...")
    chunks = chunk_corpus(args.corpus, size=args.size, overlap=args.overlap)
    print(f"  Got {len(chunks)} chunks.")

    print("Embedding ...")
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts)
    output = build_payload(
        chunks,
        vectors,
        corpus=args.corpus,
        size=args.size,
        overlap=args.overlap,
    )
    dimension = output["metadata"]["embedding_dimension"]
    print(f"  Embedded {len(vectors)} chunks; dim = {dimension}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)
    print(f"Wrote {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
