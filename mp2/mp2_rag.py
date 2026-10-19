"""MP2 · Mini-RAG — Minimal Retrieval-Augmented Generation Implementation"""

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR = Path(__file__).parent / "corpus"
DATA_DIR = Path(__file__).parent / "data"
COLLECTION_NAME = "mp2_sherlock"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 80  # characters

openai = OpenAI()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"), timeout=60
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ─── Step 1: Load the corpus ────────────────────────────────────────────


def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).
    """
    docs = []
    for filename in os.listdir(corpus_dir):
        file_path = corpus_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                title = line.strip()
                break
            text = f.read()
            docs.append({"source": filename, "title": title, "text": text})

    return docs


# ─── Step 2: Chunk each document ────────────────────────────────────────


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk has a stable id, source, title, section, and text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.
    """

    chunks = []
    text = doc["text"]
    i = 0
    while i < len(text):
        end = min(i + TARGET_CHUNK_SIZE, len(text))
        chunks.append(
            {
                "id": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        json.dumps([doc["source"], i, text[i:end]]),
                    )
                ),
                "source": doc["source"],
                "title": doc["title"],
                "section": f"Section {len(chunks) + 1}",
                "text": text[i:end],
            }
        )
        if end == len(text):
            break
        i = end - CHUNK_OVERLAP
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).
    """
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────


def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────


def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant."""
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload=chunk,
            )
            for vector, chunk in zip(embed_texts([c["text"] for c in chunks]), chunks)
        ],
    )


# ─── Step 6: Retrieve ───────────────────────────────────────────────────


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query."""
    query_vector = embed_texts([query])[0]
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    ).points

    return [{**hit.payload, "score": hit.score} for hit in search_results]


# ─────────── hybrid  rerank retrive ──────────────────────────────
def bm25_retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks by BM25 score, highest first."""
    tokenized_query = re.findall(r"\w+", query.lower())
    if k <= 0 or not tokenized_query:
        return []

    chunks = [chunk for doc in load_corpus(CORPUS_DIR) for chunk in chunk_document(doc)]
    tokens = [re.findall(r"\w+", chunk["text"].lower()) for chunk in chunks]
    if not any(tokens):
        return []

    bm25 = BM25Okapi(tokens)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks, strict=True), key=lambda pair: pair[0], reverse=True
    )
    return [{**chunk, "score": float(score)} for score, chunk in ranked[:k]]


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]], k: int = 60, top_n: int = 5
) -> list[dict[str, Any]]:
    """Combine ranked chunk lists using reciprocal rank fusion."""
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    if top_n <= 0:
        return []

    scores = {}
    docs = {}
    for ranked in ranked_lists:
        seen = set()
        for rank, hit in enumerate(ranked, start=1):
            chunk_key = hit["id"]
            if chunk_key in seen:
                continue
            seen.add(chunk_key)
            scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (k + rank)
            if chunk_key not in docs:
                docs[chunk_key] = hit

    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:top_n]
    return [{**docs[key], "rrf_score": score} for key, score in fused]


def hybrid_retrieve(
    query: str, k_per_retriever: int = 5, k_final: int = 3
) -> list[dict[str, Any]]:
    bm25_hits = bm25_retrieve(query, k=k_per_retriever)
    dense_hits = retrieve(query, k=k_per_retriever)
    result = rrf_fuse([bm25_hits, dense_hits], k=60, top_n=k_per_retriever)
    if result:
        scores = reranker.predict([(query, hit["text"]) for hit in result])
        for hit, score in zip(result, scores):
            hit["rerank_score"] = float(score)
        result.sort(key=lambda x: x["rerank_score"], reverse=True)
    return result[:k_final]


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result."""

    start_time = time.time()
    # context = retrieve(question, k=k)
    context = hybrid_retrieve(question, k_final=k)
    context_str = "\n".join(
        f"[Source: {chunk['title']} — {chunk['section']}]\n{chunk['text']}"
        for chunk in context
    )
    user_message = f"Question: {question}\n\nContext:\n{context_str}"
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer_text = response.choices[0].message.content
    citations = [chunk for chunk in context]
    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": int((time.time() - start_time) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────


def validate_against(jsonl_path: Path) -> None:
    questions = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(
            1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower
        )
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Answer: {result['answer']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────


def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "ask":
        cmd_ask()
    elif cmd == "validate":
        cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
