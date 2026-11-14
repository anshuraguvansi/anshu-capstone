"""Ingest clinical JSONL cases into the Week 8 Qdrant store."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from src.week9.rag.qdrant_store import (
    QdrantStore,
    _get_client,
    ensure_collection,
    upsert_chunks,
)

_analyzer = None
_anonymizer = None
_openai = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid PII mode requires uv sync --extra ingest"
            ) from exc

        _analyzer = AnalyzerEngine()
    return _analyzer


def _get_anonymizer():
    global _anonymizer
    if _anonymizer is None:
        from presidio_anonymizer import AnonymizerEngine

        _anonymizer = AnonymizerEngine()
    return _anonymizer


def _get_openai():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def read_corpus(corpus_path: Path):
    """Yield (file, source_id, text) without combining distinct clinical cases.

    Supports {id, text} cases and selected dataset rows with idx/full_note/note.
    Generated conversations and summaries are deliberately excluded from evidence.
    """
    if corpus_path.is_file():
        files = [corpus_path]
    elif corpus_path.is_dir():
        files = sorted(
            p
            for p in corpus_path.rglob("*")
            if p.is_file() and p.suffix.lower() == ".jsonl"
        )
    else:
        raise ValueError(f"Corpus path does not exist: {corpus_path}")
    seen = set()
    for path in files:
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a JSONL corpus file: {path}")
        records = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{location}: expected a JSON object")
                source_id = row.get("id")
                if (
                    source_id is None
                    and isinstance(row.get("idx"), int)
                    and not isinstance(row["idx"], bool)
                ):
                    source_id = f"case_{row['idx']}"
                text = next(
                    (
                        row[key]
                        for key in ("text", "full_note", "note")
                        if isinstance(row.get(key), str) and row[key].strip()
                    ),
                    None,
                )
                if (
                    not isinstance(source_id, str)
                    or not source_id.strip()
                    or text is None
                ):
                    raise ValueError(
                        f"{location}: expected id/text or idx/full_note (or note)"
                    )
                records.append((source_id.strip(), text))
        for source_id, text in records:
            if source_id in seen:
                raise ValueError(f"{path}: duplicate source_id {source_id!r}")
            if not text.strip():
                raise ValueError(f"{path}: empty text for {source_id!r}")
            seen.add(source_id)
            yield path, source_id, text


def chunk_document(text: str, max_size: int = 800, overlap: int = 80) -> list[str]:
    """Split text into bounded character windows with overlap."""
    if max_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if overlap < 0 or overlap >= max_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than size")

    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    # Require a full phone number; short clinical values and gene IDs must survive.
    "PHONE": re.compile(
        r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w.])"
    ),
    "EMP_ID": re.compile(r"\bEMP-\d{4,6}\b"),
}


def scrub_pii(text: str, *, pii_mode: str = "hybrid") -> tuple[str, list[dict]]:
    """Scrub before chunking; flags contain counts/spans, never raw identifiers.

    Hybrid adds Presidio PERSON and identifier recognition while retaining clinical
    dates, ages, locations and organizations as contextual evidence.
    """
    if pii_mode not in {"regex", "hybrid"}:
        raise ValueError("pii_mode must be regex or hybrid")
    flags = []
    scrubbed = text
    for label, pattern in PII_PATTERNS.items():

        def replace(match):
            flags.append({"source": "regex", "type": label})
            return f"[{label}]"

        scrubbed = pattern.sub(replace, scrubbed)
    if pii_mode == "hybrid":
        results = _get_analyzer().analyze(
            text=scrubbed,
            language="en",
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "IP_ADDRESS",
            ],
        )
        if results:
            scrubbed = (
                _get_anonymizer()
                .anonymize(text=scrubbed, analyzer_results=results)
                .text
            )
            flags.extend(
                {"source": "presidio", "type": r.entity_type, "score": r.score}
                for r in results
            )
    return scrubbed, flags


def enrich_metadata(
    text: str,
    path: Path,
    chunk_idx: int,
    pii_count: int,
    *,
    source_id: str | None = None,
) -> dict:
    source_id = source_id or path.stem
    now = datetime.now(timezone.utc)
    return {
        "chunk_id": f"{source_id}#{chunk_idx}",
        "source_id": source_id,
        "source": path.name,
        "doc_type": path.suffix.lstrip(".").lower(),
        "section_path": source_id,
        "page": None,
        "date": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(
            "%Y-%m-%d"
        ),
        "language": "en",
        "version": "v1",
        "ingested_at": now.isoformat(),
        "text": text,
        "pii_flags_count": pii_count,
    }


def ingest_corpus(
    corpus_dir: Path, collection_name: str = "capstone_chunks_v2"
) -> dict:
    """Read clinical JSONL cases, scrub PII, embed and upsert to Qdrant.

    Returns file, case, chunk and PII counts with a per-file breakdown.
    """
    all_chunks = []
    per_file = {}
    n_documents = n_flags = 0
    for path, source_id, text in read_corpus(Path(corpus_dir)):
        scrubbed, flags = scrub_pii(text)
        chunks = []
        for idx, part in enumerate(chunk_document(scrubbed)):
            count = len(
                re.findall(
                    r"\[(?:EMAIL|PHONE|EMP_ID)\]|<(?:PERSON|EMAIL_ADDRESS|PHONE_NUMBER|US_SSN|CREDIT_CARD|IP_ADDRESS)>",
                    part,
                )
            )
            chunks.append(enrich_metadata(part, path, idx, count, source_id=source_id))
        entry = per_file.setdefault(
            str(path), {"documents": 0, "chunks": 0, "pii_flags": 0}
        )
        entry["documents"] += 1
        entry["chunks"] += len(chunks)
        entry["pii_flags"] += len(flags)
        n_documents += 1
        n_flags += len(flags)
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError("No non-empty JSONL cases found in corpus")
    print(f"Embedding {len(all_chunks)} chunks...")
    openai = _get_openai()
    vectors = []
    for start in range(0, len(all_chunks), 100):
        batch = all_chunks[start : start + 100]
        resp = openai.embeddings.create(
            model="text-embedding-3-small", input=[c["text"] for c in batch]
        )
        items = sorted(resp.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError("Embedding response does not match input batch")
        vectors.extend(item.embedding for item in items)
    store = QdrantStore(client=_get_client(), collection=collection_name)
    try:
        ensure_collection(store)
        print(f"Upserting to Qdrant collection {collection_name!r}...")
        upsert_chunks(store, all_chunks, vectors)
    finally:
        store.client.close()
    return {
        "n_files": len(per_file),
        "n_documents": n_documents,
        "n_chunks": len(all_chunks),
        "n_pii_flags": n_flags,
        "per_file": per_file,
    }


def main():
    corpus_path = Path("data/corpus")
    stats = ingest_corpus(corpus_path)
    print("\n══ Summary ══")
    print(f"Files ingested: {stats['n_files']}")
    print(f"Cases ingested: {stats['n_documents']}")
    print(f"Total chunks:   {stats['n_chunks']}")
    print(f"PII flags:      {stats['n_pii_flags']}")


if __name__ == "__main__":
    main()
