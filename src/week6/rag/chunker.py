import json
from collections.abc import Collection, Iterator
from pathlib import Path

DEFAULT_FILE_SUFFIXES = frozenset({".jsonl"})


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over characters."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("Chunk overlap must be non-negative and smaller than size")
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
    return chunks


def _normalise_suffixes(file_suffixes: Collection[str]) -> set[str]:
    return {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in file_suffixes
    }


def _read_jsonl(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not isinstance(record.get("text"), str)
            ):
                raise ValueError(  # noqa: TRY004
                    f"{path}:{line_number} must contain string 'id' and 'text' fields"
                )
            yield record["id"], record["text"]


def _read_documents(
    path: Path, file_suffixes: Collection[str]
) -> Iterator[tuple[str, str]]:
    suffixes = _normalise_suffixes(file_suffixes)
    if not suffixes:
        raise ValueError("At least one file suffix is required")

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in suffixes
        )
    else:
        raise ValueError(f"Corpus path does not exist: {path}")

    for file in files:
        suffix = file.suffix.lower()
        if suffix not in suffixes:
            raise ValueError(f"Unsupported corpus file suffix: {suffix}")
        if suffix == ".jsonl":
            yield from _read_jsonl(file)
        else:
            yield file.stem, file.read_text(encoding="utf-8")


def chunk_corpus(
    path: Path | str,
    size: int = 500,
    overlap: int = 50,
    file_suffixes: Collection[str] = DEFAULT_FILE_SUFFIXES,
) -> list[dict]:
    """Chunk supported corpus files, preserving each document's source ID."""
    path = Path(path)
    return [
        {
            "chunk_id": f"{source_id}#{index}",
            "source_id": source_id,
            "text": chunk,
        }
        for source_id, text in _read_documents(path, file_suffixes)
        for index, chunk in enumerate(chunk_text(text, size=size, overlap=overlap))
    ]


if __name__ == "__main__":
    chunks = chunk_corpus("data/corpus/cases.jsonl")
    print(chunks)
