import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.week5.pipeline.models import Answer, Question
from src.week5.pipeline.pipeline import ask_llm, stream_answer
from src.week5.pipeline.settings import Settings
from src.week5.pipeline.store import connect, save_answer

logger = logging.getLogger(__name__)

app = FastAPI(title="Capstone API — W4")

# Single Settings instance — read once at startup.
_settings = Settings()
_db_path = Path(_settings.results_db)


@app.get("/health")
async def health() -> dict:
    """Unchanged from W3."""
    return {"status": "ok"}


@app.post("/ask_batched", response_model=Answer)
async def ask_batched(q: Question) -> Answer:
    """Non-streaming structured Answer via tool-calling. Persists to SQLite."""
    answer = await ask_llm(q, _settings)
    # Persist with the new columns. Backward-compatible with W3 callers — they
    # just don't read the new columns.
    with connect(_db_path) as conn:
        save_answer(
            conn,
            question=q.question,
            content=answer.content,
            retries=answer.retries,
            cost_usd=answer.cost_usd,
            model=_settings.model,
            confidence=answer.confidence,
            sources=answer.sources,
            schema_version=answer.schema_version,
        )
    return answer


@app.post("/ask")
async def ask(q: Question) -> StreamingResponse:
    """Real streaming text/plain. Same URL + same input as W3."""

    async def _gen():
        async for chunk in stream_answer(q.question, _settings):
            yield chunk

    return StreamingResponse(_gen(), media_type="text/plain")
