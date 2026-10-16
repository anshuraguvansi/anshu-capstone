import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.week7.pipeline.models import Answer, Question
from src.week7.pipeline.cost import compute_cost_usd
from src.week7.pipeline.pipeline import stream_answer
from src.week7.pipeline.settings import Settings
from src.week7.pipeline.store import connect, save_answer
from src.week7.rag import qdrant_rag

logger = logging.getLogger(__name__)

app = FastAPI(title="Capstone API — W7")

# Single Settings instance — read once at startup.
_settings = Settings()
_db_path = Path(_settings.results_db)


@app.get("/health")
async def health() -> dict:
    """Unchanged from W3."""
    return {"status": "ok"}


@app.post("/ask_batched", response_model=Answer)
def ask_batched(q: Question) -> Answer:
    """Answer using Qdrant context and persist the result to SQLite."""
    result = qdrant_rag.ask_rag(q.question, settings=_settings)
    answer = Answer(
        content=result["answer"],
        sources=result["sources"],
        cost_usd=compute_cost_usd(
            _settings.model, result["tokens_in"], result["tokens_out"]
        ),
    )
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
