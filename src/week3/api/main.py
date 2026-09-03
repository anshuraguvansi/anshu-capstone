import asyncio
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.week2.pipeline.pipeline import ask_llm as _pipeline_ask_llm
from src.week2.pipeline.pipeline import Question as _PipelineQuestion


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


# API models
class Question(BaseModel):
    """Public request shape — locked in ADR 0002."""

    question: str


class Answer(BaseModel):
    """Public response shape — locked in ADR 0002."""

    content: str
    cost_usd: float
    retries: int


app = FastAPI(
    title="Anshu Capstone API",
    description="Wraps the W2 async pipeline. Contract locked in ADR 0002 (W3); internals upgraded W4+.",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# /ask_batched — non-streaming reference endpoint
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/ask_batched", response_model=Answer)
async def ask_batched(question: Question) -> Answer:
    logger.info("asked_batched with question: %s", question.question[:80])
    pipeline_question = _PipelineQuestion(text=question.question)
    pipelineanswer = await _pipeline_ask_llm(pipeline_question)
    return Answer(
        content=pipelineanswer.text,
        cost_usd=pipelineanswer.cost_usd,
        retries=pipelineanswer.retries,
    )


# Health probe
@app.get("/health")
async def health():
    return {"status": "ok"}


async def stream_answer(question: str):
    pipeline_question = _PipelineQuestion(text=question)
    pipelineanswer = await _pipeline_ask_llm(pipeline_question)
    for chunk in pipelineanswer.text.split(" "):
        yield chunk + " "
        await asyncio.sleep(0.01)


@app.post("/ask", response_class=StreamingResponse)
async def ask(question: Question):
    return StreamingResponse(stream_answer(question.question), media_type="text/plain")
