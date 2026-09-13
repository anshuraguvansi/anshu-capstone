import asyncio
import csv
import json
import time
from pathlib import Path

from .cost import compute_cost_usd

from .logging_config import get_logger
from .settings import Settings, RunSummary
from .models import Answer, Question

from collections.abc import AsyncIterator

from openai import AsyncOpenAI


# W4: Structured output tool schema
# ─── Tool schema for structured outputs ─────────────────────────────────────
ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Return a structured answer with content, confidence, and sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The answer in 2-4 sentences.",
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident you are in the answer, 0.0 to 1.0.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Source identifiers or URLs you used. Empty list is fine "
                        "if you used general knowledge."
                    ),
                },
            },
            "required": ["content", "confidence", "sources"],
        },
    },
}

_settings_for_import = Settings()

if _settings_for_import.use_fake:
    from .fake_llm import FakeLLMError, fake_ask_llm
else:
    from openai import AsyncOpenAI

    _client = AsyncOpenAI()


logger = get_logger("pipeline")


def load_questions(path: str | Path = "data/questions.csv") -> list[Question]:
    file_path = Path(path)
    rows = csv.DictReader(file_path.open("r", encoding="utf-8"))
    return [Question(text=row.get("text")) for row in rows]


def summarize_run(
    answers: list[Answer],
    *,
    started_at: float,
    elapsed: float,
    fail_rate: float,
    use_fake: bool,
) -> RunSummary:
    n_questions = len(answers)
    n_successed = len(answers)
    n_retries_total = sum(a.retries for a in answers)
    total_cost_usd = sum(a.cost_usd for a in answers)
    return RunSummary(
        started_at=started_at,
        elapsed_seconds=elapsed,
        n_questions=n_questions,
        n_successed=n_successed,
        n_retries_total=n_retries_total,
        total_cost_usd=total_cost_usd,
        fail_rate=fail_rate,
        use_fake=use_fake,
    )


# ─────────────────────────────────────────────────────────────────────────────
# single LLM call
# ─────────────────────────────────────────────────────────────────────────────
async def ask_llm(q: Question, fail_rate: float = 0.0) -> Answer:
    """One LLM call. Branches on Settings.use_fake."""
    if _settings_for_import.use_fake:
        ans = await fake_ask_llm(q, fail_rate=fail_rate)
    else:
        resp = await _client.chat.completions.create(
            model=_settings_for_import.model,
            messages=[{"role": "user", "content": q.text}],
            tools=[ANSWER_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "answer_question"},
            },
        )
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        completion_tokens = usage.completion_tokens if usage is not None else 0

        # Parse the tool call's structured arguments.
        tool_calls = resp.choices[0].message.tool_calls or []
        if not tool_calls:
            # Defensive — should not happen because tool_choice forces it,
            # but if a provider misbehaves we want a clear error.
            raise RuntimeError("LLM did not call the answer_question tool")
        args_json = tool_calls[0].function.arguments
        args = json.loads(args_json)

        ans = Answer(
            question=q.text,
            text=args["content"],
            cost_usd=compute_cost_usd(
                _settings_for_import.model, prompt_tokens, completion_tokens
            ),
            confidence=args.get("confidence", 1.0),
            sources=args.get("sources", []),
            schema_version="v1",
        )
    logger.info(f"asked: {q.text[:40]}")
    return ans


# ─────────────────────────────────────────────────────────────────────────────
# retry wrapper
# ─────────────────────────────────────────────────────────────────────────────
async def ask_llm_with_retry(
    q: Question, tries: int = 3, fail_rate: float = 0.0
) -> Answer:
    """Retry up to `tries` times. Wait 1 s, 2 s, 4 s between attempts."""
    for attempt in range(tries):
        try:
            ans = await ask_llm(q, fail_rate=fail_rate)
            ans.retries = attempt
            return ans
        except FakeLLMError:
            if attempt == tries - 1:
                raise
            logger.warning(f"retry attempt {attempt + 1} for: {q.text[:40]}...")
            await asyncio.sleep(2**attempt)  # 1, 2, 4, ... seconds
    raise RuntimeError("Unreachable")


# ─────────────────────────────────────────────────────────────────────────────
# batch runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_batch(questions: list[Question], fail_rate: float = 0.0) -> list[Answer]:
    """Fire every question in parallel via asyncio.gather (with retries)."""
    tasks = [ask_llm_with_retry(q, fail_rate=fail_rate) for q in questions]
    return await asyncio.gather(*tasks)


async def run_in_batches(
    questions: list[Question], batch_size: int = 5, fail_rate: float = 0.0
) -> list[Answer]:
    out: list[Answer] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        logger.info(
            f"Running batch {i // batch_size + 1} with {len(batch)} questions..."
        )
        batch_results = await run_batch(batch, fail_rate=fail_rate)
        out.extend(batch_results)
    return out


# ─── Streaming endpoint ─────────────────────────────────────────────────────
async def stream_answer(
    question: str, settings: Settings | None = None
) -> AsyncIterator[str]:
    """Yield content tokens as they arrive from the LLM.

    Real OpenAI streaming — no asyncio.sleep, no word-splitting.
    """
    settings = settings or Settings()

    if settings.use_fake:
        full = await fake_ask_llm(question)
        for word in full.split(" "):
            await asyncio.sleep(0.05)
            yield word + " "
        return

    client = AsyncOpenAI()

    stream = await client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "user", "content": question}],
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    settings = Settings()
    logger.info(f"config: {settings.model_dump(mode='json')}")
    questions = load_questions(settings.questions_csv)
    logger.info(f"loaded {len(questions)} questions")

    start_time = time.time()
    answers = asyncio.run(
        run_in_batches(
            questions, batch_size=settings.batch_size, fail_rate=settings.fail_rate
        )
    )
    end_time = time.time()
    elapsed = end_time - start_time
    summary = summarize_run(
        answers,
        started_at=start_time,
        elapsed=elapsed,
        fail_rate=settings.fail_rate,
        use_fake=settings.use_fake,
    )
    logger.info(f"summary: {summary.model_dump(mode='json')}")
    settings.results_json.write_text(
        json.dumps(
            {
                "summary": summary.model_dump(mode="json"),
                "answers": [a.model_dump(mode="json") for a in answers],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {len(answers)} answers to {settings.results_json} in {elapsed:.2f} seconds"
    )

    # SQLite persistence
    from .store import connect, write_answers, write_run

    with connect(settings.results_db) as con:
        run_id = write_run(con, summary)
        n = write_answers(con, run_id, answers)
    logger.info(f"persisted run {run_id} with {n} answers to {settings.results_db}")
