import asyncio
import csv
import json
import time
from pathlib import Path

from .logging_config import get_logger
from .settings import Settings, RunSummary


_settings_for_import = Settings()

if _settings_for_import.use_fake:
    from .fake_llm import Answer, FakeLLMError, Question, fake_ask_llm
else:
    from openai import AsyncOpenAI
    from pydantic import BaseModel

    _client = AsyncOpenAI()

    class Question(BaseModel):
        text: str

    class Answer(BaseModel):
        question: str
        text: str
        cost_usd: float
        retries: int = 0


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
        )
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        completion_tokens = usage.completion_tokens if usage is not None else 0
        cost_usd = prompt_tokens * 0.00000015 + completion_tokens * 0.00000060
        ans = Answer(
            question=q.text,
            text=resp.choices[0].message.content,
            cost_usd=cost_usd,
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
