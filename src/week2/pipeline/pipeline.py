import asyncio
import json
import logging
import sys
import time

from .fake_llm import Answer, FakeLLMError, Question, fake_ask_llm


# ─────────────────────────────────────────────────────────────────────────────
# structured (JSON) logging
# ─────────────────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": round(time.time(), 3),
                "level": record.levelname,
                "msg": record.getMessage(),
            }
        )


logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(JsonFormatter())
logger.addHandler(stream_handler)


# ─────────────────────────────────────────────────────────────────────────────
# single LLM call
# ─────────────────────────────────────────────────────────────────────────────
async def ask_llm(q: Question, fail_rate: float = 0.0) -> Answer:
    """One LLM call. Fake for now; real client wired in via Settings.use_fake later."""
    logger.info(f"asked: {q.text[:40]}...")
    return await fake_ask_llm(q, fail_rate=fail_rate)


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


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fail_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    samples = [
        Question(text="What is RAG in one sentence?"),
        Question(text="Name three uses of vector databases."),
        Question(text="Why might an LLM hallucinate?"),
    ]
    answers = asyncio.run(run_batch(samples, fail_rate=fail_rate))
    for a in answers:
        print(f"- {a.text[:80]}")
