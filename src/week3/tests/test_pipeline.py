from unittest.mock import AsyncMock, patch
import pytest

from src.week2.pipeline.fake_llm import Question, Answer, FakeLLMError


@pytest.mark.asyncio
async def test_ask_llm_calls_fake_once():
    fake_answer = Answer(
        question="What is the leave policy?",
        text="mock response",
        cost_usd=0,
        retries=0,
    )

    with patch(
        "src.week2.pipeline.pipeline.fake_ask_llm", AsyncMock(return_value=fake_answer)
    ) as m:
        from src.week2.pipeline.pipeline import ask_llm

        result = await ask_llm(q=Question(text="What is the leave policy?"))

    assert m.call_count == 1
    assert result.text == "mock response"


@pytest.mark.asyncio
async def test_retry_three_times_on_failure():
    with (
        patch(
            "src.week2.pipeline.pipeline.fake_ask_llm",
            AsyncMock(side_effect=FakeLLMError("simulated")),
        ) as m_call,
        patch("src.week2.pipeline.pipeline.asyncio.sleep", AsyncMock()),
    ):
        from src.week2.pipeline.pipeline import ask_llm_with_retry

        with pytest.raises(FakeLLMError):
            await ask_llm_with_retry(
                q=Question(text="What is the leave policy?"), tries=3
            )

    assert m_call.call_count == 3
