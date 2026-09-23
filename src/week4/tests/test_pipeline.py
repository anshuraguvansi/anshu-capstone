import asyncio
import json

import pytest

from pipeline.models import Answer, Question
from pipeline.pipeline import ANSWER_TOOL, ask_llm
from pipeline.settings import Settings


# ─── Answer model backward compatibility ────────────────────────────────────
class TestAnswerBackwardCompatibility:
    def test_w3_shape_still_constructs(self):
        """A caller that only supplies the W3 fields must keep working.
        This is the central guarantee of the W4 additive change."""
        a = Answer(content="hello", cost_usd=0.001, retries=2)
        assert a.content == "hello"
        assert a.cost_usd == 0.001
        assert a.retries == 2

    def test_w3_shape_picks_up_defaults_for_new_fields(self):
        """Old shape implicitly fills the new W4 fields with safe defaults."""
        a = Answer(content="x")
        assert a.confidence == 1.0
        assert a.sources == []
        assert a.schema_version == "v1"

    def test_schema_version_default_is_v1(self):
        a = Answer(content="x")
        assert a.schema_version == "v1"

    def test_confidence_range_is_validated(self):
        """confidence must be between 0.0 and 1.0 inclusive."""
        with pytest.raises(Exception):
            Answer(content="x", confidence=1.5)
        with pytest.raises(Exception):
            Answer(content="x", confidence=-0.1)

    def test_sources_default_is_independent(self):
        """Each Answer must get its own sources list, not a shared default.
        (Caught by Field(default_factory=list).)"""
        a = Answer(content="A")
        b = Answer(content="B")
        a.sources.append("only-on-A")
        assert b.sources == []


# ─── Tool schema shape ──────────────────────────────────────────────────────
class TestAnswerTool:
    def test_tool_has_function_name(self):
        assert ANSWER_TOOL["type"] == "function"
        assert ANSWER_TOOL["function"]["name"] == "answer_question"

    def test_tool_requires_three_fields(self):
        required = ANSWER_TOOL["function"]["parameters"]["required"]
        assert set(required) == {"content", "confidence", "sources"}

    def test_tool_properties_have_correct_types(self):
        props = ANSWER_TOOL["function"]["parameters"]["properties"]
        assert props["content"]["type"] == "string"
        assert props["confidence"]["type"] == "number"
        assert props["sources"]["type"] == "array"
        assert props["sources"]["items"]["type"] == "string"


# ─── ask_llm via fake path ──────────────────────────────────────────────────
class TestAskLlmFakePath:
    def test_use_fake_returns_valid_answer(self):
        """No API key, no network. Just verifies the function signature
        and that the Answer it returns is well-formed."""
        settings = Settings(use_fake=True, openai_api_key="not-used")
        answer = asyncio.run(ask_llm(Question(question="What is RAG?"), settings))
        assert isinstance(answer, Answer)
        assert answer.content.startswith("[FAKE]")
        assert answer.cost_usd == 0.0
        assert answer.schema_version == "v1"


# ─── Tool-call parse logic (would-be unit test for the parse step) ──────────
class TestToolCallParse:
    def test_well_formed_arguments_parse_cleanly(self):
        """Simulates what resp.choices[0].message.tool_calls[0].function.arguments
        looks like coming back from OpenAI — a JSON string the SDK doesn't decode."""
        args_json = json.dumps(
            {
                "content": "RAG combines retrieval with generation.",
                "confidence": 0.87,
                "sources": ["wiki:RAG", "docs:rag-intro"],
            }
        )
        args = json.loads(args_json)
        ans = Answer(
            content=args["content"],
            confidence=args["confidence"],
            sources=args["sources"],
            cost_usd=0.000045,
            retries=0,
        )
        assert ans.confidence == 0.87
        assert ans.sources == ["wiki:RAG", "docs:rag-intro"]
        assert ans.schema_version == "v1"
