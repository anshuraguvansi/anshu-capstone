from openai import AsyncOpenAI
from enum import StrEnum
from pydantic import BaseModel
import json

# ************************************************************
#                       Project Setup
# ************************************************************

client = AsyncOpenAI()

# Cost rates ($ per token)
RATES = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
}


class Model(StrEnum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

    @property
    def input_token_cost(self) -> float:
        return RATES[self.value]["in"]

    @property
    def output_token_cost(self) -> float:
        return RATES[self.value]["out"]


# ************************************************************
#                       Data Loaders
# ************************************************************


class JobSnippet(BaseModel):
    id: str
    snippet: str


class GoldenSnippet(BaseModel):
    id: str
    company: str
    role: str
    notes: str
    years_experience_required: int | None = None


def load_job_snippets(file_path: str = "data/job_snippets.jsonl") -> list[JobSnippet]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [JobSnippet(**item) for item in data]


def load_golden_snippets(
    file_path: str = "data/golden_set.jsonl",
) -> list[GoldenSnippet]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [GoldenSnippet(**item) for item in data]


# ************************************************************
#                       Prompt Strategies
# ************************************************************
def prompt_zero_shot(snippet: str):
    return [
        {
            "role": "system",
            "content": "You are a job posting analyst. Extract the role, company, and years of experience required from the job snippet.",
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_few_shot(snippet: str):
    return [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_structured(snippet: str):
    return [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_cot(snippet: str):
    return [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


STRATEGIES = {
    "zero_shot": prompt_zero_shot,
    "few_shot": prompt_few_shot,
    "structured": prompt_structured,
    "cot": prompt_cot,
}


# ************************************************************
#                       Async Batching
# ************************************************************


class ExtractedData(BaseModel):
    role: str
    company: str
    years_experience_required: int | None = None


async def run_one(strategy: str, snippet: JobSnippet, model: Model) -> any:
    strategy = STRATEGIES[strategy]
    prompt = strategy(snippet.snippet)
    return await client.chat.completions.parse(
        model=model.value,
        messages=prompt,
        temperature=0.0,
        response_format=ExtractedData,
    )
