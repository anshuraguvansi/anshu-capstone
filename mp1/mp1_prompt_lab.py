import asyncio
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from openai import APIError, AsyncOpenAI, ChatCompletion
from pydantic import BaseModel, Field, ValidationError

# ************************************************************
#                       Model & Cost Setup
# ************************************************************

# Cost rates ($ per token)
RATES = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
}


class Model(StrEnum):
    """Enumeration of available language models and their associated token costs."""

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
    """Data model for a job snippet."""

    id: str
    snippet: str


class GoldenSnippet(BaseModel):
    """Data model for a golden snippet."""

    id: str
    company: str
    role: str
    notes: str
    years_experience_required: int | None = None


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_job_snippets(
    file_path: str | Path = DATA_DIR / "job_snippets.jsonl",
) -> list[JobSnippet]:
    """Load job snippets from a JSONL file."""
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [JobSnippet(**item) for item in data]


def load_golden_snippets(
    file_path: str | Path = DATA_DIR / "golden_set.jsonl",
) -> list[GoldenSnippet]:
    """Load golden snippets from a JSONL file."""
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [GoldenSnippet(**item) for item in data]


# ************************************************************
#                       Prompt Strategies
# ************************************************************
def prompt_zero_shot(snippet: str) -> list[dict[str, str]]:
    """Generate a zero-shot prompt for extracting job information."""
    return [
        {
            "role": "system",
            "content": """
            Extract the role, company, and years of experience required from the job snippet in JSON format.
            """,
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_few_shot(snippet: str) -> list[dict[str, str]]:
    """Generate a few-shot prompt for extracting job information."""
    return [
        {
            "role": "system",
            "content": """
            Extract the role, company, and years of experience required from the job snippet in JSON format.
            
            Example:
            <job_snippet>Software Engineer at OpenAI, 3 years experience required</job_snippet>
            <output_format>
                {
                    "role": "Software Engineer",
                    "company": "OpenAI",
                    "years_experience_required": 3
                }
            </output_format>
            
            Another Example:
            <job_snippet>Data Scientist at Google</job_snippet>
            <output_format>
                {
                    "role": "Data Scientist",
                    "company": "Google",
                    "years_experience_required": null
                }
            </output_format>
            """,
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_structured(snippet: str) -> list[dict[str, str]]:
    """Generate a structured prompt for extracting job information in JSON format."""
    return [
        {
            "role": "system",
            "content": """
            You are an expert job posting analyst. Extract the role, company, and years of experience required from the job snippet in JSON format.
            
            <output_format>
               {
                   "role": string | null,
                   "company": string | null,
                   "years_experience_required": int | null
               }
            </output_format>
            """,
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


def prompt_cot(snippet: str) -> list[dict[str, str]]:
    """Generate a chain-of-thought prompt for extracting job information in JSON format."""
    return [
        {
            "role": "system",
            "content": """
            You are an expert job posting analyst. Analyze the job posting carefully and extract the role, company, and years of experience required, perform an internal verification pass to ensure the accuracy of the extracted information. Provide the output in JSON format.
            
            <output_format>
               {
                   "role": string | null,
                   "company": string | null,
                   "years_experience_required": int | null
               }
            </output_format>
            """,
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


class PromptStrategy(StrEnum):
    """Enumeration of different prompt strategies for job information extraction."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    STRUCTURED = "structured"
    COT = "cot"

    @classmethod
    def all(cls) -> list[PromptStrategy]:
        """Returns a list of all PromptStrategy enum objects."""
        return [member for member in cls]

    def build_prompt(self, prompt_input: str) -> list[dict[str, str]]:
        match self.value:
            case "zero_shot":
                return prompt_zero_shot(prompt_input)
            case "few_shot":
                return prompt_few_shot(prompt_input)
            case "structured":
                return prompt_structured(prompt_input)
            case "cot":
                return prompt_cot(prompt_input)


# ************************************************************
#                       Extraction
# ************************************************************


class ExtractedData(BaseModel):
    """Data model for storing extracted job information."""

    role: str | None = None
    company: str | None = None
    years_experience_required: int | None = None


@dataclass
class ExecutionResult:
    """Data model for storing the result of executing a prompt strategy on a job snippet."""

    strategy: str
    snippet: JobSnippet
    cost: float
    latency: float
    raw_response: str | None = None
    result: ExtractedData | None = None


def parse_response(output: str | None) -> ExtractedData | None:
    """Parse the raw response text from the LLM into an ExtractedData object."""

    if output is None:
        return None

    try:
        return ExtractedData.model_validate_json(output)
    except ValidationError, TypeError:
        return None


def create_execution_result(func):
    """Decorator to create an ExecutionResult from the output of a prompt execution function."""

    async def wrapper(
        client: AsyncOpenAI,
        strategy: PromptStrategy,
        snippet: JobSnippet,
        model: Model,
    ) -> ExecutionResult:
        """Wrapper function that executes the prompt function and constructs an ExecutionResult."""

        start_time = time.time()
        try:
            raw_response = await func(client, strategy, snippet, model)
            result = parse_response(raw_response.choices[0].message.content)
        except (APIError, ValidationError) as e:
            print(e)
            raw_response = None
            result = None

        latency = time.time() - start_time

        # calculate cost based on token usage
        cost = 0.0
        if raw_response is not None:
            cost = (
                raw_response.usage.prompt_tokens * model.input_token_cost
                + raw_response.usage.completion_tokens * model.output_token_cost
            )

        return ExecutionResult(
            strategy=strategy.value,
            snippet=snippet,
            cost=cost,
            latency=latency,
            raw_response=raw_response
            if raw_response is None
            else raw_response.choices[0].message.content,
            result=result,
        )

    return wrapper


@create_execution_result
async def run_one(
    client: AsyncOpenAI, strategy: PromptStrategy, snippet: JobSnippet, model: Model
) -> ChatCompletion:
    """Execute a single prompt strategy on a job snippet and return the execution result."""
    prompt = strategy.build_prompt(snippet.snippet)
    response = await client.chat.completions.create(
        model=model.value,
        messages=prompt,
        temperature=0.0,
    )
    return response


async def run_all(client: AsyncOpenAI) -> list[ExecutionResult]:
    """Execute all prompt strategies on all job snippets and return a list of execution results."""
    snippets = load_job_snippets()
    tasks = [
        run_one(client, strategy, snippet, Model.GPT_4O_MINI)
        for strategy in PromptStrategy.all()
        for snippet in snippets
    ]
    return await asyncio.gather(*tasks)


# ************************************************************
#                       Evaluation
# ************************************************************


class EvaluationResult(BaseModel):
    """Data model for storing the result of evaluating an extracted job snippet against the golden snippet."""

    score: int = Field(ge=1, le=4)
    reasoning: str


def score_accuracy(extracted: ExtractedData | None, gold: GoldenSnippet) -> int:
    """Compute the accuracy score of the extracted data against the golden snippet."""
    if extracted is None:
        return 0
    score = 0
    if extracted.role and extracted.role.lower().strip() == gold.role.lower().strip():
        score += 1
    if (
        extracted.company
        and extracted.company.lower().strip() == gold.company.lower().strip()
    ):
        score += 1
    if extracted.years_experience_required == gold.years_experience_required:
        score += 1
    return score


async def score_llm_judge(
    client: AsyncOpenAI,
    snippet_text: str,
    extracted: ExtractedData | None,
    gold: GoldenSnippet,
) -> int:
    """Use an LLM to evaluate the extracted data against the golden snippet and return a score."""
    if extracted is None:
        return 1

    system_prompt = """
    You are an expert quality assurance judge evaluating an information extraction system. 
    Your task is to compare the 'Extracted Data' against the 'Golden Data' (the ground truth snippet data) and assign a score based strictly on the rubric below.

    Evaluate these three specific fields:
    1. role
    2. company
    3. years of experience required

    Rubric:
    - 4 (Excellent): All three fields perfectly match the Golden Data.
    - 3 (Good): Exactly two of the three fields match the Golden Data, and the third field is either empty or incorrect (but NOT fabricated/made up).
    - 2 (Fair): Exactly one of the three fields matches the Golden Data, OR one or more fields contain completely fabricated information not present in the Snippet Text.
    - 1 (Poor): None of the fields match the Golden Data, or the extraction is entirely unparsable/hallucinated.

    Provide your evaluation in a valid JSON format with two keys:
    - "reasoning": A brief, one-sentence explanation justifying the score based on the rubric.
    - "score": The integer score (1, 2, 3, or 4).
    """

    user_prompt = f"""
    Snippet Text:
    {snippet_text}

    Extracted Data:
    {extracted.model_dump_json()}
    
    Golden Data:
    {gold.model_dump_json()}
    """

    response = await client.chat.completions.parse(
        model=Model.GPT_4O.value,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format=EvaluationResult,
    )
    return response.choices[0].message.parsed.score


async def evaluate_extraction(client: AsyncOpenAI) -> list[dict[str, Any]]:
    """Evaluate all extraction results against the golden snippets and return a list of evaluation scores and reasoning."""

    gold_snippets = load_golden_snippets()
    results = await run_all(client)
    scores = []
    for result in results:
        gold = next((g for g in gold_snippets if g.id == result.snippet.id), None)
        if gold is None:
            print(f"Warning: no golden record for {result.snippet.id}")
            continue

        accuracy = score_accuracy(result.result, gold)
        score = await score_llm_judge(
            client, result.snippet.snippet, result.result, gold
        )
        scores.append(
            {
                "strategy": result.strategy,
                "accuracy": accuracy,
                "llm_judge_score": score,
                "parse_success": result.result is not None,
                "cost_usd": result.cost,
                "latency_s": result.latency,
            }
        )
    return scores


# ************************************************************
#                       Entry Point
# ************************************************************


def print_scores(scores) -> None:
    """Print the evaluation scores and a summary grouped by strategy."""

    df = pd.DataFrame(scores)
    print(df)
    summary = (
        df.groupby("strategy")
        .agg(
            {
                "accuracy": "mean",
                "parse_success": "mean",
                "llm_judge_score": "mean",
                "cost_usd": "sum",
                "latency_s": "median",
            }
        )
        .round(
            {
                "accuracy": 2,
                "parse_success": 2,
                "llm_judge_score": 2,
                "cost_usd": 6,
                "latency_s": 3,
            }
        )
    )
    summary.columns = [
        "Accuracy (mean of 3)",
        "Parse rate",
        "Judge score",
        "Total cost ($)",
        "Latency p50 (s)",
    ]
    print(summary)


async def run() -> None:
    """Main entry point for running the evaluation and printing the results."""

    async with AsyncOpenAI() as client:
        results = await evaluate_extraction(client)
    print_scores(results)


if __name__ == "__main__":
    asyncio.run(run())
