import asyncio
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

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


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_job_snippets(
    file_path: str | Path = DATA_DIR / "job_snippets.jsonl",
) -> list[JobSnippet]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [JobSnippet(**item) for item in data]


def load_golden_snippets(
    file_path: str | Path = DATA_DIR / "golden_set.jsonl",
) -> list[GoldenSnippet]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [GoldenSnippet(**item) for item in data]


# ************************************************************
#                       Prompt Strategies
# ************************************************************
def prompt_zero_shot(snippet: str):
    return [
        {
            "role": "system",
            "content": """
            Extract the role, company, and years of experience required from the job snippet.
            """,
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
            "content": """
            Extract the role, company, and years of experience required from the job snippet.
            
            Example:
            <job_snippet>Software Engineer at OpenAI, 3 years experience required</job_snippet>
            <extracted_data>
                <role>Software Engineer</role>
                <company>OpenAI</company>
                <years_experience_required>3</years_experience_required>
            </extracted_data>
            
            Another Example:
            <job_snippet>Data Scientist at Google</job_snippet>
            <extracted_data>
                <role>Data Scientist</role>
                <company>Google</company>
                <years_experience_required>None</years_experience_required>
            </extracted_data>
            """,
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
            "content": """
            You are a job posting analyst. Extract the role, company, and years of experience required from the job snippet in JSON format.
            
            <output>
               {
                   "role": string | null,
                   "company": string | null,
                   "years_experience_required": int | null
               }
            </output>
            """,
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
            "content": """
            You are a job posting analyst. Analyze the job posting carefully and extract the role, company, and years of experience required. Provide the output in JSON format.
            """,
        },
        {
            "role": "user",
            "content": f"<job_snippet>{snippet}</job_snippet>",
        },
    ]


class PromptStrategy(StrEnum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    STRUCTURED = "structured"
    COT = "cot"

    @classmethod
    def all(cls) -> list[PromptStrategy]:
        """Returns a list of all PromptStrategy enum objects."""
        return [member for member in cls]

    def build_prompt(self, prompt_input: str):
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
    role: str | None = None
    company: str | None = None
    years_experience_required: int | None = None


@dataclass
class ExecutionResult:
    strategy: str
    snippet: JobSnippet
    cost: float
    latency: float
    raw_response: any
    result: ExtractedData | None = None


def create_execution_result(func):
    async def wrapper(
        strategy: PromptStrategy, snippet: JobSnippet, model: Model
    ) -> ExecutionResult:

        start_time = time.time()
        try:
            raw_response = await func(strategy, snippet, model)
            result = raw_response.choices[0].message.parsed
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
            raw_response=raw_response,
            result=result,
        )

    return wrapper


@create_execution_result
async def run_one(strategy: PromptStrategy, snippet: JobSnippet, model: Model) -> any:
    prompt = strategy.build_prompt(snippet.snippet)
    return await client.chat.completions.parse(
        model=model.value,
        messages=prompt,
        temperature=0.0,
        response_format=ExtractedData,
    )


async def run_all():
    snippets = load_job_snippets()
    # return [await run_one(PromptStrategy.ZERO_SHOT, snippets[0], Model.GPT_4O_MINI)]
    tasks = [
        run_one(strategy, snippet, Model.GPT_4O_MINI)
        for strategy in PromptStrategy.all()
        for snippet in snippets
    ]
    return await asyncio.gather(*tasks)


# ************************************************************
#                       Evaluation
# ************************************************************


class EvaluationResult(BaseModel):
    score: int
    reasoning: str


def score_accuracy(extracted: ExtractedData | None, gold: GoldenSnippet) -> int:
    if extracted is None:
        return 0
    score = 0
    if extracted.result.role.lower().strip() == gold.role.lower().strip():
        score += 1
    if extracted.result.company.lower().strip() == gold.company.lower().strip():
        score += 1
    if extracted.result.years_experience_required == gold.years_experience_required:
        score += 1
    return score


async def score_llm_judge(
    snippet_text: str, extracted: ExtractedData | None, gold: GoldenSnippet
) -> int:
    if extracted is None:
        return 0

    system_prompt = """
    You are an expert quality assurance judge evaluating an information extraction system. 
    Your task is to compare the 'Extracted Data' against the 'Golden Data' (the ground truth snippet data) and assign a score based strictly on the rubric below.

    Evaluate these three specific fields:
    1. role
    2. company
    3. years of experience required

    Rubric:
    - 4 (Excellent): All three fields perfectly match the Golden Data.
    - 3 (Good): Exactly two of the three fields match the Golden Data, and the third field is either empty or incorrect (but NOT fabricated/made up out of thin air).
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
        model=Model.GPT_4O_MINI.value,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format=EvaluationResult,
    )
    return response.choices[0].message.parsed.score


async def evaluate_extraction():
    gold_snippets = load_golden_snippets()
    results = await run_all()
    scores = []
    for result in results:
        gold = next((g for g in gold_snippets if g.id == result.snippet.id), None)
        if gold is None:
            continue

        accuracy = score_accuracy(result.result, gold)
        score = await score_llm_judge(result.snippet.snippet, result.result, gold)
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


def print_scores(scores):
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
        .round(3)
    )
    summary.columns = [
        "Accuracy (mean of 3)",
        "Parse rate",
        "Judge score",
        "Total cost ($)",
        "Latency p50 (s)",
    ]
    print(summary)


async def run():
    results = await evaluate_extraction()
    print(f"Got {len(results)} results.")
    print_scores(results)


if __name__ == "__main__":
    asyncio.run(run())
