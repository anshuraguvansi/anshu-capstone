import time
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    questions_csv: Path = Path("data/questions.csv")
    results_json: Path = Path("src/week2/results.json")
    results_db: Path = Path("src/week2/results.db")
    batch_size: int = Field(5, gt=0, le=20)
    fail_rate: float = Field(0.0, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    use_fake: bool = False


class RunSummary(BaseModel):
    started_at: float = time.time()
    elapsed_seconds: float = Field(ge=0.0)
    n_questions: int = Field(ge=0)
    n_successed: int = Field(ge=0)
    n_retries_total: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    fail_rate: float = Field(ge=0.0, le=1.0)
    use_fake: bool
