from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    questions_csv: Path = Path("data/questions.csv")
    results_json: Path = Path("result.json")
    results_db: Path = Path("result.db")
    batch_size: int = Field(5, gt=0, le=20)
    fail_rate: float = Field(0.0, ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"
    use_fake: bool = True
