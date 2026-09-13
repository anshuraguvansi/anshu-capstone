import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

# Make the `src` package reachable when invoked as `python scripts/X.py`.
# Without this, Python's sys.path[0] is `scripts/` (the script's directory),
# not the project root, so `from src.pipeline...` raises ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.week2.pipeline.models import Answer, Question
from src.week2.pipeline.pipeline import ask_llm
from src.week2.pipeline.settings import RunSummary, Settings
from src.week2.pipeline.store import connect, save_answer, write_run


async def run_one_model(questions: list[str], model: str, db_path: Path) -> dict:
    """Run all questions through one model. Returns aggregate stats."""
    settings = Settings(model=model)
    started_at = time.time()
    start = time.perf_counter()
    total_cost = 0.0
    total_retries = 0
    answers: list[tuple[str, Answer]] = []

    for q in questions:
        answer = await ask_llm(Question(text=q), settings)
        answers.append((q, answer))
        total_cost += answer.cost_usd
        total_retries += answer.retries

    elapsed = time.perf_counter() - start
    summary = RunSummary(
        started_at=started_at,
        elapsed_seconds=elapsed,
        n_questions=len(answers),
        n_successed=len(answers),
        n_retries_total=total_retries,
        total_cost_usd=total_cost,
        fail_rate=0.0,
        use_fake=settings.use_fake,
    )
    with connect(db_path) as conn:
        run_id = write_run(conn, summary)
        for q, answer in answers:
            save_answer(
                conn,
                run_id=run_id,
                question=q,
                content=answer.text,
                retries=answer.retries,
                cost_usd=answer.cost_usd,
                model=model,
                confidence=answer.confidence,
                sources=answer.sources,
                schema_version=answer.schema_version,
            )

    return {
        "model": model,
        "n": len(questions),
        "total_cost_usd": round(total_cost, 6),
        "total_retries": total_retries,
        "elapsed_s": round(elapsed, 2),
        "avg_cost_usd": round(total_cost / len(questions), 6),
    }


def load_questions(path: Path) -> list[str]:
    """Read questions from a one-column CSV (header optional)."""
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if rows and rows[0] and rows[0][0].lower().startswith("question"):
        rows = rows[1:]  # skip header
    return [row[0] for row in rows if row and row[0].strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run two-model comparison for W4 Lab Step 3"
    )
    p.add_argument(
        "--models",
        default="gpt-4o-mini,gpt-4o",
        help="Comma-separated model ids (default: gpt-4o-mini,gpt-4o)",
    )
    p.add_argument(
        "--questions",
        default="data/questions.csv",
        help="Path to questions CSV (default: data/questions.csv)",
    )
    p.add_argument(
        "--db",
        default="data/results.db",
        help="Path to SQLite db (default: data/results.db)",
    )
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    questions = load_questions(Path(args.questions))
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Running {len(questions)} questions through: {', '.join(models)}")
    print(f"Persisting to: {db_path}")
    print()

    summaries = []
    for m in models:
        print(f"  → {m} …")
        summary = await run_one_model(questions, m, db_path)
        summaries.append(summary)
        print(
            f"    done — n={summary['n']}  "
            f"cost=${summary['total_cost_usd']}  "
            f"avg=${summary['avg_cost_usd']}/q  "
            f"retries={summary['total_retries']}  "
            f"time={summary['elapsed_s']}s"
        )

    print()
    print("=" * 60)
    print(f"{'Model':<20}  {'n':>4}  {'Total $':>10}  {'Avg $/q':>10}  {'Time':>8}")
    print("-" * 60)
    for s in summaries:
        print(
            f"{s['model']:<20}  {s['n']:>4}  "
            f"{s['total_cost_usd']:>10.6f}  "
            f"{s['avg_cost_usd']:>10.6f}  "
            f"{s['elapsed_s']:>7.2f}s"
        )
    if len(summaries) == 2:
        a, b = summaries
        ratio = b["total_cost_usd"] / a["total_cost_usd"] if a["total_cost_usd"] else 0
        print()
        print(
            f"  → {b['model']} cost {ratio:.1f}× more than {a['model']} on the same questions."
        )
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
