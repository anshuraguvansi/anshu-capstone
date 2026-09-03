# Capstone — Knowledge Assistant
A 30-week build of a Q&A assistant over a small document corpus,
completed as part of
the *Agentic AI & RAG Engineering* programme.

## Corpus
Public regulatory PDFs (a SEBI / RBI / GDPR document set), source: https://www.sebi.gov.in/legal.html

## Structure
- `src/` — application code
- `docs/adr/` — Architecture Decision Records (one per major design
choice)
- `docs/runs/` — saved LLM outputs for evidence and reference

## Setup 
This project is using `UV` as depdency manager. To run this project do the following:
1. Use `uv sync` : install dependecies and setup the virtual environment.
2. Add `.env` file with a valid OpenAI key.
3. Run `uv run --env-file .env python src/hello_llm.py`
4. Run server `uv run --env-file .env uvicorn src.week3.api.main:app --reload --port 8000`
5. Run Streamlit UI `streamlit run scr/week3/ui/app.py --server.port 8501`

## Week 1
- [x] Set up repo + secrets discipline
- [x] Build `hello_llm.py` (Lab Step 2)
- [ ] Write ADR v1 (Lab Step 3)

## Week 2
- [x] src/pipeline/ package with pipeline.py , fake_llm.py , logging_config.py ,
settings.py , store.py , query_results.py .
- [x] data/questions.csv with 20 rows; logs/pipeline.log with one JSON record per
call.
- [x] Pipeline runs end-to-end against the real OpenAI API for all 20 questions in batches of 5
with retries.
- [x] Two Pydantic models authored by you — Settings (with Field constraints) and
RunSummary . You can demonstrate one ValidationError firing on a bad input.
- [x] results.json written with summary + answers; results.db populated with rows in
both runs and answers ; query_results.py returns matches for --runs and a substring
search.
- [x] .env is not tracked.
- [x] docs/lab2-assistant-notes.md describes one improvement and the verification
workflow.
- [ ] Repo URL submitted in the tracker.


## Week 3
- [x] api/main.py — FastAPI service with /ask (streaming), /ask_batched (non-
streaming), /health.
- [x] ui/app.py — Streamlit UI hitting /ask.
- [x] tests/test_pipeline.py — 2 mocked unit tests (fake call + retry).
- [x] tests/test_api.py — 2 API surface tests (validation + health).
- [x] tests/conftest.py + pytest.ini — pytest config.
- [x] scripts/stress_test.py — 50-parallel stress harness.
- [x] docs/wk3-stress-notes.md — 4 stress findings documented.
- [x] docs/adr/0002-api-contract.md — /v1/ask contract locked.
- [x] Git history with at least 4 W3 commits.