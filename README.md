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

## Week 1
- [x] Set up repo + secrets discipline
- [x] Build `hello_llm.py` (Lab Step 2)
- [ ] Write ADR v1 (Lab Step 3)