# Lab 4 — Model comparison: gpt-4o-mini vs gpt-4o

**Cohort member:** Anshu Kumar
**Date:** 13/09/2026

## Numbers (filled in by `scripts/compare_models.py`)

```
Model            n    Total $     Avg $/q     Time
---------------------------------------------------
gpt-4o-mini     21    0.001665    0.000079    32.22s
gpt-4o          21    0.033535    0.001597    42.45s
```

> gpt-4o cost 20.1x more than gpt-4o-mini on the same questions.

## Two-paragraph eyeball reflection

### Paragraph 1 — where the gap mattered

For these 21 short, factual questions, I could not see a large quality gap
between the two models. Both gave clear explanations of rate limiting, HTTP 422,
dependency injection, SQL statements, and Python virtual environments. The gap
was more noticeable on the questions that asked for several reasons or steps.
For example, gpt-4o gave a more structured explanation of why an LLM can
hallucinate and a more complete outline of a typical RAG pipeline. Its answer on
parameterised SQL was also a little more direct about avoiding SQL injection.
Still, the mini model was accurate and usable for nearly all of the questions in
this comparison.

### Paragraph 2 — your rough rule for when to reach for the bigger model

I would use gpt-4o when the question needs more careful reasoning, a structured
multi-part explanation, or a response that will be shown directly to a user.
For routine definitions and straightforward developer questions, gpt-4o-mini is
the better default because its answers were already good enough here. The larger
model cost about 21 times more per question, so using it for every request would
add cost without a matching improvement on simple tasks. I
would reserve it for the smaller set of requests where the extra clarity or
depth has real value.

## Confidence calibration (optional)

The lab pipeline asks the model to return a `confidence` value in `[0, 1]`.
Skim the persisted rows in SQLite:

```bash
sqlite3 data/results.db \
  "SELECT model, AVG(confidence), AVG(cost_usd) FROM answers WHERE run_id IN (4, 5) GROUP BY model;"
```

The average reported confidence was similar: 0.905 for gpt-4o-mini and 0.928
for gpt-4o, even though their average costs were very different.
