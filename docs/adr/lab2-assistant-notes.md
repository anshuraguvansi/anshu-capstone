# Lab 2 — Coding-assistant verification note

## The change
Compute a real per-call cost from resp.usage instead of the placeholder 0.0001.
cost_usd = (resp.usage.prompt_tokens * 0.00000015 + resp.usage.completion_tokens * 0.00000060)

## The ask
In pipeline.py calculate actual cost of the inference from openAI response and update in the Answer.

## What it produced
Main calculation formula is same the only thing changed is guard arund prompt_tokens and completion_tokens, Coding agent added if response have these keys if not set value to 0.
## What I verified before accepting
- Diff read: If the formula of calcualtion is matching.
- Test run: pipeline again and no issue is observed.
- Security check: new dependencies

## What I changed before committing
both are fine answers