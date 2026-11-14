# Document Update Runbook

Use this when a clinical case changes and retrieval needs the updated note.

## Steps

1. Update the case's `text` in `data/corpus/cases.jsonl`, keeping its `id` unchanged.
2. Set `source` in `scripts/update_document.py` to that case ID. It currently uses `case_2481`.
3. Run from the project root:

   ```bash
   uv run --env-file .env python -m scripts.update_document
   ```

   This marks live chunks deleted, inserts replacements with fresh point IDs and clears saved answers. Existing cases should report a non-zero tombstoned count on every update.

4. Restart the API if it is running, so its in-memory BM25 index reloads. Then verify with a question about the updated case:

   ```bash
   uv run --env-file .env python -c "from src.week10.rag.qdrant_rag import ask_rag; print(ask_rag('What symptoms brought the 76-year-old man with severe COPD for evaluation?')['answer'])"
   ```

   The answer should reflect the updated note. In Qdrant, filter by `source_id = case_2481`: old points should have `deleted_at`, and fresh points should not.

5. Record the date, case ID, corpus file and tombstoned count in `docs/kb-changelog.md` (create it on the first entry).

## Rollback

In Qdrant, identify the previous batch using `source_id` and `ingested_at`. Delete the replacement batch by its exact point IDs, then remove `deleted_at` from only the previous batch's point IDs. Do not restore every deleted point for the source; that would activate multiple old versions. Restore the matching case text in the corpus file.

Clear saved answers and restart the API before verifying again:

```bash
uv run --env-file .env python -c "from src.week10.rag import cache; cache.clear()"
```
