# W8 PII Audit — capstone_chunks_v2

**Date:** 2026-11-11  
**Corpus:** data/corpus/ (1 JSONL file, 10 cases)  
**Total chunks ingested:** 75  
**PII flags raised:** 28

I checked the flagged text. All 28 flags were false positives: medical terms and figure references, not personal information.

## 10 samples from Qdrant

I fetched 10 chunks using `client.scroll(limit=10)` and compared the redactions with the original cases.

| Chunk ID | Text preview | Notes |
|---|---|---|
| `case_83092#4` | “mass effect (&lt;PERSON&gt;. )” | Two `Fig.` references were wrongly redacted. |
| `case_2481#0` | “A 76-year-old man with severe chronic obstructive pulmonary disease…” | No PII found. |
| `case_75567#8` | “Patient's hospital course was complicated by repeated pulmonary infections.” | No PII found. |
| `case_52085#5` | “He was treated symptomatically with combination of levodopa-carbidopa…” | No PII found. |
| `case_133814#6` | “&lt;PERSON&gt; agglutination (&lt;PERSON&gt;)” | `Brucella`, `wright` and `Moxifloxacin` were wrongly redacted. |
| `case_2050#6` | “Her drug lymphocyte stimulation test was positive…” | No PII found. |
| `case_52085#0` | “a 25-year-old young man had high-grade fever, headache and vomiting…” | No PII found. |
| `case_181392#3` | “&lt;PERSON&gt; and ds-DNA” | `ANA` was wrongly redacted. `ds-DNA` was kept. |
| `case_152245#6` | “History of nicotine addition; 7- Anxiety; 8- Chronic pain…” | No PII found. |
| `case_133814#1` | “the white blood cell (WBC) count was 6900…” | No PII found. |

Three of these 10 chunks had six redactions. All six were false positives.

## Next steps

- [ ] Fix the scrubber so it keeps medical terms and figure references.
- [ ] Check the remaining chunks for missed PII.
- [ ] Re-ingest the original cases after fixing the scrubber.
