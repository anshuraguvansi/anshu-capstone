# Clinical Simulation and Feedback

### Context

Medical students and junior residents develop diagnostic reasoning through question banks, written case studies, and supervised clinical simulations. Question banks and written cases are easy to access, but they usually provide the relevant history and findings up front. Learners therefore get fewer opportunities to decide what questions to ask and uncover important details themselves.

Actor-led simulations provide a more interactive experience, but they require preparation, scheduling, and trained staff. This limits how often they can be offered and the range of cases they can cover. This project explores whether synthetic patient conversations based on published clinical cases can provide an additional, repeatable way to practise.

### Problem Statement

Learners have limited opportunities to practise the full process of interviewing a patient, deciding which details are important, and explaining how they reached a diagnosis. Written cases provide most of the relevant information up front, while instructor-led simulations take time to prepare, run, and review. This makes it difficult to offer repeated practice across a wide range of cases and provide timely, case-specific feedback.

### Scope

**Decision:** Build a case-based clinical interview simulation where a learner speaks with a synthetic patient and receives feedback on their diagnosis and reasoning.

* **What it does:** Each simulation begins with a short introduction and an opening complaint from the synthetic patient. The learner leads the rest of the interview by asking questions, and the patient replies using facts from the source note that a patient could reasonably know. The patient does not reveal clinician-only information, such as the final diagnosis or later test results. After the learner submits a diagnosis and supporting reasoning, the platform compares the answer with the diagnosis and key findings recorded in the selected case. It then provides feedback supported by evidence from the source note.
* **Who it is designed for:** Medical students and junior residents who want to practise patient interviews and diagnostic reasoning.
* **What it does not do:** It does not diagnose real patients, recommend treatment, provide emergency guidance, or replace an instructor's assessment. It does not connect to hospital systems or live patient records.
* **Initial data:** The platform will use 200 cases automatically selected from the `AGBonnet/augmented-clinical-notes` dataset using fixed validation rules.

### Stakeholders

The platform is designed around two learner roles.

* **Medical student**
  * *Current workflow:* Works through question banks and written cases where the relevant symptoms and findings are usually provided up front.
  * *What changes:* Decides which questions to ask, identifies relevant details from the patient's answers, and explains a diagnosis before receiving case-based feedback.
* **Junior resident**
  * *Current workflow:* Builds diagnostic experience through clinical work, case discussions, and independent study, but may have limited exposure to uncommon cases.
  * *What changes:* Can work through additional cases as patient interviews and compare their reasoning with the diagnosis and findings recorded in each source case.

### KPI Targets

The following targets will be used to evaluate the completed platform.

A standard simulation consists of an opening complaint, up to 10 learner questions, one submitted diagnosis with supporting reasoning, and one feedback response.

1. **Task success rate: at least 90% of test simulations complete successfully.**
   * *Why this target:* A simulation is successful when the synthetic patient provides an opening complaint, answers the test questions, accepts the learner's diagnosis and reasoning, and returns feedback with source evidence without an application error. A result below 90% would make the complete workflow too unreliable.
2. **Groundedness: at least 90% of clinical claims are supported by the selected source case.**
   * *Why this target:* The synthetic patient and the final feedback should use information from the selected case rather than inventing details. Responses will be evaluated automatically against the retrieved passages and the source note. A 90% target allows a small margin for evaluation error while still requiring strong grounding.
3. **Retrieval hit rate: at least 85% at Top-K=3.**
   * *Why this target:* One of the top three retrieved chunks should contain the information needed to answer the question. The test questions will come from the existing conversations in the selected cases. An 85% target is high enough to make retrieval dependable while allowing for difficult or ambiguous questions.

### Technology Approach

* **Application:** Python will be used for the main application, FastAPI for the API, and Streamlit for the user interface. The selected case ID and conversation history will be stored for the duration of each session.
* **Data preparation:** A Python pipeline will load records from the `AGBonnet/augmented-clinical-notes` dataset and parse the JSON stored in `summary`. Cases with invalid JSON or missing complaints, symptoms, or diagnoses will be skipped automatically. The pipeline will use a fixed random seed to select 200 valid cases so that the same collection can be reproduced.
* **Vector storage:** Qdrant is the selected vector store as of W7, replacing the initial ChromaDB plan. The current implementation stores `chunk_id`, `source_id`, and `text` with each vector. Case-specific filtering and separation of patient-interview and evaluation information remain planned extensions.
* **Patient information:** The patient collection will contain the presenting complaint, patient information, medical history, symptoms, and suitable patient responses from `conversation`. Chunks containing the recorded diagnosis will be excluded automatically.
* **Evaluation information:** A separate collection will contain chunks from `full_note`, examinations, diagnostic tests, recorded conditions, treatments, and outcomes. This information will remain unavailable to the synthetic patient.
* **Case selection and retrieval:** At the start of a simulation, the platform will select one of the 200 cases and keep its `case_id` fixed for the full session. Every retrieval request will filter by that ID, preventing information from different patients from being mixed. Interview questions will search only the patient collection. The submitted diagnosis and reasoning will search only the evaluation collection.
* **Generation:** An OpenAI language model will use the retrieved patient information to answer in the synthetic patient's voice. If the selected case does not contain the requested information, the patient will say that they do not know or cannot remember. After the learner submits an answer, the model will compare it with the diagnosis recorded in the dataset and generate feedback using evidence retrieved from the source note.
* **Evaluation:** Automated test runs will measure task success, groundedness, and retrieval hit rate. Retrieval questions will be taken from the existing case conversations. The prompt version, model, selected case IDs, retrieved chunks, and generated responses will be recorded so that the results can be reproduced.

### Alternatives Considered

1. **A chatbot for SEBI documents — rejected.**
   * I first considered building a chatbot that would answer questions using documents published by the Securities and Exchange Board of India. Although it was a valid RAG use case, it felt too similar to existing document-question-answering tools. I also could not identify a specific group that would use it regularly or a clear problem that it would solve better than document search.
2. **An agent for processing administrative emails — rejected.**
   * Another idea was to extract information from emails and attachments and update an internal system automatically. For example, the agent could identify that medication stock had arrived and update the inventory. This would reduce a real manual task, but it required access to private organisational emails, attachments, and internal systems. That data could not be made available for the project, so the idea was not practical.

### Trade-offs

* **Automatic case selection vs. manual quality checks:** Automatic validation makes the dataset preparation repeatable and avoids creating case data by hand. However, fixed rules may not detect subtle errors, contradictions, or unrealistic conversations. Some lower-quality cases may therefore pass the checks.
* **RAG retrieval vs. using the complete case:** Retrieving a few relevant chunks keeps prompts smaller and makes it easier to show which evidence was used. It also supports larger datasets later. However, individual case notes may already fit within the model's context window, and retrieval introduces the risk of missing an important detail that exists elsewhere in the note.
* **Synthetic conversation vs. human interaction:** Synthetic patients provide repeatable conversations, but they cannot reproduce the behaviour, emotion, or judgement of a real patient or trained actor. The dataset's generated conversations may also contain repetitive answers, missing information, or generation errors.
* **Hosted models vs. local control:** Hosted embedding and language models reduce the work needed to build the platform. In return, the project depends on an external service and has variable cost and response time. The platform will use only the selected public dataset and will not process live patient data.
* **Source-based feedback vs. clinical validation:** The platform can compare an answer with the diagnosis and findings recorded in the dataset, but this does not prove that the feedback is medically complete or educationally effective. The dataset summaries were generated automatically and are not a substitute for review by a qualified medical professional.

### W7 — Qdrant-backed retrieval

**Decision:** Use Qdrant with cosine distance, OpenAI `text-embedding-3-small` (1536 dimensions), 800-character chunks with 80-character overlap, and top-k 3. The default `capstone_chunks` collection contains 75 points. Retain Week 6 JSON caches and retrieval code as backups and baselines.

**Evidence:** Migration preserves 27/30 evidence hits (90%) and 30/30 source hits. In the controlled 500/50 comparison, OpenAI achieves 24/30 evidence hits versus MiniLM's 22/30, and 30/30 source hits versus 29/30. MiniLM is faster (276.0 ms median versus 966.5 ms) and has no embedding API charge, but OpenAI's measured query embedding cost is only $0.000000374 per question. Keep MiniLM as an experimental option.

**Trade-offs:** Qdrant manages vector persistence and supports future metadata filtering, but adds network dependency. The default pipeline's median embedding-plus-retrieval latency increased from 678.7 ms to 1047.6 ms in this run. Local embeddings alone do not keep the application local: chunk text is stored in Qdrant and retrieved context is sent to OpenAI for generation.

See [Week 7 KPI snapshot](../kpi/wk7-snapshot.md) for results and methodology.

## W8 — JSONL ingestion + PII discipline

**Decision:** I use `src/week8/ingest/pipeline.py` to ingest the clinical JSONL corpus into `capstone_chunks_v2`. Each case keeps its source ID. The current collection contains 10 cases and 75 chunks.

**Ingestion pipeline steps:**

1. Read JSONL cases, using `text` or `full_note` with `note` as a fallback.
2. Scrub each case with regex + Presidio before chunking.
3. Split into 800-character windows with 80-character overlap, keeping case boundaries.
4. Add metadata: 12 stored fields including text, source IDs, dates, language and PII flag count.
5. Embed with `text-embedding-3-small` (1536 dimensions).
6. Upsert to Qdrant collection `capstone_chunks_v2` using stable point IDs and cosine distance.

**PII detection:** Regex checks EMAIL, PHONE and EMP_ID. Presidio checks PERSON, email addresses, phone numbers, US SSNs, credit cards and IP addresses. Dates, ages, locations and organizations are kept as clinical context. I reviewed 10 chunks fetched with `scroll(limit=10)`; these were not randomly selected. See [the PII audit](../wk8-pii-audit.md).

**Known gaps:** All 28 flags were false positives. Figure references and medical terms such as Brucella, Wright, Moxifloxacin and ANA were wrongly redacted. I need to fix the scrubber, check for missed PII and re-ingest the original cases. Metadata filtering is available but is not used in retrieval yet.

**Data residency:**

- Embeddings: OpenAI API; processing region not verified.
- Vectors and chunk text: Qdrant Cloud; cluster region not verified.
- LLM inference: OpenAI API; processing region not verified.
- App layer: deployment location not recorded.
- Data flow: the ingest process sends scrubbed text to OpenAI for embeddings and stores text and vectors in Qdrant. At query time, retrieved context goes to OpenAI for generation. Cross-jurisdictional flows depend on the regions above.

**Rejected alternatives:**

- Multiple document parsers: the corpus is JSONL only, so PDF, HTML and DOCX support was removed.
- Structure-aware or semantic chunking: kept fixed character windows for this baseline; other strategies have not been evaluated.
- Presidio alone: kept regex for explicit identifier patterns alongside Presidio. The hybrid still needs tuning for medical text.

**Operational KPIs:** On the same 30 golden questions, source hit rate is 100% and evidence hit rate is 90%, unchanged from W7. Median embedding-plus-retrieval latency is 1038.4 ms, and mean query embedding cost is $0.000000374. Grounded response rate and generation cost were not measured. There were 28 PII flags and no confirmed PII removed. See [the W8 snapshot](../kpi/wk8-snapshot.md).
