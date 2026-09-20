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
* **Initial data:** The platform will use 100 cases automatically selected from the `AGBonnet/augmented-clinical-notes` dataset using fixed validation rules.

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
* **Data preparation:** A Python pipeline will load records from the `AGBonnet/augmented-clinical-notes` dataset and parse the JSON stored in `summary`. Cases with invalid JSON or missing complaints, symptoms, or diagnoses will be skipped automatically. The pipeline will use a fixed random seed to select 100 valid cases so that the same collection can be reproduced.
* **Vector storage:** ChromaDB will store the case information as embeddings created through the OpenAI embeddings API. Every chunk will include the dataset `idx` as its `case_id`, along with metadata describing its source and whether it can be used during the patient interview or only during evaluation.
* **Patient information:** The patient collection will contain the presenting complaint, patient information, medical history, symptoms, and suitable patient responses from `conversation`. Chunks containing the recorded diagnosis will be excluded automatically.
* **Evaluation information:** A separate collection will contain chunks from `full_note`, examinations, diagnostic tests, recorded conditions, treatments, and outcomes. This information will remain unavailable to the synthetic patient.
* **Case selection and retrieval:** At the start of a simulation, the platform will select one of the 100 cases and keep its `case_id` fixed for the full session. Every retrieval request will filter by that ID, preventing information from different patients from being mixed. Interview questions will search only the patient collection. The submitted diagnosis and reasoning will search only the evaluation collection.
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
