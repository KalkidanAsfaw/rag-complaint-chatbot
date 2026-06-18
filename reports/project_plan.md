# Project Plan — Intelligent Complaint Analysis (RAG Chatbot)

## 1. Objective

Build a RAG agent so CrediTrust's Product/Support/Compliance teams can ask plain-English
questions about customer complaints and get evidence-backed answers in seconds, across
Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

Success = (1) trend identification time days → minutes, (2) non-technical teams self-serve
answers, (3) shift from reactive to proactive issue spotting.

## 2. Architecture

```
CFPB raw data ──Task1──▶ filtered_complaints.csv
                              │
                    (own pipeline, 10-15k sample)
                              ▼
                Task2: chunk → embed (all-MiniLM-L6-v2) → ChromaDB/FAISS index
                              │
        (for dev/learning; production path below uses the pre-built store)
                              ▼
        complaint_embeddings.parquet (pre-built, ~1.37M chunks, full 464K complaints)
                              │
                              ▼
                Task3: retriever (top-k=5) → prompt template → LLM generator
                              │
                              ▼
                Task4: Gradio app — question box, answer, sources, clear button
```

Two-track rationale: Task 2 is a learning exercise on a stratified sample (cheap to iterate
on chunk size/overlap). Tasks 3-4 build the user-facing product against the pre-built
full-scale vector store, so retrieval quality reflects the entire complaint corpus, not just
the sample.

## 3. Tech choices (initial; revisit if evaluation says otherwise)

- **Chunking:** LangChain `RecursiveCharacterTextSplitter`, start at the spec'd size/overlap
  (500/50, matching the pre-built store) so chunks from our own pipeline and the pre-built
  store are comparable.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — matches the pre-built store
  (required for query/document embedding compatibility), small (~80MB), fast on CPU.
- **Vector store:** ChromaDB primary (pre-built store ships in this format); keep FAISS as a
  fallback path since raw embeddings are also provided.
- **LLM generator:** Hugging Face `pipeline` or LangChain integration with an open model
  (e.g. Mistral-7B-Instruct via HF Inference API, or a smaller local model if compute-limited)
  — final choice depends on latency/quality tradeoff observed in Task 3 evaluation.
- **UI:** Gradio (faster to ship a chat-style interface with streaming than Streamlit for this
  scope).

## 4. Task breakdown

### Task 1 — EDA & Preprocessing
- Load full CFPB dataset; profile product distribution, narrative length, narrative
  presence/absence.
- Filter to the 4 target products + drop empty-narrative rows.
- Clean text: lowercase, strip special chars/boilerplate, normalize whitespace.
- Output: `data/processed/filtered_complaints.csv`, EDA findings (2-3 paragraphs) for the report.

### Task 2 — Chunking, Embedding, Indexing (sample pipeline)
- Stratified sample (10K-15K rows, proportional across the 4 products) from the cleaned data.
- Chunk with `RecursiveCharacterTextSplitter`; experiment with chunk_size/overlap, document
  the chosen values and why.
- Embed each chunk with all-MiniLM-L6-v2.
- Persist to `vector_store/` with metadata (complaint_id, product_category) for traceability.

### Task 3 — RAG Core + Evaluation
- Load the pre-built vector store (full dataset).
- Retriever: embed query, top-k=5 similarity search.
- Prompt template grounding the LLM in retrieved context only (template per challenge doc).
- Generator: combine prompt + context + question, call LLM, return answer.
- Evaluate on 5-10 representative questions; build the Question / Answer / Sources / Score
  (1-5) / Comments table.

### Task 4 — Interactive UI
- Gradio app: question input, Ask button, answer display, **sources shown below the answer**,
  Clear button. Streaming if time allows.

## 5. Timeline (today: Wed 2026-06-17)

| Date | Focus |
|---|---|
| Wed 2026-06-17 | Repo scaffolded (done). Start Task 1: load data, EDA. |
| Thu 2026-06-18 | Finish Task 1 cleaning/filtering. Start Task 2: stratified sample + chunking experiments. |
| Fri 2026-06-19 | Finish Task 2: embeddings + persisted vector store. Draft interim report (Task 1+2 sections). |
| Sat 2026-06-20 | Buffer / catch-up. Polish interim report, push to main. |
| **Sun 2026-06-21, 8PM UTC** | **Interim submission**: main branch with Task 1+2 merged, interim report. |
| Mon 2026-06-22 | Task 3: retriever + generator against pre-built store, run evaluation set, fill table. |
| Tue 2026-06-23 (day) | Task 4: Gradio UI + sources display + screenshots. Write final report (Medium-style). |
| **Tue 2026-06-23, 8PM UTC** | **Final submission**: main branch + final report. |

## 6. Risks & mitigations

- **LLM access/rate limits** (HF Inference API) — have a smaller local fallback model ready.
- **Pre-built store schema mismatch** — verify metadata field names against the spec before
  building the retriever, fail fast if they differ.
- **Chunking sample not representative** — stratify explicitly by `product_category`, log the
  per-category counts before/after sampling.

## 7. Submission checklist

- [ ] `data/filtered_complaints.csv` produced and EDA summary written (Task 1)
- [ ] Sampling strategy + chunking/embedding justification written (Task 2)
- [ ] Vector store persisted under `vector_store/`
- [ ] RAG module(s) in `src/`, evaluation table in report (Task 3)
- [ ] `app.py` working with sources display + clear button (Task 4)
- [ ] Interim report (Task 1+2) — due Sun 2026-06-21 8PM UTC
- [ ] Final Medium-style report (Intro, Technical Choices, Evaluation, UI Showcase, Conclusion)
      — due Tue 2026-06-23 8PM UTC
