# Interim Report — Intelligent Complaint Analysis (RAG Chatbot)

**10 Academy KAIM Week 7 · Tasks 1–2 · CrediTrust Financial**

This interim report covers data understanding/preprocessing (Task 1) and the
chunking → embedding → vector-store pipeline (Task 2). Tasks 3–4 (RAG core,
evaluation, and UI) follow in the final submission.

---

## Task 1 — Exploratory Data Analysis & Preprocessing

The raw CFPB export is a ~5.6 GB CSV with **9,609,797 complaints** across 18
columns. It contains a malformed quote that crashes pandas' C parser
(`ParserError: EOF inside string`), so the pipeline is built on **polars**, which
reads the file reliably. Because CSV is not seekable, every query would otherwise
load the whole file into RAM; we convert it once to a columnar **Parquet** cache
(`data/processed/complaints.parquet`, 780 MB) via a streaming sink, after which
EDA queries use column pushdown and streaming collects and stay well within an
8 GB machine.

**Product distribution.** The corpus is heavily skewed toward credit reporting:
the two credit-reporting product labels together account for ~73% of all records,
followed by debt collection (799K) and mortgage (422K). The four products this
project targets are a comparatively small slice. After mapping the verbose raw
labels to four canonical categories and dropping rows without a narrative, the
usable dataset is **463,929 complaints**.

**Narrative presence.** Only **31.0%** of complaints (2,980,756) include a
free-text consumer narrative; the other 69% are empty. Since the RAG system can
only reason over text it can read, narratives are the scarce, high-value signal
and empty-narrative rows are dropped — the usable corpus is under a third of the
raw row count.

**Narrative length.** Among complaints with a narrative, length is right-skewed:
mean ≈ 176 words, median 114, IQR 59–209, p95 519, max 6,469. About 4.2% are very
short (<20 words) and 14.2% are long (>300 words). Embedding a long narrative as a
single vector would blur many distinct issues together — this directly motivates
the chunking strategy in Task 2.

**Cleaning.** Each narrative is lowercased; CFPB redaction placeholders (`XXXX`)
are stripped (this also removes redacted names/amounts, a deliberate choice since
those tokens add noise without semantic value); boilerplate openers (e.g. "I am
writing to file a complaint…") and special characters are removed and whitespace
collapsed. 4 records became empty after cleaning and were dropped. Output:
`data/processed/filtered_complaints.csv`.

**Deliverables:** [src/data_processing.py](../src/data_processing.py),
[notebooks/01_eda_polars.ipynb](../notebooks/01_eda_polars.ipynb),
`data/processed/filtered_complaints.csv`.

---

## Task 2 — Chunking, Embedding & Vector-Store Indexing

### Sampling strategy

Embedding the full 463,929-complaint corpus is expensive, so Task 2 builds the
pipeline on a **stratified sample of 12,000 complaints** drawn proportionally
across the four product categories. Proportional (rather than equal-size)
allocation preserves the real-world class balance so retrieval behaviour on the
sample mirrors the full dataset. We use largest-remainder rounding so the
per-category quotas sum exactly to 12,000, and sample without replacement within
each category using a fixed seed (42) for reproducibility.

| Product category | Full dataset | Share | Sample (n=12,000) |
|---|---:|---:|---:|
| Credit Card | 189,333 | 40.8% | 4,897 |
| Savings Account | 140,317 | 30.2% | 3,629 |
| Money Transfer | 98,684 | 21.3% | 2,553 |
| Personal Loan | 35,595 | 7.7% | 921 |
| **Total** | **463,929** | 100% | **12,000** |

### Chunking approach

Narratives vary from a handful of words to 6,000+, so each cleaned narrative is
split with LangChain's **`RecursiveCharacterTextSplitter`** using
**chunk_size = 500 characters** and **chunk_overlap = 50**. Rationale:

- **500 characters** (~80–90 words) keeps each chunk to roughly a single
  complaint, which produces focused embeddings and avoids diluting the signal of
  short complaints, while staying well inside MiniLM's 256-token window.
- **50-character overlap** (10%) preserves context across chunk boundaries so a
  sentence split mid-thought is still retrievable from either side.
- These values **match the pre-built full-scale store** (500/50) provided for
  Tasks 3–4, so chunks from our own pipeline are directly comparable.
- The recursive separators (`\n\n`, `\n`, `. `, ` `, ``) prefer to break on
  paragraph/sentence boundaries before resorting to hard character cuts.

The 12,000 sampled complaints produced **35,347 chunks** (≈2.95 chunks per
complaint; mean chunk length 368 chars, median 406).

### Embedding model

We use **`sentence-transformers/all-MiniLM-L6-v2`** (384-dimensional, ~80 MB).
Reasons: (1) it is the model behind the provided pre-built store, so query and
document embeddings must come from the same model to be compatible in Tasks 3–4;
(2) it offers a strong quality/speed trade-off and runs comfortably on CPU, which
matters given the no-GPU, 8 GB-RAM constraint; (3) its small footprint keeps the
index size and query latency low for an interactive chatbot.

### Vector store

Embeddings are persisted to a **ChromaDB** collection (`complaints`) under
`vector_store/`, using cosine similarity. Each chunk is stored with metadata for
traceability back to its source complaint: `complaint_id`, `product_category`,
`product`, `issue`, `sub_issue`, `company`, `state`, `date_received`,
`chunk_index`, and `total_chunks`. A sanity query ("Why are people unhappy with
credit cards?") returns relevant, correctly-tagged Credit Card chunks, confirming
the retriever works end-to-end.

**Deliverables:** [src/build_vector_store.py](../src/build_vector_store.py)
(`python -m src.build_vector_store --sample-size 12000`), persisted index in
`vector_store/` (ChromaDB), unit tests in
[tests/test_build_vector_store.py](../tests/test_build_vector_store.py).

---

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Task 1: filter + clean the raw CSV (writes data/processed/filtered_complaints.csv)
python -m src.data_processing data/raw/complaints.csv

# Task 2: stratified sample -> chunk -> embed -> persist to vector_store/
python -m src.build_vector_store --sample-size 12000

pytest -q   # 11 tests
```
