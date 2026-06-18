# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) chatbot that turns CrediTrust Financial's raw customer
complaint narratives (CFPB dataset) into evidence-backed answers for Product, Support, and
Compliance teams — across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

10 Academy KAIM — Week 7 challenge.

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/    # CI (pytest on push/PR)
├── .vscode/               # Editor settings (gitignored, local only)
├── data/
│   ├── raw/               # Original CFPB dataset (gitignored)
│   └── processed/         # filtered_complaints.csv etc. (gitignored)
├── vector_store/          # Persisted Chroma/FAISS index (gitignored)
├── notebooks/              # EDA, chunking/embedding, RAG evaluation notebooks
├── src/                    # Reusable pipeline modules (preprocessing, chunking, rag_pipeline)
├── tests/                  # Unit tests
├── reports/                # Interim and final report drafts
├── app.py                  # Gradio chat interface
└── requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd rag-complaint-chatbot

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Data

Download from the links in the challenge doc and place under `data/raw/`:
- Full CFPB complaint dataset (Task 1)
- `complaint_embeddings.parquet` — pre-built embeddings/chunks/metadata (Tasks 3–4)

## Running

```bash
# Task 1/2 notebooks under notebooks/
jupyter notebook

# Task 4 UI
python app.py
```

See [reports/project_plan.md](reports/project_plan.md) for the full task breakdown and timeline.
