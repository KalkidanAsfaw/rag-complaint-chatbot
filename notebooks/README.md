# Notebooks

- `01_eda_preprocessing.ipynb` — Task 1: EDA on the full CFPB dataset, filtering to the four target products, text cleaning.
- `02_chunking_embedding.ipynb` — Task 2: stratified sampling, chunking experiments, embedding generation, vector store indexing.
- `03_rag_evaluation.ipynb` — Task 3: retriever/generator smoke tests and the qualitative evaluation table.

Heavy logic (chunking, embedding, retrieval, generation) lives in `src/` and is imported here — notebooks stay thin and are for exploration/reporting, not implementation.
