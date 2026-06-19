"""Task 2: stratified sampling, chunking, embedding, and vector-store indexing.

Pipeline (run on the cleaned Task 1 output, not the full corpus):

    filtered_complaints.csv
        -> stratified sample (proportional across the 4 product categories)
        -> RecursiveCharacterTextSplitter (500 char / 50 overlap)
        -> all-MiniLM-L6-v2 embeddings (384-d)
        -> ChromaDB persistent index under vector_store/ (with per-chunk metadata)

Heavy dependencies (sentence-transformers, chromadb, langchain) are imported
lazily inside the functions that need them, so the sampling/chunking logic can be
imported and unit-tested without them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data_processing import (
    FILTERED_COMPLAINTS_PATH,
    PROJECT_ROOT,
    TARGET_PRODUCTS,
)

# --- Config ----------------------------------------------------------------

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "complaints"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_SAMPLE_SIZE = 12_000
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

TEXT_COL = "cleaned_narrative"
ID_COL = "Complaint ID"

# Map source columns -> per-chunk metadata field names (spec-aligned).
METADATA_COLS = {
    "Complaint ID": "complaint_id",
    "product_category": "product_category",
    "Product": "product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "Date received": "date_received",
}


# --- Loading & sampling ----------------------------------------------------


def load_filtered(path: str | Path = FILTERED_COMPLAINTS_PATH) -> pl.DataFrame:
    """Load the cleaned, filtered Task 1 dataset."""
    return pl.read_csv(path, infer_schema_length=0)


def stratified_sample(
    df: pl.DataFrame,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
    group_col: str = "product_category",
) -> pl.DataFrame:
    """Proportionally sample ``n`` rows, preserving each category's share.

    Allocates per-group counts proportional to each group's size (largest-
    remainder rounding so the totals add up), then samples without replacement
    within each group. If ``n`` exceeds the available rows the whole frame is
    returned.
    """
    total = df.height
    if n >= total:
        return df

    counts = (
        df.group_by(group_col)
        .agg(pl.len().alias("size"))
        .sort(group_col)
    )
    sizes = dict(zip(counts[group_col].to_list(), counts["size"].to_list()))

    # Largest-remainder allocation so per-group quotas sum exactly to n.
    raw = {g: n * s / total for g, s in sizes.items()}
    alloc = {g: int(v) for g, v in raw.items()}
    remainder = n - sum(alloc.values())
    for g, _ in sorted(raw.items(), key=lambda kv: kv[1] - alloc[kv[0]], reverse=True):
        if remainder <= 0:
            break
        alloc[g] += 1
        remainder -= 1

    parts = [
        df.filter(pl.col(group_col) == g).sample(
            n=min(k, sizes[g]), seed=seed, shuffle=True
        )
        for g, k in alloc.items()
        if k > 0
    ]
    return pl.concat(parts).sample(fraction=1.0, seed=seed, shuffle=True)


# --- Chunking --------------------------------------------------------------


def chunk_records(
    df: pl.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split each narrative into overlapping chunks with traceable metadata.

    Returns a list of ``{"id", "text", "metadata"}`` dicts. ``metadata`` carries
    the source complaint fields plus ``chunk_index`` / ``total_chunks`` so any
    retrieved chunk can be traced back to its original complaint.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        text = row.get(TEXT_COL) or ""
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        total = len(chunks)
        base_meta = {
            field: (row.get(src) or "") for src, field in METADATA_COLS.items()
        }
        cid = row.get(ID_COL) or "unknown"
        for i, chunk in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_index"] = i
            meta["total_chunks"] = total
            records.append(
                {"id": f"{cid}-{i}", "text": chunk, "metadata": meta}
            )
    return records


# --- Embedding & indexing --------------------------------------------------


def build_index(
    records: list[dict],
    persist_dir: str | Path = VECTOR_STORE_DIR,
    model_name: str = EMBEDDING_MODEL,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 512,
) -> int:
    """Embed chunk texts and persist them to a ChromaDB collection.

    Returns the number of chunks indexed.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    # Recreate the collection so reruns are idempotent.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    texts = [r["text"] for r in records]
    for start in range(0, len(records), batch_size):
        end = start + batch_size
        embeddings = model.encode(
            texts[start:end], show_progress_bar=False, convert_to_numpy=True
        ).tolist()
        collection.add(
            ids=[r["id"] for r in records[start:end]],
            documents=texts[start:end],
            embeddings=embeddings,
            metadatas=[r["metadata"] for r in records[start:end]],
        )
    return len(records)


def run_pipeline(
    filtered_path: str | Path = FILTERED_COMPLAINTS_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_dir: str | Path = VECTOR_STORE_DIR,
    seed: int = 42,
) -> dict:
    """End-to-end Task 2: load -> sample -> chunk -> embed -> persist."""
    df = load_filtered(filtered_path)
    sample = stratified_sample(df, n=sample_size, seed=seed)
    records = chunk_records(sample, chunk_size, chunk_overlap)
    n_indexed = build_index(records, persist_dir)
    return {
        "sampled_complaints": sample.height,
        "chunks": len(records),
        "indexed": n_indexed,
        "persist_dir": str(persist_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 2 vector-store build.")
    parser.add_argument("--filtered", default=str(FILTERED_COMPLAINTS_PATH))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--persist-dir", default=str(VECTOR_STORE_DIR))
    args = parser.parse_args()

    stats = run_pipeline(
        args.filtered,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
    )
    print(
        f"Sampled {stats['sampled_complaints']:,} complaints -> "
        f"{stats['chunks']:,} chunks indexed in {stats['persist_dir']}"
    )
