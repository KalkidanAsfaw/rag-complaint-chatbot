"""Task 1: EDA and preprocessing for the CFPB complaint dataset.

The raw CFPB export is a ~5.6 GB CSV with ~9.6M rows. It contains at least one
malformed quote that makes pandas' C parser fail with
``ParserError: EOF inside string``. We use **polars** instead: its CSV reader
handles the malformed quoting natively, scans the file lazily (predicate/column
pushdown), and is fast and memory-efficient on a file this size.

This module provides reusable functions to scan, explore, filter, and clean the
raw data so it is ready for the chunking/embedding pipeline.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FILTERED_COMPLAINTS_PATH = PROCESSED_DATA_DIR / "filtered_complaints.csv"
# Parquet cache of the raw CSV. The raw file is ~5.6 GB and CSV is not seekable,
# so any polars query has to read the whole thing into memory (~5.7 GB) — too
# much for a typical laptop. Converting once to columnar Parquet lets later
# queries push down column selection and run in a fraction of the memory.
RAW_PARQUET_PATH = PROCESSED_DATA_DIR / "complaints.parquet"

# --- Column names (CFPB export) -------------------------------------------

PRODUCT_COL = "Product"
NARRATIVE_COL = "Consumer complaint narrative"

# Columns we keep for the filtered dataset, so retrieved chunks can later be
# traced back to their source complaint (Task 2/3 metadata).
KEEP_COLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    NARRATIVE_COL,
    "Company",
    "State",
    "Complaint ID",
]

# --- Product mapping -------------------------------------------------------
# The raw CFPB "Product" field uses verbose labels. Map the ones we care about
# to the four canonical categories used throughout this project.
PRODUCT_MAP = {
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Checking or savings account": "Savings Account",
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
}

TARGET_PRODUCTS = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]


# --- Loading ---------------------------------------------------------------


def scan_complaints(path: str | Path) -> pl.LazyFrame:
    """Lazily scan the complaints source (Parquet or CSV).

    Returns a :class:`polars.LazyFrame` so callers can push down column
    selection and filters before materializing. For CSV, ``infer_schema_length=0``
    reads every column as a string, which avoids dtype-inference surprises on the
    huge, messy file and is robust to the malformed quoting that breaks pandas.
    Prefer Parquet (see :func:`convert_to_parquet`) for low-memory queries.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, infer_schema_length=0)


def convert_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path = RAW_PARQUET_PATH,
    overwrite: bool = False,
) -> Path:
    """Stream the raw CSV to a Parquet cache (idempotent, low memory).

    Uses ``sink_parquet`` so the conversion streams batch-by-batch instead of
    materializing the whole 5.6 GB file. Returns the Parquet path; skips work if
    it already exists (unless ``overwrite``).
    """
    parquet_path = Path(parquet_path)
    if parquet_path.exists() and not overwrite:
        return parquet_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_csv(csv_path, infer_schema_length=0).sink_parquet(parquet_path)
    return parquet_path


def load_complaints(path: str | Path) -> pl.DataFrame:
    """Eagerly load the raw CFPB CSV into a DataFrame (robust to bad quotes)."""
    return scan_complaints(path).collect()


# --- EDA -------------------------------------------------------------------


def product_distribution(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Count complaints per raw product (descending)."""
    lf = df.lazy()
    return (
        lf.group_by(PRODUCT_COL)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")
    )


def add_narrative_length(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Add a ``narrative_word_count`` column (0 for missing narratives)."""
    return (
        df.lazy()
        .with_columns(
            pl.col(NARRATIVE_COL)
            .fill_null("")
            .str.count_matches(r"\S+")
            .alias("narrative_word_count")
        )
        .collect(engine="streaming")
    )


def narrative_presence(df: pl.DataFrame | pl.LazyFrame) -> dict[str, int]:
    """Return counts of complaints with vs. without a narrative."""
    has_expr = pl.col(NARRATIVE_COL).is_not_null() & (
        pl.col(NARRATIVE_COL).str.strip_chars() != ""
    )
    counts = (
        df.lazy()
        .select(
            has_expr.cast(pl.Int64).sum().alias("with"),
            pl.len().alias("total"),
        )
        .collect(engine="streaming")
    )
    with_narrative = int(counts["with"][0])
    total = int(counts["total"][0])
    return {"with_narrative": with_narrative, "without_narrative": total - with_narrative}


# --- Filtering -------------------------------------------------------------


def filter_complaints(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Keep the four target products and drop empty narratives.

    Adds a canonical ``product_category`` column.
    """
    narrative = pl.col(NARRATIVE_COL)
    return (
        df.lazy()
        .with_columns(
            pl.col(PRODUCT_COL)
            .replace_strict(PRODUCT_MAP, default=None)
            .alias("product_category")
        )
        .filter(pl.col("product_category").is_in(TARGET_PRODUCTS))
        .filter(narrative.is_not_null() & (narrative.str.strip_chars() != ""))
        .collect(engine="streaming")
    )


# --- Cleaning --------------------------------------------------------------

# Common boilerplate openers seen in CFPB narratives.
_BOILERPLATE_PATTERNS = [
    r"i am writing to (?:file|submit) a complaint(?: about| regarding)?",
    r"i am writing to you to (?:file|submit|express)",
    r"to whom it may concern",
    r"this is a complaint (?:about|regarding)",
]
_BOILERPLATE_RE = "|".join(_BOILERPLATE_PATTERNS)


def _clean_expr(col: str) -> pl.Expr:
    """Polars expression that normalizes a narrative column.

    Lowercase, strip CFPB redaction placeholders (``XXXX``), remove boilerplate
    openers and non-basic characters, then collapse whitespace. Uses Rust regex
    (no look-around needed).
    """
    return (
        pl.col(col)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"x{2,}", " ")
        .str.replace_all(_BOILERPLATE_RE, " ")
        .str.replace_all(r"[^a-z0-9\s.,!?']", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def clean_text(text) -> str:
    """Normalize a single narrative string (scalar convenience wrapper)."""
    if not isinstance(text, str):
        return ""
    return (
        pl.DataFrame({"t": [text]})
        .select(_clean_expr("t").alias("t"))["t"][0]
    )


def clean_narratives(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Add a ``cleaned_narrative`` column and drop rows empty after cleaning."""
    return (
        df.lazy()
        .with_columns(_clean_expr(NARRATIVE_COL).alias("cleaned_narrative"))
        .filter(pl.col("cleaned_narrative").str.strip_chars() != "")
        .collect()
    )


# --- Persistence -----------------------------------------------------------


def save_filtered(
    df: pl.DataFrame, path: str | Path = FILTERED_COMPLAINTS_PATH
) -> Path:
    """Persist the cleaned, filtered dataset to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return path


def run_pipeline(
    raw_path: str | Path, out_path: str | Path = FILTERED_COMPLAINTS_PATH
) -> pl.DataFrame:
    """End-to-end Task 1 pipeline: cache as Parquet -> filter -> clean -> save."""
    raw_path = Path(raw_path)
    source = convert_to_parquet(raw_path) if raw_path.suffix != ".parquet" else raw_path
    lf = scan_complaints(source).select(KEEP_COLS)
    filtered = filter_complaints(lf)
    cleaned = clean_narratives(filtered)
    save_filtered(cleaned, out_path)
    return cleaned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 1 preprocessing pipeline.")
    parser.add_argument("raw_path", help="Path to the raw CFPB complaints CSV.")
    parser.add_argument(
        "--out", default=str(FILTERED_COMPLAINTS_PATH), help="Output CSV path."
    )
    args = parser.parse_args()

    result = run_pipeline(args.raw_path, args.out)
    print(f"Saved {len(result):,} cleaned complaints to {args.out}")
