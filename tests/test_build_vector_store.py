"""Unit tests for src.build_vector_store (Task 2)."""

import polars as pl
import pytest

from src.build_vector_store import chunk_records, stratified_sample


def _imbalanced_df(n_credit=800, n_savings=150, n_loan=50):
    rows = (
        [("Credit Card", f"c{i}") for i in range(n_credit)]
        + [("Savings Account", f"s{i}") for i in range(n_savings)]
        + [("Personal Loan", f"l{i}") for i in range(n_loan)]
    )
    return pl.DataFrame(
        {
            "product_category": [r[0] for r in rows],
            "Complaint ID": [r[1] for r in rows],
            "cleaned_narrative": ["some complaint text here" for _ in rows],
        }
    )


def test_stratified_sample_preserves_proportions_and_size():
    df = _imbalanced_df()
    sample = stratified_sample(df, n=200, seed=0)
    assert sample.height == 200
    counts = dict(
        sample.group_by("product_category")
        .agg(pl.len().alias("k"))
        .iter_rows()
    )
    # Proportional: 800/1000*200=160, 150/1000*200=30, 50/1000*200=10
    assert counts["Credit Card"] == 160
    assert counts["Savings Account"] == 30
    assert counts["Personal Loan"] == 10


def test_stratified_sample_returns_all_when_n_exceeds_rows():
    df = _imbalanced_df(10, 5, 5)
    sample = stratified_sample(df, n=10_000)
    assert sample.height == df.height


def test_stratified_sample_is_deterministic():
    df = _imbalanced_df()
    a = stratified_sample(df, n=100, seed=7)["Complaint ID"].to_list()
    b = stratified_sample(df, n=100, seed=7)["Complaint ID"].to_list()
    assert a == b


def test_chunk_records_metadata_and_indices():
    pytest.importorskip("langchain_text_splitters")
    df = pl.DataFrame(
        {
            "product_category": ["Credit Card"],
            "Complaint ID": ["123"],
            "Product": ["Credit card"],
            "Issue": ["Billing"],
            "Sub-issue": ["Late fee"],
            "Company": ["Acme"],
            "State": ["CA"],
            "Date received": ["2025-01-01"],
            "cleaned_narrative": [" ".join(["word"] * 400)],  # long -> multi-chunk
        }
    )
    records = chunk_records(df, chunk_size=200, chunk_overlap=20)
    assert len(records) > 1
    total = records[0]["metadata"]["total_chunks"]
    assert all(r["metadata"]["total_chunks"] == total for r in records)
    assert [r["metadata"]["chunk_index"] for r in records] == list(range(total))
    assert records[0]["id"] == "123-0"
    assert records[0]["metadata"]["complaint_id"] == "123"
    assert records[0]["metadata"]["product_category"] == "Credit Card"


def test_chunk_records_skips_empty_text():
    pytest.importorskip("langchain_text_splitters")
    df = pl.DataFrame(
        {
            "product_category": ["Credit Card"],
            "Complaint ID": ["1"],
            "Product": ["Credit card"],
            "Issue": ["x"],
            "Sub-issue": ["y"],
            "Company": ["z"],
            "State": ["CA"],
            "Date received": ["2025-01-01"],
            "cleaned_narrative": ["   "],
        }
    )
    assert chunk_records(df) == []
