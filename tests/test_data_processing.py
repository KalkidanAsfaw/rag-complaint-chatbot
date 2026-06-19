"""Unit tests for src.data_processing (Task 1, polars-based)."""

import polars as pl

from src.data_processing import (
    NARRATIVE_COL,
    PRODUCT_COL,
    add_narrative_length,
    clean_text,
    filter_complaints,
    narrative_presence,
)


def test_clean_text_lowercases_and_strips_redactions():
    raw = "I am writing to file a complaint about XXXX charges!!!"
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "xxxx" not in cleaned
    assert "writing to file a complaint" not in cleaned
    assert "charges" in cleaned


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def _sample_df():
    return pl.DataFrame(
        {
            PRODUCT_COL: [
                "Credit card or prepaid card",
                "Checking or savings account",
                "Mortgage",  # not a target product
                "Money transfer, virtual currency, or money service",
            ],
            NARRATIVE_COL: ["valid narrative", "", "ignored", "another one"],
        }
    )


def test_filter_keeps_only_target_products_with_narratives():
    df = filter_complaints(_sample_df())
    # Mortgage dropped (not target); savings row dropped (empty narrative)
    assert set(df["product_category"].to_list()) == {"Credit Card", "Money Transfer"}
    assert len(df) == 2


def test_narrative_presence_counts():
    counts = narrative_presence(_sample_df())
    assert counts["with_narrative"] == 3
    assert counts["without_narrative"] == 1


def test_add_narrative_length():
    df = add_narrative_length(_sample_df())
    assert "narrative_word_count" in df.columns
    assert df["narrative_word_count"].to_list() == [2, 0, 1, 2]
