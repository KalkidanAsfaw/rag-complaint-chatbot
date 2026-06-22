"""Unit tests for src.rag_pipeline pure logic (no model loading)."""

from src.rag_pipeline import PROMPT_TEMPLATE, RetrievedChunk, format_context


def _chunk(text, product="Credit Card", issue="Fees"):
    return RetrievedChunk(
        text=text,
        metadata={"product_category": product, "issue": issue},
        distance=0.1,
    )


def test_format_context_numbers_and_tags_chunks():
    chunks = [_chunk("late fee charged"), _chunk("interest too high", issue="Interest")]
    ctx = format_context(chunks)
    assert "1. [Credit Card | Fees] late fee charged" in ctx
    assert "2. [Credit Card | Interest] interest too high" in ctx


def test_prompt_template_injects_context_and_question():
    prompt = PROMPT_TEMPLATE.format(context="some context", question="why?")
    assert "some context" in prompt
    assert "why?" in prompt
    # Grounding instruction is present
    assert "don't have enough information" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_format_context_empty():
    assert format_context([]) == ""
