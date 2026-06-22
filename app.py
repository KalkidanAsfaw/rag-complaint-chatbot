"""Task 4: Gradio chat interface for the CrediTrust complaint-analysis RAG agent.

A non-technical user types a plain-English question, gets an evidence-backed
answer, and sees the source complaint excerpts the answer was grounded in
(crucial for trust/verification). Includes a Clear button to reset.

Run: python app.py
"""

from __future__ import annotations

import gradio as gr

from src.build_vector_store import TARGET_PRODUCTS
from src.rag_pipeline import RAGPipeline

# Load the pipeline once at startup (model + vector store) and reuse it.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _format_sources(sources) -> str:
    if not sources:
        return "_No sources retrieved._"
    blocks = []
    for i, c in enumerate(sources, 1):
        m = c.metadata
        header = (
            f"**Source {i}** — {m.get('product_category', '?')} · "
            f"{m.get('issue', '?')} · {m.get('company', '?')} ({m.get('state', '?')})"
        )
        blocks.append(f"{header}\n\n> {c.text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, product: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "_No sources retrieved._"
    category = None if product == "All products" else product
    result = get_pipeline().answer(question, product_category=category)
    return result.answer, _format_sources(result.sources)


def clear():
    return "", "All products", "", "_No sources retrieved._"


with gr.Blocks(title="CrediTrust Complaint Insights") as demo:
    gr.Markdown(
        "# CrediTrust Complaint Insights\n"
        "Ask a plain-English question about customer complaints. Answers are "
        "grounded in real complaint narratives, shown below each response."
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="Why are people unhappy with credit cards?",
            scale=4,
        )
        product_dropdown = gr.Dropdown(
            choices=["All products", *TARGET_PRODUCTS],
            value="All products",
            label="Filter by product",
            scale=1,
        )

    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    answer_box = gr.Markdown(label="Answer")
    gr.Markdown("### Sources")
    sources_box = gr.Markdown("_No sources retrieved._")

    ask_btn.click(
        answer_question,
        inputs=[question_box, product_dropdown],
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        answer_question,
        inputs=[question_box, product_dropdown],
        outputs=[answer_box, sources_box],
    )
    clear_btn.click(
        clear,
        outputs=[question_box, product_dropdown, answer_box, sources_box],
    )


if __name__ == "__main__":
    demo.launch()
