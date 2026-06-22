"""Task 3 evaluation: run representative questions through the RAG pipeline and
emit a Markdown table (Question | Answer | Top sources | Score | Comments) for the
report. Scores/comments are left blank for manual qualitative grading.
"""

from __future__ import annotations

from pathlib import Path

from src.data_processing import PROJECT_ROOT
from src.rag_pipeline import RAGPipeline

EVAL_QUESTIONS = [
    "Why are people unhappy with credit cards?",
    "What are common complaints about personal loans?",
    "What problems do customers report with money transfers?",
    "Are there issues with unauthorized or fraudulent charges?",
    "What complaints relate to savings account fees?",
    "Do customers report difficulty closing their accounts?",
    "What issues do people have with billing disputes?",
    "Are there complaints about poor customer service responses?",
]

OUTPUT_PATH = PROJECT_ROOT / "reports" / "rag_evaluation.md"


def _truncate(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def build_table(questions: list[str], top_k: int = 5) -> str:
    rag = RAGPipeline(top_k=top_k)
    rows = [
        "| # | Question | Generated Answer | Top Retrieved Sources (1-2) | Score (1-5) | Comments |",
        "|---|---|---|---|---|---|",
    ]
    for i, q in enumerate(questions, 1):
        result = rag.answer(q)
        srcs = []
        for c in result.sources[:2]:
            m = c.metadata
            srcs.append(f"[{m.get('product_category', '?')} / {m.get('issue', '?')}] {_truncate(c.text, 90)}")
        sources_md = "<br>".join(srcs) if srcs else "(none)"
        answer_md = _truncate(result.answer, 200).replace("|", "\\|")
        rows.append(
            f"| {i} | {q} | {answer_md} | {sources_md.replace('|', chr(92) + '|')} |  |  |"
        )
    return "\n".join(rows)


def main(out_path: str | Path = OUTPUT_PATH) -> Path:
    out_path = Path(out_path)
    table = build_table(EVAL_QUESTIONS)
    header = (
        "# RAG Evaluation\n\n"
        "Qualitative evaluation of the retrieval-augmented generation pipeline on "
        "representative questions. Fill in **Score (1-5)** and **Comments** after "
        "reviewing each answer against its retrieved sources.\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + table + "\n")
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Wrote evaluation table to {path}")
