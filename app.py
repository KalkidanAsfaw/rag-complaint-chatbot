"""Gradio chat interface for the CrediTrust complaint-analysis RAG agent.

Wired up to src.rag_pipeline once Task 3 is implemented.
"""
import gradio as gr


def answer_question(question: str, history):
    # TODO(Task 3/4): replace with src.rag_pipeline.run(question)
    return "RAG pipeline not yet implemented.", []


with gr.Blocks(title="CrediTrust Complaint Insights") as demo:
    gr.Markdown("# CrediTrust Complaint Insights\nAsk a question about customer complaints.")

    question_box = gr.Textbox(label="Your question", placeholder="Why are people unhappy with Credit Cards?")
    answer_box = gr.Textbox(label="Answer", interactive=False)
    sources_box = gr.JSON(label="Retrieved sources")

    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    ask_btn.click(answer_question, inputs=[question_box, gr.State([])], outputs=[answer_box, sources_box])
    clear_btn.click(lambda: ("", "", None), outputs=[question_box, answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
