from uuid import uuid4

import gradio as gr

from api.agent import ConversationalRAGAgent

_agent = ConversationalRAGAgent()


def _ask(question: str, history: list[list], session_id: str):
    if not question.strip():
        return history, [], session_id

    answer, chunks = _agent.ask(session_id, question)

    history = history + [[question, answer]]
    sources = [
        [c.chunk_id, c.source, round(c.score, 3), c.text[:200]]
        for c in chunks
    ]
    return history, sources, session_id


def _evaluate(session_id: str):
    scores = _agent.evaluate_last(session_id)
    return scores


def _clear(session_id: str):
    _agent._history.clear(session_id)
    return [], {}, [], str(uuid4())


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Demo") as demo:
        gr.Markdown("# RAG Demo\nAsk questions about your indexed documents.")

        # Initialized empty; demo.load() sets a unique UUID per user on page load.
        # Using load() instead of gr.State(lambda: ...) ensures the value reaches
        # the frontend before any button (e.g. Clear) can be clicked.
        session_id = gr.State("")

        chatbot = gr.Chatbot(label="Conversation", height=450)

        with gr.Row():
            query = gr.Textbox(
                label="Your question",
                placeholder="Ask something about the documents...",
                scale=5,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)

        with gr.Row():
            evaluate_btn = gr.Button("Evaluate last answer (RAGAS)", variant="secondary")
            clear = gr.Button("Clear History", variant="secondary")

        with gr.Accordion("RAGAS Evaluation Scores", open=True):
            gr.Markdown(
                "Click **Evaluate last answer** to score the most recent response. "
                "Powered by GPT-4o as judge (~10-15s).\n\n"
                "- **Faithfulness** — are all claims in the answer supported by the retrieved chunks? "
                "Catches hallucinations.\n"
                "- **Answer relevancy** — does the answer actually address the question asked? "
                "Penalises vague or off-topic replies.\n"
                "- **Context precision** — were the retrieved chunks genuinely useful for producing "
                "the answer? Low score means the retriever pulled in irrelevant passages."
            )
            ragas_display = gr.JSON(label="Scores (0–1, higher is better)")

        with gr.Accordion("Retrieved Sources", open=False):
            sources_display = gr.Dataframe(
                headers=["chunk_id", "source", "score", "text (preview)"],
                label="Top-k chunks used to answer",
                wrap=True,
            )

        submit.click(
            fn=_ask,
            inputs=[query, chatbot, session_id],
            outputs=[chatbot, sources_display, session_id],
        )
        query.submit(
            fn=_ask,
            inputs=[query, chatbot, session_id],
            outputs=[chatbot, sources_display, session_id],
        )
        evaluate_btn.click(
            fn=_evaluate,
            inputs=[session_id],
            outputs=[ragas_display],
        )
        clear.click(
            fn=_clear,
            inputs=[session_id],
            outputs=[chatbot, ragas_display, sources_display, session_id],
        )
        demo.load(fn=lambda: str(uuid4()), inputs=None, outputs=[session_id])

    return demo
