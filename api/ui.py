from uuid import uuid4

import gradio as gr

from api.agent import ConversationalRAGAgent

_agent = ConversationalRAGAgent()


def _ask(question: str, history: list[dict], session_id: str):
    if not question.strip():
        return history, {}, [], session_id

    answer, chunks, scores = _agent.ask(session_id, question)

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    sources = [
        [c.chunk_id, c.source, round(c.score, 3), c.text[:200]]
        for c in chunks
    ]
    return history, scores, sources, session_id


def _clear(session_id: str):
    _agent._history.clear(session_id)
    return [], {}, [], str(uuid4())


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Demo") as demo:
        gr.Markdown("# RAG Demo\nAsk questions about your indexed documents.")

        session_id = gr.State(lambda: str(uuid4()))

        chatbot = gr.Chatbot(label="Conversation", height=450)

        with gr.Row():
            query = gr.Textbox(
                label="Your question",
                placeholder="Ask something about the documents...",
                scale=5,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)

        clear = gr.Button("Clear History", variant="secondary")

        with gr.Accordion("RAGAS Evaluation Scores", open=False):
            gr.Markdown(
                "Real-time quality scores computed after each response. "
                "Powered by GPT-4o as judge."
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
            outputs=[chatbot, ragas_display, sources_display, session_id],
        )
        query.submit(
            fn=_ask,
            inputs=[query, chatbot, session_id],
            outputs=[chatbot, ragas_display, sources_display, session_id],
        )
        clear.click(
            fn=_clear,
            inputs=[session_id],
            outputs=[chatbot, ragas_display, sources_display, session_id],
        )

    return demo
