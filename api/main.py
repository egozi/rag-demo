import logging
import gradio as gr
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
from fastapi import HTTPException
from pydantic import BaseModel

from api.agent import ConversationalRAGAgent
from api.history import ConversationHistory
from api.config import get_settings
from api.ui import build_demo

app = FastAPI(title="RAG Demo API", version="1.0.0")

_settings = get_settings()
_agent = ConversationalRAGAgent()
_history = _agent._history


class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    chunks: list[dict]


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_backend": _settings.llm_backend}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, chunks = _agent.ask(req.session_id, req.question, top_k=req.top_k)
    return ChatResponse(
        answer=answer,
        chunks=[
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "score": c.score,
                "text": c.text,
            }
            for c in chunks
        ],
    )


@app.post("/api/sessions/{session_id}/evaluate")
def evaluate_last(session_id: str):
    """Run RAGAS evaluation on the most recent answer for this session.

    HTTP example:
        curl -X POST http://localhost:8000/api/sessions/SESSION_ID/evaluate
    """
    scores = _agent.evaluate_last(session_id)
    if not scores:
        raise HTTPException(status_code=404, detail="No answer to evaluate for this session — ask a question first.")
    return {"session_id": session_id, "ragas_scores": scores}


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str):
    context = _history.get_context(session_id)
    return {"session_id": session_id, "context": context}


@app.delete("/api/sessions/{session_id}/clear")
def clear_session(session_id: str):
    _history.clear(session_id)
    return {"session_id": session_id, "cleared": True}


# Mount Gradio UI at root
demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")
