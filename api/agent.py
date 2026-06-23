from retriever.retrieve import RetrievalResult

from api.config import get_settings
from api.embed import get_embedder
from api.history import ConversationHistory
from api.llm import LLMClient
from api.qdrant_store import QdrantStore
from api.ragas_eval import evaluate_response
from api.tracing import get_tracer

ANSWER_SYSTEM = """\
You are a helpful research assistant. Answer the user's question using ONLY \
the provided context. If the context does not contain the answer, say so plainly. \
Cite the chunk_id of any fact you use, in square brackets like [chunk_id].\
"""

REPHRASE_SYSTEM = """\
Given the conversation history below and a follow-up question, rephrase the \
follow-up question as a standalone question that can be understood without \
the history. Output only the rephrased question, nothing else.\
"""


class ConversationalRAGAgent:
    def __init__(
        self,
        store: QdrantStore | None = None,
        history: ConversationHistory | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        settings = get_settings()
        self._store = store or QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            path=settings.qdrant_path,
        )
        self._history = history or ConversationHistory(db_path=settings.history_db)
        self._llm = llm or LLMClient()
        self._embedder = embedder or get_embedder()
        # last turn per session — used by evaluate_last()
        self._last: dict[str, tuple[str, str, list[str]]] = {}

    def ask(
        self, session_id: str, question: str, top_k: int = 5
    ) -> tuple[str, list[RetrievalResult]]:
        tracer = get_tracer()
        trace = tracer.start_trace(session_id=session_id, input=question)

        # Step 1: Get conversation context
        context = self._history.get_context(session_id)

        # Step 2: Rephrase question if there is history
        search_query = question
        if context:
            rephrase_messages = [
                {"role": "system", "content": REPHRASE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Conversation history:\n{context}\n\nFollow-up question: {question}",
                },
            ]
            search_query = self._llm.chat(rephrase_messages, span_name="rephrase")

        # Step 3: Retrieve from Qdrant
        [query_vector] = self._embedder([search_query])
        chunks = self._store.search(query_vector=query_vector, top_k=top_k)

        # Step 4: Build prompt and answer
        formatted_chunks = "\n\n".join(
            f"[{c.chunk_id}] (source={c.source})\n{c.text}" for c in chunks
        )
        history_section = f"\n\nConversation so far:\n{context}" if context else ""
        user_content = (
            f"{history_section}\n\nContext:\n{formatted_chunks}\n\nQuestion: {question}"
        )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        answer_text = self._llm.chat(messages, span_name="answer")

        # Step 5: Persist turn and maybe summarize
        self._history.add_turn(session_id, question, answer_text)
        self._history.maybe_summarize(session_id, self._llm)

        # Store last turn so evaluate_last() can score it on demand
        self._last[session_id] = (question, answer_text, [c.text for c in chunks])

        tracer.flush()
        return answer_text, chunks

    def evaluate_last(self, session_id: str) -> dict[str, float]:
        """Run RAGAS evaluation on the most recent answer for this session."""
        if session_id not in self._last:
            return {}
        question, answer, contexts = self._last[session_id]
        tracer = get_tracer()
        trace = tracer.start_trace(session_id=session_id, input=f"[eval] {question}")
        scores: dict[str, float] = {}
        try:
            scores = evaluate_response(question, answer, contexts)
            for metric, value in scores.items():
                trace.log_score(name=metric, value=value)
        except Exception:
            pass
        tracer.flush()
        return scores
