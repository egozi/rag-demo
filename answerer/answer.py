from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from retriever.retrieve import RetrievalResult

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT = """\
You are a helpful research assistant. Answer the user's question using ONLY the provided context. \
If the context does not contain the answer, say so plainly. \
Cite the chunk_id of any fact you use, in square brackets like [chunk_id].

Question: {query}

Context:
{chunks}
"""


@dataclass
class Answer:
    query: str
    text: str
    model: str
    chunks: list[RetrievalResult]


def answer(
        query: str,
        chunks: list[RetrievalResult],
        prompt: str = DEFAULT_PROMPT,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> Answer:
    
    # load .env so OpenAI() picks up OPENAI_API_KEY when no explicit key is passed
    if api_key is None:
        load_dotenv()
    client = OpenAI(api_key=api_key)

    formatted_chunks = "\n\n".join(
        f"[{c.chunk_id}] (source={c.source})\n{c.text}"
        for c in chunks
    )
    user_content = prompt.format(query=query, chunks=formatted_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )

    return Answer(
        query=query,
        text=response.choices[0].message.content,
        model=model,
        chunks=chunks,
    )
