"""Run the RAG eval suite: retrieval accuracy + answer accuracy.

Assumes the indexing pipeline has already populated `data/embeddings/`.
Usage (from repo root): python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from answerer import answer
from embedder import LocalEmbedder, ParquetStore
from retriever import retrieve

EMBEDDINGS_DIR = ROOT_DIR / "data" / "embeddings"
QUESTIONS_FILE = ROOT_DIR / "eval" / "questions.json"

# Tighter than DEFAULT_PROMPT — for eval we want the bare answer, no prose.
EVAL_PROMPT = """\
Answer the following question using ONLY the provided context. \
Give just the answer, no explanation, no citations.

Question: {query}

Context:
{chunks}
"""

DEFAULT_TOP_K = 5


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def retrieval_hit(retrieved_chunks, expected_excerpt: str) -> bool:
    excerpt_norm = normalize(expected_excerpt)
    return any(excerpt_norm in normalize(c.text) for c in retrieved_chunks)


def answer_hit(model_answer: str, expected: str) -> bool:
    return normalize(expected) in normalize(model_answer)


def run_eval(top_k: int = DEFAULT_TOP_K, verbose: bool = True) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    embedder = LocalEmbedder()
    data_store = ParquetStore(directory=EMBEDDINGS_DIR)

    results = []
    for question in questions:
        retrieved = retrieve(
            query=question["question"],
            embedder=embedder,
            data_store=data_store,
            top_k=top_k,
        )
        r_hit = retrieval_hit(retrieved, question["answer_location"]["excerpt"])

        if retrieved:
            response = answer(query=question["question"], chunks=retrieved, prompt=EVAL_PROMPT)
            a_hit = answer_hit(response.text, question["answer"])
            answer_text = response.text.strip()
        else:
            a_hit = False
            answer_text = "(no chunks retrieved)"

        results.append({
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "expected": question["answer"],
            "got": answer_text,
            "retrieval_hit": r_hit,
            "answer_hit": a_hit,
        })

        if verbose:
            r_mark = "PASS" if r_hit else "FAIL"
            a_mark = "PASS" if a_hit else "FAIL"
            print(f"[{question['id']}]  retrieval: {r_mark}   answer: {a_mark}")
            print(f"  Q: {question['question']}")
            print(f"  expected: {question['answer']}")
            print(f"  got:      {answer_text[:140]}")
            print()

    total = len(results)
    r_count = sum(r["retrieval_hit"] for r in results)
    a_count = sum(r["answer_hit"] for r in results)

    print(f"=== Summary (n={total}, top_k={top_k}) ===")
    print(f"retrieval accuracy: {r_count}/{total}  ({r_count/total:.0%})")
    print(f"answer accuracy:    {a_count}/{total}  ({a_count/total:.0%})")
    for qtype in ["multi_choice", "one_word"]:
        subset = [r for r in results if r["type"] == qtype]
        if subset:
            r_t = sum(r["retrieval_hit"] for r in subset)
            a_t = sum(r["answer_hit"] for r in subset)
            print(
                f"  [{qtype:12s}] retrieval={r_t}/{len(subset)} ({r_t/len(subset):.0%})  "
                f"answer={a_t}/{len(subset)} ({a_t/len(subset):.0%})"
            )

    return results


if __name__ == "__main__":
    run_eval()
