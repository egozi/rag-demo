from __future__ import annotations

import logging

from api.config import get_settings

logger = logging.getLogger(__name__)


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    # Lazy imports so the module loads without ragas installed, and tests can
    # mock this function without triggering the heavy import chain.

    # ragas 0.2.x references langchain_community.chat_models.vertexai which was
    # removed in langchain-community >= 0.2. Stub it out — we never use VertexAI.
    import sys
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")
        _stub.ChatVertexAI = None  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = _stub

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    settings = get_settings()

    logger.info("RAGAS | question: %s", question)
    logger.info("RAGAS | answer (%d chars): %s", len(answer), answer[:300])
    for i, ctx in enumerate(contexts):
        logger.info("RAGAS | context[%d] (%d chars): %s", i, len(ctx), ctx[:200])

    llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.ragas_model, api_key=settings.openai_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.openai_api_key)
    )
    dataset = EvaluationDataset.from_list([
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
    ])
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    logger.info("RAGAS | full result:\n%s", df.to_string())

    # ResponseRelevancy generates hypothetical questions internally; ragas logs
    # "LLM returned N generations" to stdout but doesn't expose them in the
    # result dataframe. Log what we can from the raw scores row.
    row = df.iloc[0].to_dict()
    logger.info(
        "RAGAS | scores — faithfulness=%.3f  answer_relevancy=%.3f  context_precision=%.3f",
        row.get("faithfulness") or 0.0,
        row.get("answer_relevancy") or 0.0,
        row.get("llm_context_precision_without_reference") or 0.0,
    )

    return {
        "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
        "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
        "context_precision": float(row.get("llm_context_precision_without_reference", 0.0) or 0.0),
    }
