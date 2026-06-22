from __future__ import annotations

from api.config import get_settings


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    # Lazy imports: ragas pulls in langchain_community at import time which may
    # not be fully installed in all environments. Importing here keeps the module
    # loadable without ragas, and lets tests mock this function without triggering
    # the heavy import chain.
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    settings = get_settings()
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.ragas_model, api_key=settings.openai_api_key)
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
    )
    scores = result.to_pandas().iloc[0].to_dict()
    return {
        "faithfulness": float(scores.get("faithfulness", 0.0) or 0.0),
        "response_relevancy": float(scores.get("response_relevancy", 0.0) or 0.0),
        "context_precision": float(scores.get("llm_context_precision_without_reference", 0.0) or 0.0),
    }
