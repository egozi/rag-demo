import time
from typing import Any

import ollama as ollama_client
from openai import OpenAI

from api.config import get_settings
from api.tracing import get_tracer


class LLMClient:
    def __init__(self):
        self._settings = get_settings()
        if self._settings.llm_backend == "openai":
            self._openai = OpenAI(api_key=self._settings.openai_api_key)
            self._model = self._settings.openai_model
        elif self._settings.llm_backend == "huggingface":
            from transformers import pipeline
            self._pipe = pipeline(
                "text-generation",
                model=self._settings.hf_llm_model,
                device_map="auto",
                torch_dtype="auto",
            )
            self._model = self._settings.hf_llm_model
        else:
            self._ollama = ollama_client.Client(host=self._settings.ollama_host)
            self._model = self._settings.ollama_model

    def chat(self, messages: list[dict[str, str]], span_name: str = "llm-chat") -> str:
        tracer = get_tracer()
        start = time.time()

        if self._settings.llm_backend == "openai":
            resp = self._openai.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            text = resp.choices[0].message.content
            usage: dict[str, Any] = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }
        elif self._settings.llm_backend == "huggingface":
            result = self._pipe(messages, max_new_tokens=512, do_sample=False)
            text = result[0]["generated_text"][-1]["content"]
            usage = {"input": 0, "output": 0}
        else:
            resp = self._ollama.chat(
                model=self._model,
                messages=messages,
            )
            text = resp["message"]["content"]
            usage = {
                "input": resp.get("prompt_eval_count", 0),
                "output": resp.get("eval_count", 0),
            }

        latency_ms = int((time.time() - start) * 1000)
        tracer.log_generation(
            name=span_name,
            model=self._model,
            input=messages,
            output=text,
            usage=usage,
            latency_ms=latency_ms,
        )
        return text
