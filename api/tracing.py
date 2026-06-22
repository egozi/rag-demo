from __future__ import annotations

from functools import lru_cache
from typing import Any


class _NoOpTracer:
    """Returned when LANGFUSE_ENABLED=false. All methods are silent no-ops."""

    def log_generation(self, **_: Any) -> None:
        pass

    def log_span(self, **_: Any) -> None:
        pass

    def log_score(self, **_: Any) -> None:
        pass

    def start_trace(self, **_: Any) -> _TraceHandle:
        return _TraceHandle(None)

    def flush(self) -> None:
        pass


class _TraceHandle:
    """Wraps a Langfuse trace (or None for no-op) so callers don't branch."""

    def __init__(self, trace: Any):
        self._trace = trace

    def log_score(self, name: str, value: float, comment: str = "") -> None:
        if self._trace is None:
            return
        self._trace.score(name=name, value=value, comment=comment)


class _LangfuseTracer:
    def __init__(self, client: Any):
        self._lf = client

    def log_generation(
        self,
        name: str,
        model: str,
        input: list[dict],
        output: str,
        usage: dict[str, int],
        latency_ms: int,
    ) -> None:
        self._lf.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
            metadata={"latency_ms": latency_ms},
        )

    def log_span(self, name: str, **metadata: Any) -> None:
        self._lf.span(name=name, metadata=metadata)

    def log_score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        self._lf.score(trace_id=trace_id, name=name, value=value, comment=comment)

    def start_trace(self, session_id: str, input: str) -> _TraceHandle:
        trace = self._lf.trace(session_id=session_id, input=input)
        return _TraceHandle(trace)

    def flush(self) -> None:
        self._lf.flush()


@lru_cache
def get_tracer() -> _LangfuseTracer | _NoOpTracer:
    from api.config import get_settings

    settings = get_settings()
    if not settings.langfuse_enabled:
        return _NoOpTracer()

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return _LangfuseTracer(client)
    except Exception:
        return _NoOpTracer()
