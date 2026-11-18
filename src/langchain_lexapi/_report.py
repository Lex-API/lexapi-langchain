"""Per-call delivery of a :class:`RetrievalReport` through LangChain callbacks.

A retriever sends its report as a custom callback event
(:data:`REPORT_EVENT`) on its own run. ``invoke_with_report`` adds a
handler to that one call's callbacks and reads the report back from it, so
concurrent calls never see each other's report and the retriever holds no
per-call state.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.runnables import RunnableConfig, ensure_config

from langchain_lexapi._documents import REPORT_EVENT, RetrievalReport


class ReportCatcher(BaseCallbackHandler):
    """Keeps the report of the one call it is attached to."""

    run_inline = True

    def __init__(self) -> None:
        self.report: RetrievalReport | None = None

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if name == REPORT_EVENT and isinstance(data, RetrievalReport) and self.report is None:
            self.report = data


def config_with_catcher(config: RunnableConfig | None) -> tuple[RunnableConfig, ReportCatcher]:
    """A copy of ``config`` whose callbacks also include a new catcher."""
    catcher = ReportCatcher()
    out = ensure_config(config)
    callbacks = out.get("callbacks")
    if callbacks is None:
        out["callbacks"] = [catcher]
    elif isinstance(callbacks, BaseCallbackManager):
        manager = callbacks.copy()
        manager.add_handler(catcher, inherit=True)
        out["callbacks"] = manager
    else:
        out["callbacks"] = [*callbacks, catcher]
    return out, catcher


def caught(catcher: ReportCatcher) -> RetrievalReport:
    if catcher.report is None:  # pragma: no cover - every call path sends one
        raise RuntimeError("the retriever sent no report")
    return catcher.report
