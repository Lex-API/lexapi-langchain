"""The per-call report: delivered by callback event, never stored on the
retriever, correct under concurrency."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

from conftest import Stub, load, offline_clients
from langchain_lexapi import REPORT_EVENT, LexAPILegislationRetriever, RetrievalReport


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, UUID]] = []
        self.retriever_runs: list[UUID] = []

    def on_retriever_start(self, serialized: Any, query: str, *, run_id: UUID, **kw: Any) -> None:
        self.retriever_runs.append(run_id)

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, **kw: Any) -> None:
        self.events.append((name, data, run_id))


def per_query_stub() -> Stub:
    """Charges as many credits as the query's number, so each call's report
    is recognisable."""

    def reply(request: httpx.Request) -> httpx.Response:
        body = load("legislation-semantic.no-date")
        n = int(json.loads(request.content)["query"])
        body["credits"] = {"operation_weight": n, "remaining": 1000 - n}
        body["results"] = body["results"][: 1 + n % 2]
        return httpx.Response(200, json=body)

    return Stub(routes={"/legislation/semantic": reply})


def test_report_event_reaches_callbacks_on_the_retriever_run() -> None:
    client, _ = offline_clients(per_query_stub())
    recorder = Recorder()
    LexAPILegislationRetriever(client=client).invoke("3", config={"callbacks": [recorder]})
    ((name, data, run_id),) = recorder.events
    assert name == REPORT_EVENT
    assert isinstance(data, RetrievalReport)
    assert data.units_charged == 3
    assert run_id == recorder.retriever_runs[0]


def test_invoke_with_report_keeps_the_callers_callbacks() -> None:
    client, _ = offline_clients(per_query_stub())
    recorder = Recorder()
    r = LexAPILegislationRetriever(client=client)
    _, report = r.invoke_with_report("4", {"callbacks": [recorder]})
    assert report.units_charged == 4
    assert len(recorder.events) == 1
    manager = CallbackManager(handlers=[recorder])
    _, report = r.invoke_with_report("5", {"callbacks": manager})
    assert report.units_charged == 5
    assert len(recorder.events) == 2
    assert manager.handlers == [recorder]


def test_retriever_holds_no_per_call_state() -> None:
    client, _ = offline_clients(per_query_stub())
    r = LexAPILegislationRetriever(client=client)
    before = dict(r.__dict__), dict(r.__pydantic_private__ or {})
    r.invoke_with_report("2")
    assert (dict(r.__dict__), dict(r.__pydantic_private__ or {})) == before


def test_concurrent_threads_get_their_own_reports() -> None:
    client, _ = offline_clients(per_query_stub())
    r = LexAPILegislationRetriever(client=client)

    def run(n: int) -> tuple[int, RetrievalReport, int]:
        docs, report = r.invoke_with_report(str(n))
        return n, report, len(docs)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(run, range(1, 201)))
    for n, report, count in results:
        assert report.units_charged == n
        assert report.credits_remaining == 1000 - n
        assert report.returned == count == 1 + n % 2


async def test_concurrent_tasks_get_their_own_reports() -> None:
    _, async_client = offline_clients(per_query_stub())
    r = LexAPILegislationRetriever(async_client=async_client)
    results = await asyncio.gather(*(r.ainvoke_with_report(str(n)) for n in range(1, 101)))
    for n, (_, report) in enumerate(results, 1):
        assert report.units_charged == n
