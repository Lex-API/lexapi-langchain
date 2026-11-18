"""Construction and client lifetime: unknown arguments raise; an instance
closes only the clients it built; the async client is built on first use."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from lexapi import AsyncLexAPI, LexAPI
from pydantic import ValidationError

import langchain_lexapi._clients as clients_module
from conftest import LOOPBACK, TEST_KEY, Stub, load
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

ALL: list[type[Any]] = [
    LexAPILegislationRetriever,
    LexAPICaseLawRetriever,
    LexAPISearchLegislationTool,
    LexAPIReadDocumentTool,
    LexAPILegislationAsOfTool,
]


# ── unknown arguments ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (LexAPILegislationRetriever, {"emit_witheld": False}),
        (LexAPILegislationRetriever, {"top_k": 3}),
        (LexAPICaseLawRetriever, {"as_of": "2099-06-01"}),
        (LexAPICaseLawRetriever, {"unresolved": "drop"}),
        (LexAPISearchLegislationTool, {"celex": "32099R0001"}),
        (LexAPIReadDocumentTool, {"k": 3}),
        (LexAPILegislationAsOfTool, {"unresolved": "drop"}),
    ],
)
def test_unknown_constructor_arguments_raise(cls: type[Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        cls(**kwargs)


@pytest.mark.parametrize("cls", ALL)
def test_langchain_base_fields_still_construct(cls: type[Any]) -> None:
    obj = cls(tags=["t"], metadata={"m": 1})
    assert obj.tags == ["t"] and obj.metadata == {"m": 1}


# ── lifetime ────────────────────────────────────────────────────────────


class RecordingTransport(httpx.MockTransport):
    def __init__(self, stub: Stub, log: list[str], name: str) -> None:
        super().__init__(stub.handle)
        self.log, self.name = log, name

    def close(self) -> None:
        self.log.append(f"close {self.name}")


class AsyncRecordingTransport(httpx.MockTransport):
    def __init__(self, stub: Stub, log: list[str], name: str) -> None:
        super().__init__(stub.ahandle)
        self.log, self.name = log, name

    async def aclose(self) -> None:
        self.log.append(f"close {self.name}")


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Clients the package builds itself get recording transports; the log
    records each build and each close."""
    log: list[str] = []
    stub = Stub(routes={"/legislation/semantic": (200, load("legislation-semantic.mixed"))})

    def make_sync(key: Any, **kwargs: Any) -> LexAPI:
        log.append("build sync")
        transport = RecordingTransport(stub, log, "sync")
        return LexAPI(TEST_KEY, base_url=LOOPBACK, transport=transport, max_retries=0)

    def make_async(key: Any, **kwargs: Any) -> AsyncLexAPI:
        log.append("build async")
        transport = AsyncRecordingTransport(stub, log, "async")
        return AsyncLexAPI(TEST_KEY, base_url=LOOPBACK, transport=transport, max_retries=0)

    monkeypatch.setattr(clients_module, "LexAPI", make_sync)
    monkeypatch.setattr(clients_module, "AsyncLexAPI", make_async)
    return log


def test_async_client_is_built_on_first_async_use(built: list[str]) -> None:
    r = LexAPILegislationRetriever()
    r.invoke("q")
    assert built == ["build sync"]


async def test_async_use_builds_it_once(built: list[str]) -> None:
    r = LexAPILegislationRetriever()
    await r.ainvoke("q")
    await r.ainvoke("q")
    assert built == ["build sync", "build async"]


def test_close_closes_the_sync_client_it_built(built: list[str]) -> None:
    r = LexAPILegislationRetriever()
    r.close()
    assert built == ["build sync", "close sync"]


async def test_aclose_closes_both_clients_it_built(built: list[str]) -> None:
    r = LexAPILegislationRetriever()
    await r.ainvoke("q")
    await r.aclose()
    assert built == ["build sync", "build async", "close async", "close sync"]


def test_sync_context_manager(built: list[str]) -> None:
    with LexAPISearchLegislationTool() as tool:
        tool.invoke({"query": "q"})
    assert built == ["build sync", "close sync"]


async def test_async_context_manager(built: list[str]) -> None:
    async with LexAPISearchLegislationTool() as tool:
        await tool.ainvoke({"query": "q"})
    assert built == ["build sync", "build async", "close async", "close sync"]


async def test_clients_passed_in_are_never_closed() -> None:
    log: list[str] = []
    stub = Stub(routes={"/legislation/semantic": (200, load("legislation-semantic.mixed"))})
    sync = LexAPI(TEST_KEY, base_url=LOOPBACK, transport=RecordingTransport(stub, log, "sync"))
    asyn = AsyncLexAPI(
        TEST_KEY, base_url=LOOPBACK, transport=AsyncRecordingTransport(stub, log, "async")
    )
    for cls in ALL:
        obj = cls(client=sync, async_client=asyn)
        obj.close()
        await obj.aclose()
        with obj:
            pass
        async with obj:
            pass
    only_sync = LexAPILegislationRetriever(client=sync)
    await only_sync.ainvoke("q")  # runs the sync client in an executor; builds nothing
    await only_sync.aclose()
    assert log == []
    assert only_sync._async is None


def test_owned_clients_are_per_instance(built: list[str]) -> None:
    a, b = LexAPILegislationRetriever(), LexAPILegislationRetriever()
    a.close()
    assert built == ["build sync", "build sync", "close sync"]
    b.invoke("q")  # still open
