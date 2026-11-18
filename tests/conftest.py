"""Offline test harness. No test reaches a network, a LexAPI server or a model.

Layers:

1. ``pytest-socket`` (``--disable-socket`` in ``pyproject.toml``).
2. The autouse ``_offline`` fixture: the real httpx transports and
   ``socket.connect`` raise, LangSmith clients cannot be created, and the
   environment holds a fake LexAPI key and no model keys.
3. :func:`offline_clients` builds SDK clients on ``httpx.MockTransport`` and
   refuses any base URL that is not ``http://127.0.0.1:``.

Every fixture body is synthetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import langsmith
import pytest
from lexapi import AsyncLexAPI, LexAPI

# Before anything can read them.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _name in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(_name, None)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LOOPBACK = "http://127.0.0.1:9/api/v1"
TEST_KEY = "lex_offline_test_key_0123456789"
SYN = "[SYNTHETIC FIXTURE - not the text of the act] "
OFFLINE_ERROR = "Offline tests blocked a real network request"


def load(name: str) -> Any:
    """A fixture body, freshly parsed."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def assert_local_base_url(url: str) -> None:
    """Refuse any base URL that is not loopback over plain HTTP."""
    if not url.startswith("http://127.0.0.1:"):
        raise RuntimeError(f"Offline tests refuse the base URL {url!r}")


Reply = tuple[int, Any] | Callable[[httpx.Request], httpx.Response]


@dataclass
class Stub:
    """Answers by path; records every request it is sent."""

    routes: dict[str, Reply] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) if r.content else None for r in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1", request.url
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        reply = self.routes.get(path)
        if reply is None:
            return httpx.Response(404, json={"success": False, "error": {"code": "NOT_FOUND"}})
        if callable(reply):
            return reply(request)
        status, body = reply
        headers = {"Retry-After": "7"} if status == 429 else {}
        return httpx.Response(status, json=body, headers=headers)

    async def ahandle(self, request: httpx.Request) -> httpx.Response:
        return self.handle(request)


def offline_clients(stub: Stub, base_url: str = LOOPBACK) -> tuple[LexAPI, AsyncLexAPI]:
    """A sync and an async SDK client answering from ``stub``."""
    assert_local_base_url(base_url)
    sync = LexAPI(
        TEST_KEY, base_url=base_url, transport=httpx.MockTransport(stub.handle), max_retries=0
    )
    asyn = AsyncLexAPI(
        TEST_KEY, base_url=base_url, transport=httpx.MockTransport(stub.ahandle), max_retries=0
    )
    for client in (sync, asyn):
        assert_local_base_url(str(client._http.base_url))
    return sync, asyn


def stub_for(path: str, body: Any, status: int = 200) -> Stub:
    return Stub(routes={path: (status, body)})


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(OFFLINE_ERROR)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(langsmith.Client, "__init__", refuse)
    monkeypatch.setenv("LEXAPI_API_KEY", TEST_KEY)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for name in (
        "LEXAPI_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
