"""The offline guard works, the key stays secret, nothing happens at import."""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
from typing import Any

import langsmith
import pytest

from conftest import (
    OFFLINE_ERROR,
    TEST_KEY,
    assert_local_base_url,
    load,
    offline_clients,
    stub_for,
)
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)
from langchain_lexapi._clients import USER_AGENT_SUFFIX

ALL: list[type[Any]] = [
    LexAPILegislationRetriever,
    LexAPICaseLawRetriever,
    LexAPISearchLegislationTool,
    LexAPIReadDocumentTool,
    LexAPILegislationAsOfTool,
]
SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


@pytest.mark.parametrize(
    "url",
    [
        "https://lex-api.com/api/v1",
        "http://lex-api.com/api/v1",
        "http://127.0.0.1.example.com/api/v1",
        "http://localhost:9/api/v1",
    ],
)
def test_helper_refuses_non_loopback_base_urls(url: str) -> None:
    with pytest.raises(RuntimeError, match="refuse the base URL"):
        assert_local_base_url(url)
    with pytest.raises(RuntimeError, match="refuse the base URL"):
        offline_clients(stub_for("/x", {}), base_url=url)


def test_a_retriever_with_only_a_key_cannot_reach_the_real_service() -> None:
    r = LexAPILegislationRetriever(api_key=TEST_KEY)
    assert r._sync is not None
    assert str(r._sync._http.base_url).rstrip("/") == "https://lex-api.com/api/v1"
    with pytest.raises(RuntimeError, match=OFFLINE_ERROR):
        r.invoke("q")


async def test_async_real_transport_is_blocked_too() -> None:
    r = LexAPICaseLawRetriever(api_key=TEST_KEY)
    with pytest.raises(RuntimeError, match=OFFLINE_ERROR):
        await r.ainvoke("q")


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_sockets_are_blocked() -> None:
    with pytest.raises(BaseException):  # noqa: B017 - pytest-socket or our patch
        socket.create_connection(("192.0.2.1", 443), timeout=0.01)


def test_no_langsmith_client_is_created() -> None:
    with pytest.raises(RuntimeError):
        langsmith.Client()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    stub = stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    client, async_client = offline_clients(stub)
    LexAPILegislationRetriever(client=client).invoke("q")  # would raise on a client
    LexAPISearchLegislationTool(client=client).invoke({"query": "q"})


@pytest.mark.parametrize("cls", ALL)
def test_key_is_never_in_repr_or_dump(cls: type[Any]) -> None:
    obj = cls(api_key="lex_secret_value_for_repr_test")
    dump = obj.model_dump(exclude={"args_schema"})
    for text in (repr(obj), str(obj), str(dump), obj.model_dump_json(exclude={"args_schema"})):
        assert "lex_secret_value_for_repr_test" not in text
    assert "api_key" not in dump and "client" not in dump


def test_built_clients_use_the_key_base_url_and_user_agent() -> None:
    r = LexAPILegislationRetriever(api_key="lex_given", base_url="http://127.0.0.1:9/api/v1")
    assert r._sync is not None and r._async is None  # async is built on first use
    built = r._async_client()
    assert built is not None and r._async_client() is built
    for http in (r._sync._http, built._http):
        assert http.headers["x-api-key"] == "lex_given"
        assert str(http.base_url).rstrip("/") == "http://127.0.0.1:9/api/v1"
        assert http.headers["user-agent"].endswith(USER_AGENT_SUFFIX)
    assert USER_AGENT_SUFFIX == "langchain-lexapi/0.1.0"


def test_env_key_is_used_when_no_key_is_given() -> None:
    r = LexAPILegislationRetriever()
    assert r._sync is not None
    assert r._sync._http.headers["x-api-key"] == TEST_KEY


def test_no_key_at_all_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    with pytest.raises(ValueError, match="api_key is required"):
        LexAPILegislationRetriever()


def test_each_instance_builds_its_own_clients() -> None:
    a, b = LexAPILegislationRetriever(), LexAPILegislationRetriever()
    assert a._sync is not b._sync and a._async_client() is not b._async_client()


def test_import_opens_no_connection_and_builds_no_client() -> None:
    code = (
        "import socket, lexapi\n"
        "def refuse(*a, **k): raise SystemExit('network at import')\n"
        "socket.socket.connect = refuse\n"
        "socket.create_connection = refuse\n"
        "def no_client(*a, **k): raise SystemExit('client built at import')\n"
        "lexapi.LexAPI.__init__ = no_client\n"
        "lexapi.AsyncLexAPI.__init__ = no_client\n"
        "import langchain_lexapi\n"
        "print(sorted(langchain_lexapi.__all__))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LEXAPI_", "LANGSMITH_"))}
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert "LexAPILegislationRetriever" in done.stdout
