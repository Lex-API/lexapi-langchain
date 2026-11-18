"""Strict answers, copies, races with close, the read budget, the cut point
and strict arguments."""

from __future__ import annotations

import copy
import datetime
import pickle
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException
from lexapi import AsyncLexAPI, LexAPI
from pydantic import ValidationError

import langchain_lexapi._clients as clients_module
from conftest import LOOPBACK, SYN, TEST_KEY, Stub, load, offline_clients, stub_for
from langchain_lexapi import (
    ClientClosedError,
    InvalidArgumentError,
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)
from langchain_lexapi._clients import LexAPIClientFields
from langchain_lexapi._documents import cut_text
from langchain_lexapi.tools import read_description

UNREADABLE = "LexAPI error INVALID_RESPONSE: The answer could not be read."
ALL: list[type[Any]] = [
    LexAPILegislationRetriever,
    LexAPICaseLawRetriever,
    LexAPISearchLegislationTool,
    LexAPIReadDocumentTool,
    LexAPILegislationAsOfTool,
]

# ── 1. a document needs a celex; fullText must be a string ──────────────


def document_body(**document: Any) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    body["document"] = {k: v for k, v in {**body["document"], **document}.items() if v != ...}
    return body


def full_text_body(value: Any) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    body["document"]["content"] = {"fullText": value}
    return body


MALFORMED: dict[str, Any] = {
    "empty document": {"success": True, "document": {}},
    "celex missing": document_body(celex=...),
    "celex null": document_body(celex=None),
    "celex a number": document_body(celex=5),
    "celex blank": document_body(celex="  "),
    "fullText a list": full_text_body(["x"]),
    "fullText a number": full_text_body(5),
}


@pytest.mark.parametrize("shape", list(MALFORMED))
async def test_read_tool_rejects_documents_without_a_celex_or_with_bad_full_text(
    shape: str,
) -> None:
    client, async_client = offline_clients(stub_for("/documentContent", MALFORMED[shape]))
    tool = LexAPIReadDocumentTool(client=client, async_client=async_client)
    args = {"celex": "32099R0001"}
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == UNREADABLE
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == UNREADABLE
    handled = tool.model_copy(update={"handle_tool_error": True})
    message = handled.invoke({"type": "tool_call", "id": "c", "name": tool.name, "args": args})
    assert message.content == UNREADABLE


def test_a_full_text_document_is_read() -> None:
    client, _ = offline_clients(stub_for("/documentContent", full_text_body(SYN + "Whole text.")))
    content = LexAPIReadDocumentTool(client=client).invoke({"celex": "32099R0001"})
    assert content.startswith("[1] " + SYN + "Whole text.\nSource: CELEX 32099R0001")


# ── 2. copy.copy shares clients and never closes them ───────────────────


class Recording(httpx.MockTransport):
    def __init__(self, stub: Stub, log: list[str]) -> None:
        super().__init__(stub.handle)
        self.log = log

    def close(self) -> None:
        self.log.append("close")


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    stub = Stub(
        routes={
            "/legislation/semantic": (200, load("legislation-semantic.mixed")),
            "/documentContent": (200, load("document.read")),
        }
    )

    def make_sync(*args: Any, **kwargs: Any) -> LexAPI:
        return LexAPI(TEST_KEY, base_url=LOOPBACK, transport=Recording(stub, log), max_retries=0)

    def make_async(*args: Any, **kwargs: Any) -> AsyncLexAPI:
        transport = httpx.MockTransport(stub.ahandle)
        return AsyncLexAPI(TEST_KEY, base_url=LOOPBACK, transport=transport, max_retries=0)

    monkeypatch.setattr(clients_module, "LexAPI", make_sync)
    monkeypatch.setattr(clients_module, "AsyncLexAPI", make_async)
    return log


@pytest.mark.parametrize("cls", [LexAPILegislationRetriever, LexAPISearchLegislationTool])
def test_copy_copy_never_closes_the_source(cls: type[Any], built: list[str]) -> None:
    source = cls()
    duplicate = copy.copy(source)
    duplicate.close()
    assert built == []
    args: Any = "q" if cls is LexAPILegislationRetriever else {"query": "q"}
    assert source.invoke(args)
    source.close()
    assert built == ["close"]
    with pytest.raises((ClientClosedError, ToolException), match="is closed; create a new one."):
        copy.copy(source).invoke(args)


# ── 3. connection settings in a copy; deep copies and pickles ───────────


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("field", ["client", "async_client", "api_key", "base_url"])
def test_copies_cannot_change_connection_settings(cls: type[Any], field: str) -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    obj = cls(client=client)
    with pytest.raises(
        ValueError,
        match=f"Connection settings cannot be changed in a copy; create a new {cls.__name__}.",
    ):
        obj.model_copy(update={field: None})


@pytest.mark.parametrize("cls", ALL)
def test_deep_copies_and_pickles_are_refused(cls: type[Any]) -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    obj = cls(client=client)
    expected = f"{cls.__name__} holds an open client and cannot be deep-copied or pickled."
    for attempt in (
        lambda: obj.model_copy(deep=True),
        lambda: copy.deepcopy(obj),
        lambda: pickle.dumps(obj),
    ):
        with pytest.raises(TypeError) as info:
            attempt()
        assert str(info.value) == expected


# ── 4. a call that races close() ────────────────────────────────────────


@pytest.fixture
def no_open_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The open check passed just before close(): the call reaches httpx's
    closed client."""
    monkeypatch.setattr(LexAPIClientFields, "_check_open", lambda self: None)


async def test_a_retriever_call_racing_close_raises_client_closed(
    built: list[str], no_open_check: None
) -> None:
    r = LexAPILegislationRetriever()
    r.close()
    with pytest.raises(ClientClosedError, match="LexAPILegislationRetriever is closed"):
        r.invoke("q")
    r2 = LexAPILegislationRetriever()
    await r2.ainvoke("q")  # builds the async client
    await r2.aclose()
    with pytest.raises(ClientClosedError, match="LexAPILegislationRetriever is closed"):
        await r2.ainvoke("q")


@pytest.mark.parametrize(
    ("cls", "args"),
    [(LexAPISearchLegislationTool, {"query": "q"}), (LexAPIReadDocumentTool, {"celex": "x"})],
)
async def test_a_tool_call_racing_close_raises_a_tool_exception(
    cls: type[Any], args: dict[str, Any], built: list[str], no_open_check: None
) -> None:
    tool = cls()
    await tool.ainvoke(args)
    await tool.aclose()
    expected = f"{cls.__name__} is closed; create a new one."
    for run in (tool.invoke, tool.ainvoke):
        with pytest.raises(ToolException) as info:
            result = run(args)
            if hasattr(result, "__await__"):
                await result
        assert str(info.value) == expected


def test_an_unrelated_runtime_error_is_not_relabelled(built: list[str]) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("client has been closed")  # but this object is open

    client, _ = offline_clients(Stub(routes={"/legislation/semantic": boom}))
    with pytest.raises(RuntimeError) as info:
        LexAPILegislationRetriever(client=client).invoke("q")
    assert not isinstance(info.value, ClientClosedError)
    with pytest.raises(RuntimeError) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert not isinstance(info.value, (ClientClosedError, ToolException))


# ── 7. the read budget is whole articles ────────────────────────────────


def articles(*texts: str) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    for article, text in zip(body["document"]["content"]["articles"], texts, strict=True):
        article["content"] = text
    return body


def test_only_the_first_article_may_be_cut() -> None:
    body = articles("a" * 10, "aaaa\n\n" + "b" * 300, "c" * 5)
    client, _ = offline_clients(stub_for("/documentContent", body))
    message = LexAPIReadDocumentTool(client=client, max_chars=200).invoke(
        {"type": "tool_call", "id": "c", "name": "lexapi_read_document", "args": {"celex": "x"}}
    )
    content = str(message.content)
    assert content.startswith("[1] " + "a" * 10 + "\nSource:")
    assert "[2]" not in content and "Text cut" not in content
    assert content.endswith(
        "[Not shown: 2 more article(s) of CELEX x: art_2, art_3. "
        "Read one with the article argument.]"
    )
    report = message.artifact["report"]
    assert (report["returned"], report["not_shown"], report["truncated"]) == (1, 2, 0)


def test_a_long_first_article_is_cut_and_the_rest_listed() -> None:
    body = articles("x" * 500, "y" * 10, "z" * 10)
    client, _ = offline_clients(stub_for("/documentContent", body))
    content = LexAPIReadDocumentTool(client=client, max_chars=200).invoke({"celex": "x"})
    assert content.startswith("[1] " + "x" * 200 + "\n[Text cut by langchain-lexapi: 200 of 500")
    assert content.endswith("art_2, art_3. Read one with the article argument.]")


# ── 8. a separator that starts at the limit ─────────────────────────────


def test_a_separator_starting_at_the_limit_keeps_the_limit() -> None:
    text = "a" * 200 + "\n\n" + "b" * 100
    cut, was_cut = cut_text(text, 200)
    assert was_cut
    assert cut.startswith("a" * 200 + "\n[Text cut by langchain-lexapi: 200 of 302")


# ── 9. the read description follows the budget ──────────────────────────


def test_read_description_follows_max_chars() -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    tool = LexAPIReadDocumentTool(client=client)
    assert tool.description == read_description(20_000)
    assert "description" not in tool.model_fields_set
    smaller = tool.model_copy(update={"max_chars": 3000})
    assert smaller.description == read_description(3000)
    unbounded = tool.model_copy(update={"max_chars": None})
    assert unbounded.description == read_description(None)
    assert "budget" not in unbounded.description
    with pytest.raises(ValidationError, match="frozen"):
        tool.max_chars = 10  # type: ignore[misc]
    kept = LexAPIReadDocumentTool(client=client, description="Mine.")
    assert kept.model_copy(update={"max_chars": 500}).description == "Mine."


# ── 10. strict k and as_of ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "2099-06-01T10:00:00",
        "2099-06-01T00:00",
        datetime.datetime(2099, 6, 1),
        "20990601",
        20990601,
    ],
)
def test_as_of_with_a_time_part_or_another_form_is_refused(value: Any) -> None:
    client, _ = offline_clients(
        stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    )
    r = LexAPILegislationRetriever(client=client)
    with pytest.raises(InvalidArgumentError, match="as_of must be a date"):
        r.invoke("q", as_of=value)
    with pytest.raises(ValidationError, match="as_of must be a date"):
        LexAPILegislationRetriever(client=client, as_of=value)


def test_a_date_object_and_a_date_string_are_accepted() -> None:
    client, _ = offline_clients(
        stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    )
    for value in (datetime.date(2099, 6, 1), "2099-06-01"):
        assert LexAPILegislationRetriever(client=client, as_of=value).as_of == datetime.date(
            2099, 6, 1
        )


@pytest.mark.parametrize(
    "cls", [LexAPILegislationRetriever, LexAPICaseLawRetriever, LexAPISearchLegislationTool]
)
@pytest.mark.parametrize("k", ["3", 3.0, True])
def test_k_is_not_coerced(cls: type[Any], k: Any) -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    with pytest.raises(ValidationError):
        cls(client=client, k=k)


def test_a_later_article_is_never_cut_to_fit() -> None:
    # Cut at its space, the second article would fit what is left; it is listed instead.
    body = articles("a" * 10, "x" * 300 + " " + "y" * 500, "c" * 5)
    client, _ = offline_clients(stub_for("/documentContent", body))
    content = LexAPIReadDocumentTool(client=client, max_chars=500).invoke({"celex": "x"})
    assert "[2]" not in content and "Text cut" not in content
    assert content.endswith("art_2, art_3. Read one with the article argument.]")


async def test_as_of_tool_rejects_a_whitespace_only_document() -> None:
    """LexAPI's client accepts it; this package's own check does not."""
    body = load("as-of.unresolved")
    body["document"] = "   "
    client, async_client = offline_clients(stub_for("/legislation/as-of", body))
    tool = LexAPILegislationAsOfTool(client=client, async_client=async_client)
    args = {"celex": "32099R0001", "as_of": "2099-06-01", "provision": "art_17"}
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == UNREADABLE
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == UNREADABLE
