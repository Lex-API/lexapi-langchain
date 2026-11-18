"""Answers without their documented shape, the cut rule, copies and their
clients, and use after close with clients passed in."""

from __future__ import annotations

import contextlib
import io
import pathlib
import random
import re
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException
from lexapi import AsyncLexAPI, LexAPI

import langchain_lexapi._clients as clients_module
from conftest import LOOPBACK, TEST_KEY, Stub, load, offline_clients, stub_for
from langchain_lexapi import (
    ClientClosedError,
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)
from langchain_lexapi._documents import cut_text
from langchain_lexapi.tools import search_description

UNREADABLE = "LexAPI error INVALID_RESPONSE: The answer could not be read."
EMPTY_STRINGS = ("LexAPI returned no text", "LexAPI returned no answer")

# ── 1. malformed document and as-of answers ─────────────────────────────


def document_with(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    body["document"].update(changes)
    return body


def content_with(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    body["document"]["content"].update(changes)
    return body


SHAPES: dict[str, Any] = {
    "only success": {"success": True},
    "document is a string": {"success": True, "document": "x"},
    "document is null": {"success": True, "document": None},
    "content is a list": document_with(content=[]),
    "articles is a string": content_with(articles="x"),
    "articles holds a string": content_with(articles=["x"]),
}
TOOLS: list[tuple[type[Any], str, dict[str, Any]]] = [
    (LexAPIReadDocumentTool, "/documentContent", {"celex": "32099R0001"}),
    (LexAPIReadDocumentTool, "/documentContent", {"celex": "32099R0001", "article": "art_1"}),
    (
        LexAPILegislationAsOfTool,
        "/legislation/as-of",
        {"celex": "32099R0001", "as_of": "2099-06-01", "provision": "art_17"},
    ),
]


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize(("cls", "path", "args"), TOOLS)
async def test_malformed_answers_are_unreadable_never_empty(
    cls: type[Any], path: str, args: dict[str, Any], shape: str
) -> None:
    client, async_client = offline_clients(stub_for(path, SHAPES[shape]))
    tool = cls(client=client, async_client=async_client)
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == UNREADABLE
    assert info.value.__cause__ is None and info.value.__context__ is None
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == UNREADABLE
    handled = cls(client=client, async_client=async_client, handle_tool_error=True)
    call = {"type": "tool_call", "id": "c", "name": handled.name, "args": args}
    for message in (handled.invoke(call), await handled.ainvoke(call)):
        assert message.content == UNREADABLE
        assert not str(message.content).startswith(EMPTY_STRINGS)


@pytest.mark.parametrize(
    "body",
    [
        {"success": True, "document": "32099R0001"},  # no asOfResolved
        {"success": True, "document": "", "asOfResolved": False},
        {"success": True, "document": "32099R0001", "asOfResolved": "true"},
    ],
)
def test_as_of_answers_need_a_document_and_a_resolved_flag(body: dict[str, Any]) -> None:
    client, _ = offline_clients(stub_for("/legislation/as-of", body))
    with pytest.raises(ToolException, match=re.escape(UNREADABLE)):
        LexAPILegislationAsOfTool(client=client).invoke(
            {"celex": "32099R0001", "as_of": "2099-06-01"}
        )


def test_no_text_strings_remain_for_well_formed_documents() -> None:
    missing_article = content_with(articles=[])
    client, _ = offline_clients(stub_for("/documentContent", missing_article))
    tool = LexAPIReadDocumentTool(client=client)
    assert tool.invoke({"celex": "32099R0001", "article": "99"}) == (
        "LexAPI returned no text for article 99 of CELEX 32099R0001."
    )
    no_text = document_with(content=None)
    client, _ = offline_clients(stub_for("/documentContent", no_text))
    assert LexAPIReadDocumentTool(client=client).invoke({"celex": "32099R0001"}) == (
        "LexAPI returned no text for CELEX 32099R0001."
    )


def test_the_no_answer_string_is_gone() -> None:
    source = (pathlib.Path(__file__).parent.parent / "src/langchain_lexapi/tools.py").read_text()
    assert "returned no answer" not in source


def test_readme_tool_example_raises_on_a_bare_success(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = Stub(routes={"/legislation/as-of": (200, {"success": True})})
    monkeypatch.setattr(clients_module, "LexAPI", lambda *a, **k: offline_clients(stub)[0])
    monkeypatch.setattr(clients_module, "AsyncLexAPI", lambda *a, **k: offline_clients(stub)[1])
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    example = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)[2]
    with pytest.raises(ToolException, match=re.escape(UNREADABLE)):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(example, "README example 3", "exec"), {})


# ── 2. the cut rule ─────────────────────────────────────────────────────

NOTE = "\n[Text cut by langchain-lexapi: "


def test_pinned_cuts() -> None:
    assert cut_text("aaaa bbbbb cccc", 10)[0].startswith("aaaa bbbbb" + NOTE)
    heading = "Heading\n\n" + " ".join(["word"] * 1000)  # 5,000 characters of words
    cut, was_cut = cut_text(heading, 4000)
    kept = cut.split(NOTE, 1)[0]
    assert was_cut and len(kept) > 3990 and kept.endswith("word")
    assert kept.startswith("Heading\n\nword")


def test_cut_property_separators_only_in_the_second_half() -> None:
    rng = random.Random(11)
    alphabet = ["a", "b", " ", "\n", "\n\n", "\t"]
    for _ in range(4000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 400)))
        limit = rng.randrange(1, 300)
        cut, was_cut = cut_text(text, limit)
        body = text.lstrip()
        if not was_cut:
            assert cut in (text, body) and len(cut) <= limit
            continue
        kept, note = cut.split(NOTE, 1)
        assert 1 <= len(kept) <= limit and body.startswith(kept)
        assert note.startswith(f"{len(kept)} of {len(text)} characters shown.")
        expected = None
        for separator in ("\n\n", "\n", " "):
            at = body.rfind(separator, 0, limit + 1)
            if at > 0 and at >= limit // 2:
                expected = body[:at].rstrip()
                break
        assert kept == (expected if expected is not None else body[:limit].rstrip())


# ── 4. copies share clients but never close them ────────────────────────


class Recording(httpx.MockTransport):
    def __init__(self, stub: Stub, log: list[str]) -> None:
        super().__init__(stub.handle)
        self.log = log

    def close(self) -> None:
        self.log.append("close")


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    stub = Stub(routes={"/legislation/semantic": (200, load("legislation-semantic.mixed"))})

    def make_sync(*args: Any, **kwargs: Any) -> LexAPI:
        return LexAPI(TEST_KEY, base_url=LOOPBACK, transport=Recording(stub, log), max_retries=0)

    def make_async(*args: Any, **kwargs: Any) -> AsyncLexAPI:
        transport = httpx.MockTransport(stub.ahandle)
        return AsyncLexAPI(TEST_KEY, base_url=LOOPBACK, transport=transport, max_retries=0)

    monkeypatch.setattr(clients_module, "LexAPI", make_sync)
    monkeypatch.setattr(clients_module, "AsyncLexAPI", make_async)
    return log


def test_a_copy_never_closes_the_clients(built: list[str]) -> None:
    source = LexAPILegislationRetriever()
    copy = source.model_copy(update={"k": 2})
    copy.close()
    assert built == []
    assert len(source.invoke("q")) == 4
    with pytest.raises(ClientClosedError, match="LexAPILegislationRetriever is closed"):
        copy.invoke("q")


async def test_a_copy_after_its_source_closed_the_clients_raises(built: list[str]) -> None:
    source = LexAPILegislationRetriever()
    copy = source.model_copy()
    assert len(copy.invoke("q")) == 4
    await source.aclose()
    assert built == ["close"]
    assert copy._closed is False  # closing the source marks nothing on the copy
    with pytest.raises(ClientClosedError, match="is closed; create a new one."):
        copy.invoke("q")
    with pytest.raises(ClientClosedError, match="is closed; create a new one."):
        await copy.ainvoke("q")


async def test_a_tool_copy_after_its_source_closed_gives_a_tool_exception(
    built: list[str],
) -> None:
    source = LexAPISearchLegislationTool()
    copy = source.model_copy(update={"unresolved": "drop"})
    source.close()
    with pytest.raises(ToolException, match="LexAPISearchLegislationTool is closed"):
        copy.invoke({"query": "q"})
    with pytest.raises(ToolException, match="LexAPISearchLegislationTool is closed"):
        await copy.ainvoke({"query": "q"})
    handled = copy.model_copy(update={"handle_tool_error": True})
    message = handled.invoke(
        {"type": "tool_call", "id": "c", "name": copy.name, "args": {"query": "q"}}
    )
    assert message.content == "LexAPISearchLegislationTool is closed; create a new one."


def test_copies_of_passed_in_clients_are_independent() -> None:
    client, _ = offline_clients(
        stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    )
    source = LexAPILegislationRetriever(client=client)
    copy = source.model_copy()
    source.close()
    assert len(copy.invoke("q")) == 4  # passed-in clients stay open
    later = source.model_copy()  # a copy of a closed object starts open
    assert len(later.invoke("q")) == 4


# ── 5. model_copy with a generated description in the update ────────────


@pytest.mark.parametrize("target", ["withhold", "drop", "keep_labelled"])
@pytest.mark.parametrize("given", ["withhold", "drop", "keep_labelled"])
def test_copy_with_a_generated_description_in_the_update_regenerates(
    target: str, given: str
) -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    tool = LexAPISearchLegislationTool(client=client)
    copy = tool.model_copy(
        update={"unresolved": target, "description": search_description(given)}  # type: ignore[arg-type]
    )
    assert copy.description == search_description(copy.unresolved)
    assert "description" not in copy.model_fields_set
    explicit = tool.model_copy(update={"unresolved": target, "description": "Mine."})
    assert explicit.description == "Mine."


# ── 6. use after close with clients passed in ───────────────────────────

ALL_TOOLS: list[tuple[type[Any], dict[str, Any]]] = [
    (LexAPISearchLegislationTool, {"query": "q"}),
    (LexAPIReadDocumentTool, {"celex": "32099R0001"}),
    (LexAPILegislationAsOfTool, {"celex": "32099R0001", "as_of": "2099-06-01"}),
]
ROUTES = {
    "/legislation/semantic": (200, load("legislation-semantic.mixed")),
    "/search/semantic": (200, load("case-law.with-text")),
    "/documentContent": (200, load("document.read")),
    "/legislation/as-of": (200, load("as-of.resolved")),
}


def passed_in(which: str) -> dict[str, Any]:
    stub = Stub(routes=dict(ROUTES))
    client, async_client = offline_clients(stub)
    options: dict[str, dict[str, Any]] = {
        "both": {"client": client, "async_client": async_client},
        "sync": {"client": client},
        "async": {"async_client": async_client},
    }
    return options[which]


@pytest.mark.parametrize("which", ["both", "sync", "async"])
@pytest.mark.parametrize("closer", ["close", "aclose"])
@pytest.mark.parametrize("cls", [LexAPILegislationRetriever, LexAPICaseLawRetriever])
async def test_retrievers_with_passed_in_clients_refuse_use_after_close(
    cls: type[Any], closer: str, which: str
) -> None:
    r = cls(**passed_in(which))
    if closer == "close":
        r.close()
    else:
        await r.aclose()
    with pytest.raises(ClientClosedError, match=f"{cls.__name__} is closed"):
        r.invoke("q")
    with pytest.raises(ClientClosedError, match=f"{cls.__name__} is closed"):
        await r.ainvoke("q")


@pytest.mark.parametrize("which", ["both", "sync", "async"])
@pytest.mark.parametrize("closer", ["close", "aclose"])
@pytest.mark.parametrize(("cls", "args"), ALL_TOOLS)
async def test_tools_with_passed_in_clients_refuse_use_after_close(
    cls: type[Any], args: dict[str, Any], closer: str, which: str
) -> None:
    tool = cls(**passed_in(which))
    if closer == "close":
        tool.close()
    else:
        await tool.aclose()
    expected = f"{cls.__name__} is closed; create a new one."
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == expected
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == expected
