"""The read budget as a prefix of whole articles, empty arguments, and the
search tool's own name in its configuration error."""

from __future__ import annotations

import random
from typing import Any

import pytest
from langchain_core.tools import ToolException

from conftest import Stub, load, offline_clients, stub_for
from langchain_lexapi import (
    InvalidArgumentError,
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

NOTE = "\n[Text cut by langchain-lexapi: "


def document(*texts: str) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    body["document"]["content"]["articles"] = [
        {"id": f"art_{n}", "number": f"Article {n}", "content": text}
        for n, text in enumerate(texts, 1)
    ]
    return body


def read(body: dict[str, Any], max_chars: int | None) -> Any:
    client, _ = offline_clients(stub_for("/documentContent", body))
    tool = LexAPIReadDocumentTool(client=client, max_chars=max_chars)
    call = {"type": "tool_call", "id": "c", "name": tool.name, "args": {"celex": "x"}}
    return tool.invoke(call)


def shown_texts(message: Any) -> list[str]:
    """The text of each shown article, without its cut note and Source line."""
    content = str(message.content).split("\n\n[Not shown: ")[0]
    out = []
    for n in range(1, len(message.artifact["passages"]) + 1):
        piece = content.split(f"[{n}] ", 1)[1].split("\nSource: ", 1)[0]
        out.append(piece.split(NOTE, 1)[0])
    return out


def test_after_a_cut_first_article_nothing_else_is_shown() -> None:
    words = " ".join(["word"] * 6008)  # about 30,040 characters
    message = read(document(words, "short article", "another"), 20_000)
    content = str(message.content)
    assert "[2]" not in content and NOTE in content
    assert content.endswith("art_2, art_3. Read one with the article argument.]")
    assert message.artifact["report"]["not_shown"] == 2


def test_after_a_stripped_first_article_over_the_budget_nothing_else_is_shown() -> None:
    message = read(document("\n" * 50 + "a" * 190, "b"), 200)
    content = str(message.content)
    assert content.startswith("[1] " + "a" * 190 + "\nSource:")
    assert "[2]" not in content and content.endswith("art_2. Read one with the article argument.]")


def test_read_budget_property() -> None:
    rng = random.Random(5)
    for _ in range(300):
        budget = rng.randrange(200, 1200)
        texts = [
            ("x" * rng.randrange(1, 60) + " ") * rng.randrange(1, 25) + "end"
            for _ in range(rng.randrange(1, 7))
        ]
        message = read(document(*texts), budget)
        shown = shown_texts(message)
        refs = [p["article_ref"] for p in message.artifact["passages"]]
        assert refs == [f"art_{n}" for n in range(1, len(refs) + 1)]  # a prefix
        assert sum(len(t) for t in shown) <= budget
        cut = NOTE in str(message.content)
        if cut or len(texts[0]) > budget:
            assert len(shown) == 1
        for text, original in zip(shown[1:], texts[1:], strict=False):
            assert text == original  # later articles are whole
        not_shown = message.artifact["report"]["not_shown"]
        assert len(shown) + not_shown == len(texts)


# ── empty arguments ─────────────────────────────────────────────────────

ROUTES = {
    "/legislation/semantic": (200, load("legislation-semantic.mixed")),
    "/search/semantic": (200, load("case-law.with-text")),
    "/documentContent": (200, load("document.read")),
    "/legislation/as-of": (200, load("as-of.resolved")),
}


@pytest.mark.parametrize("query", ["", " ", "\n\t "])
@pytest.mark.parametrize("cls", [LexAPILegislationRetriever, LexAPICaseLawRetriever])
async def test_retrievers_refuse_an_empty_query(cls: type[Any], query: str) -> None:
    stub = Stub(routes=dict(ROUTES))
    client, async_client = offline_clients(stub)
    r = cls(client=client, async_client=async_client)
    with pytest.raises(InvalidArgumentError, match="^query must not be empty$"):
        r.invoke(query)
    with pytest.raises(InvalidArgumentError, match="^query must not be empty$"):
        await r.ainvoke(query)
    assert stub.requests == []


@pytest.mark.parametrize(
    ("cls", "args", "expected"),
    [
        (LexAPISearchLegislationTool, {"query": "  "}, "Invalid argument: query must not be empty"),
        (LexAPIReadDocumentTool, {"celex": " "}, "Invalid argument: celex must not be empty"),
        (
            LexAPILegislationAsOfTool,
            {"celex": "\t", "as_of": "2099-06-01"},
            "Invalid argument: celex must not be empty",
        ),
    ],
)
async def test_tools_refuse_empty_arguments(
    cls: type[Any], args: dict[str, Any], expected: str
) -> None:
    stub = Stub(routes=dict(ROUTES))
    client, async_client = offline_clients(stub)
    for kwargs in ({"client": client, "async_client": async_client}, {"client": client}):
        tool = cls(**kwargs)
        with pytest.raises(ToolException) as info:
            tool.invoke(args)
        assert str(info.value) == expected
        with pytest.raises(ToolException) as info:
            await tool.ainvoke(args)
        assert str(info.value) == expected
    assert stub.requests == []


def test_search_tool_names_itself_when_only_an_async_client_was_given() -> None:
    _, async_client = offline_clients(Stub(routes=dict(ROUTES)))
    with pytest.raises(ValueError) as info:
        LexAPISearchLegislationTool(async_client=async_client).invoke({"query": "q"})
    assert str(info.value).startswith("LexAPISearchLegislationTool was given only an async_client")
