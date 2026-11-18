"""The three tools against synthetic responses."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from lexapi import passages
from lexapi.models import DocumentResponse, LegislationAsOfResponse, SemanticLegislationResponse

from conftest import SYN, Stub, load, offline_clients, stub_for
from langchain_lexapi import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)


def make(cls: type[Any], path: str, name: str, **kwargs: Any) -> tuple[Any, Stub]:
    stub = stub_for(path, load(name))
    client, async_client = offline_clients(stub)
    return cls(client=client, async_client=async_client, **kwargs), stub


def call(tool: Any, args: dict[str, Any]) -> ToolMessage:
    message = tool.invoke({"type": "tool_call", "id": "call-1", "name": tool.name, "args": args})
    assert isinstance(message, ToolMessage)
    return message


# ── search ──────────────────────────────────────────────────────────────


def test_search_tool_content_is_numbered_passages_from_the_core() -> None:
    tool, stub = make(
        LexAPISearchLegislationTool, "/legislation/semantic", "legislation-semantic.mixed", k=4
    )
    message = call(tool, {"query": "erasure", "as_of": "2099-06-01"})
    expected = passages.from_legislation_search(
        SemanticLegislationResponse(load("legislation-semantic.mixed")), as_of="2099-06-01"
    )
    assert message.content == "\n\n".join(
        f"[{n}] {passages.format_passage(p)}" for n, p in enumerate(expected.passages, 1)
    )
    assert "MUST-NOT-LEAK" not in str(message.content) + repr(message.artifact)
    assert stub.bodies() == [
        {"query": "erasure", "limit": 4, "include": ["text"], "asOf": "2099-06-01"}
    ]
    artifact = message.artifact
    assert artifact["report"]["units_charged"] == 5
    assert artifact["report"]["withheld"] == 1
    assert [m["id"] for m in artifact["passages"]] == [p.id for p in expected.passages]


def test_search_tool_without_date() -> None:
    tool, stub = make(
        LexAPISearchLegislationTool, "/legislation/semantic", "legislation-semantic.no-date"
    )
    message = call(tool, {"query": "erasure"})
    assert "asOf" not in stub.bodies()[0]
    assert str(message.content).startswith("[1] " + SYN)


def test_search_tool_empty_result_says_so() -> None:
    tool, _ = make(
        LexAPISearchLegislationTool, "/legislation/semantic", "legislation-semantic.empty"
    )
    message = call(tool, {"query": "nothing"})
    assert message.content == "LexAPI found no matching legislation for this query."


def test_search_tool_drop_policy_says_hits_were_left_out() -> None:
    tool, _ = make(
        LexAPISearchLegislationTool,
        "/legislation/semantic",
        "legislation-semantic.unresolved",
        unresolved="drop",
    )
    message = call(tool, {"query": "q", "as_of": "2099-06-01"})
    assert "1 hit(s) without a text for the date were left out" in str(message.content)
    assert "MUST-NOT-LEAK" not in str(message.content)


def test_search_tool_bad_date_is_a_tool_exception() -> None:
    tool, stub = make(
        LexAPISearchLegislationTool, "/legislation/semantic", "legislation-semantic.mixed"
    )
    with pytest.raises(ToolException, match="Invalid argument"):
        tool.invoke({"query": "q", "as_of": "2099-02-30"})
    assert stub.requests == []


# ── read a document ─────────────────────────────────────────────────────


def test_read_one_article() -> None:
    tool, stub = make(LexAPIReadDocumentTool, "/documentContent", "document.article")
    message = call(tool, {"celex": "32099R0001", "article": "art_2"})
    assert stub.bodies() == [
        {"celexNumber": "32099R0001", "include": ["metadata", "articles"], "articleId": "art_2"}
    ]
    (p,) = passages.from_document(DocumentResponse(load("document.article"))).passages
    assert message.content == f"[1] {passages.format_passage(p)}"
    assert message.artifact["passages"][0]["article_ref"] == "art_2"


def test_read_whole_document_asks_for_full_text_too() -> None:
    tool, stub = make(LexAPIReadDocumentTool, "/documentContent", "document.read")
    message = call(tool, {"celex": "32099R0001", "language": "de"})
    assert stub.bodies() == [
        {
            "celexNumber": "32099R0001",
            "language": "de",
            "include": ["metadata", "articles", "fullText"],
        }
    ]
    assert str(message.content).count("Source: CELEX 32099R0001") == 3


def test_read_whole_document_stops_at_an_article_boundary_and_lists_the_rest() -> None:
    tool, _ = make(LexAPIReadDocumentTool, "/documentContent", "document.read", max_chars=200)
    message = call(tool, {"celex": "32099R0001"})
    content = str(message.content)
    assert content.count("Source:") == 1
    assert content.endswith(
        "[Not shown: 2 more article(s) of CELEX 32099R0001: art_2, art_3. "
        "Read one with the article argument.]"
    )
    assert message.artifact["report"]["returned"] == 1


def test_read_missing_article_says_no_text() -> None:
    body = load("document.article")
    body["document"]["content"]["articles"] = []
    stub = stub_for("/documentContent", body)
    client, _ = offline_clients(stub)
    message = call(LexAPIReadDocumentTool(client=client), {"celex": "32099R0001", "article": "99"})
    assert message.content == "LexAPI returned no text for article 99 of CELEX 32099R0001."


# ── a provision for a date ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "provision", "start"),
    [
        ("as-of.resolved", "art_17", SYN + "Article 17 for the date."),
        ("as-of.unresolved", "art_17", "[No text: CELEX 32099R0001 art_17 is not established"),
        ("as-of.version-only", None, "[Version 02099R0001-20990101 (consolidated) is the version"),
    ],
)
def test_legislation_as_of(name: str, provision: str | None, start: str) -> None:
    tool, stub = make(LexAPILegislationAsOfTool, "/legislation/as-of", name)
    args: dict[str, Any] = {"celex": "32099R0001", "as_of": "2099-06-01"}
    if provision:
        args["provision"] = provision
    message = call(tool, args)
    (p,) = passages.from_legislation_as_of(
        LegislationAsOfResponse(load(name)), provision=provision
    ).passages
    assert message.content == passages.format_passage(p)
    assert str(message.content).startswith(start)
    assert stub.bodies()[0] == {
        "document": "32099R0001",
        "asOf": "2099-06-01",
        **({"provision": provision} if provision else {}),
    }


def test_as_of_unresolved_carries_the_reason_and_counts_it() -> None:
    tool, _ = make(LexAPILegislationAsOfTool, "/legislation/as-of", "as-of.unresolved")
    message = call(tool, {"celex": "32099R0001", "as_of": "2099-06-01", "provision": "art_17"})
    assert "amendment_not_consolidated" in str(message.content)
    assert message.artifact["report"]["withheld"] == 1
    assert message.artifact["passages"][0]["text_withheld"] is True


def test_as_of_answer_without_document() -> None:
    body = load("as-of.unresolved")
    body.pop("document")
    stub = stub_for("/legislation/as-of", body)
    client, _ = offline_clients(stub)
    message = call(LexAPILegislationAsOfTool(client=client), {"celex": "x", "as_of": "2099-06-01"})
    assert message.content == "LexAPI's answer named no document."


# ── async ───────────────────────────────────────────────────────────────


async def test_tools_async_equal_sync() -> None:
    cases: list[tuple[type[Any], str, str, dict[str, Any]]] = [
        (
            LexAPISearchLegislationTool,
            "/legislation/semantic",
            "legislation-semantic.mixed",
            {"query": "q", "as_of": "2099-06-01"},
        ),
        (LexAPIReadDocumentTool, "/documentContent", "document.read", {"celex": "32099R0001"}),
        (
            LexAPILegislationAsOfTool,
            "/legislation/as-of",
            "as-of.resolved",
            {"celex": "32099R0001", "as_of": "2099-06-01", "provision": "art_17"},
        ),
    ]
    for cls, path, name, args in cases:
        tool, stub = make(cls, path, name)
        tool_call = {"type": "tool_call", "id": "c", "name": tool.name, "args": args}
        sync_message = tool.invoke(tool_call)
        async_message = await tool.ainvoke(tool_call)
        assert sync_message.content == async_message.content
        assert sync_message.artifact == async_message.artifact
        assert stub.bodies()[0] == stub.bodies()[1]
        # Sync-only client: async falls back to an executor and still works.
        only_sync = cls(client=offline_clients(stub_for(path, load(name)))[0])
        assert (await only_sync.ainvoke(tool_call)).content == sync_message.content
