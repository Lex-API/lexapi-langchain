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
    assert message.content == "LexAPI returned no matching articles."


DROPPED_NOTE = "[1 hit(s) left out: LexAPI did not establish their text for 2099-06-01.]"


@pytest.mark.parametrize(
    ("name", "starts"),
    [
        ("legislation-semantic.unresolved", "LexAPI returned no matching articles."),
        ("legislation-semantic.mixed", "[1] " + SYN),
    ],
)
def test_search_tool_drop_policy_always_says_how_many_were_left_out(name: str, starts: str) -> None:
    tool, _ = make(LexAPISearchLegislationTool, "/legislation/semantic", name, unresolved="drop")
    message = call(tool, {"query": "q", "as_of": "2099-06-01"})
    content = str(message.content)
    assert content.startswith(starts)
    assert content.endswith("\n\n" + DROPPED_NOTE)
    assert "MUST-NOT-LEAK" not in content


def test_search_tool_keep_labelled_labels_the_latest_text() -> None:
    tool, _ = make(
        LexAPISearchLegislationTool,
        "/legislation/semantic",
        "legislation-semantic.unresolved",
        unresolved="keep_labelled",
    )
    content = str(call(tool, {"query": "q", "as_of": "2099-06-01"}).content)
    assert content.startswith("[1] [Latest indexed text of CELEX 32099L0003 art_12; not est")


def test_search_tool_with_only_an_async_client_raises_the_configuration_error() -> None:
    stub = stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    _, async_client = offline_clients(stub)
    tool = LexAPISearchLegislationTool(async_client=async_client)
    with pytest.raises(ValueError, match="only an async_client") as info:
        tool.invoke({"query": "q"})
    assert not isinstance(info.value, ToolException)


def test_search_tool_bad_date_is_a_tool_exception() -> None:
    tool, stub = make(
        LexAPISearchLegislationTool, "/legislation/semantic", "legislation-semantic.mixed"
    )
    with pytest.raises(ToolException) as info:
        tool.invoke({"query": "q", "as_of": "2099-02-30"})
    assert str(info.value) == (
        "Invalid argument: as_of must be a date in the form YYYY-MM-DD; got '2099-02-30'"
    )
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


def long_document(*sizes: int, prefix: str = SYN) -> dict[str, Any]:
    body: dict[str, Any] = load("document.read")
    for article, size in zip(body["document"]["content"]["articles"], sizes, strict=True):
        article["content"] = prefix + "z" * (size - len(prefix))
    return body


def read(body: dict[str, Any], **kwargs: Any) -> ToolMessage:
    client, _ = offline_clients(stub_for("/documentContent", body))
    return call(LexAPIReadDocumentTool(client=client, **kwargs), {"celex": "32099R0001"})


def test_read_budget_bounds_the_text_and_lists_the_rest() -> None:
    message = read(long_document(150, 100, 100), max_chars=300)
    content = str(message.content)
    assert content.count("Source:") == 2
    assert content.endswith(
        "[Not shown: 1 more article(s) of CELEX 32099R0001: art_3. "
        "Read one with the article argument.]"
    )
    report = message.artifact["report"]
    assert (report["returned"], report["not_shown"], report["truncated"]) == (2, 1, 0)


def test_read_budget_counts_only_what_was_shown_as_truncated() -> None:
    message = read(long_document(500, 500, 100, prefix=""), max_chars=300)
    content = str(message.content)
    assert content.count("Source:") == 1
    assert "[Text cut by langchain-lexapi: " in content
    assert content.endswith("art_2, art_3. Read one with the article argument.]")
    report = message.artifact["report"]
    assert (report["returned"], report["not_shown"], report["truncated"]) == (1, 2, 1)


def test_read_text_shown_never_exceeds_the_budget() -> None:
    for sizes in [(250, 40, 40), (90, 90, 90), (400, 10, 10), (10, 10, 400)]:
        message = read(long_document(*sizes), max_chars=200)
        docs = message.artifact["passages"]
        texts = [str(message.content).split(f"[{n}] ", 1)[1] for n in range(1, len(docs) + 1)]
        shown = sum(len(t.split("\nSource: ", 1)[0]) for t in texts)
        assert shown <= 200 + len(
            "\n[Text cut by langchain-lexapi: 000 of 000 characters shown. "
            "The full text is at the source.]"
        )


def test_read_without_budget_returns_everything() -> None:
    message = read(long_document(5000, 5000, 5000), max_chars=None)
    assert str(message.content).count("Source:") == 3
    assert message.artifact["report"]["not_shown"] == 0


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
