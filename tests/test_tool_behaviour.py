"""Tool behaviour: date checks, the description staying true, unreadable
answers, output strings, construction rules and use after close."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException
from lexapi import InvalidResponseError, LexAPIError
from pydantic import ValidationError

from conftest import TEST_KEY, Stub, load, offline_clients, stub_for
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
    RetrievalReport,
)
from langchain_lexapi._clients import ClientClosedError
from langchain_lexapi.tools import search_description, tool_error

POLICIES = ("withhold", "drop", "keep_labelled")


def as_of_tool(**kwargs: Any) -> tuple[LexAPILegislationAsOfTool, Stub]:
    stub = stub_for("/legislation/as-of", load("as-of.resolved"))
    client, async_client = offline_clients(stub)
    return LexAPILegislationAsOfTool(client=client, async_client=async_client, **kwargs), stub


# ── B1: dates on the as-of tool ─────────────────────────────────────────


async def test_as_of_tool_refuses_an_impossible_date_before_any_request() -> None:
    tool, stub = as_of_tool()
    args = {"celex": "32099R0001", "as_of": "2099-02-30"}
    expected = "Invalid argument: as_of must be a date in the form YYYY-MM-DD; got '2099-02-30'"
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == expected
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == expected
    assert stub.requests == []


def test_as_of_tool_schema_mismatch_is_the_frameworks_validation_error() -> None:
    tool, stub = as_of_tool()
    with pytest.raises(ValidationError):
        tool.invoke({"celex": "32099R0001", "as_of": "1 June 2099"})
    assert stub.requests == []


# ── B3: the search description stays true ───────────────────────────────


def search_tool(**kwargs: Any) -> LexAPISearchLegislationTool:
    client, _ = offline_clients(stub_for("/x", {}))
    return LexAPISearchLegislationTool(client=client, **kwargs)


@pytest.mark.parametrize("policy", POLICIES)
def test_generated_description_is_not_an_explicit_field(policy: str) -> None:
    tool = search_tool(unresolved=policy)
    assert tool.description == search_description(policy)  # type: ignore[arg-type]
    assert "description" not in tool.model_fields_set


@pytest.mark.parametrize("start", POLICIES)
@pytest.mark.parametrize("target", POLICIES)
def test_model_copy_regenerates_a_generated_description(start: str, target: str) -> None:
    copy = search_tool(unresolved=start).model_copy(update={"unresolved": target})
    assert copy.unresolved == target
    assert copy.description == search_description(copy.unresolved)
    assert "description" not in copy.model_fields_set


def test_model_copy_keeps_an_explicit_description() -> None:
    tool = search_tool(description="Mine.")
    assert tool.model_copy(update={"unresolved": "drop"}).description == "Mine."
    assert search_tool().model_copy(update={"description": "Theirs."}).description == "Theirs."


def test_unresolved_cannot_be_assigned() -> None:
    tool = search_tool()
    with pytest.raises(ValidationError, match="frozen"):
        tool.unresolved = "drop"  # type: ignore[misc]
    assert tool.description == search_description("withhold")


@pytest.mark.parametrize("policy", POLICIES)
def test_dump_and_rebuild_keeps_the_description_true(policy: str) -> None:
    dump = search_tool(unresolved=policy).model_dump(exclude={"args_schema"})
    rebuilt = LexAPISearchLegislationTool(**dump)
    assert rebuilt.description == search_description(policy)  # type: ignore[arg-type]
    for other in POLICIES:
        changed = LexAPISearchLegislationTool(**{**dump, "unresolved": other})
        assert changed.description == search_description(other)  # type: ignore[arg-type]


# ── unreadable answers ──────────────────────────────────────────────────

SECRET = "UNRESOLVED-TEXT-IN-A-CUT-OFF-BODY"
UNREADABLE = "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."


def cut_off(request: httpx.Request) -> httpx.Response:
    body = load("legislation-semantic.unresolved")
    hit = {"text": SECRET, **{k: v for k, v in body["results"][0].items() if k != "text"}}
    raw = json.dumps({"results": [hit], **{k: v for k, v in body.items() if k != "results"}})
    return httpx.Response(
        200,
        content=raw[: raw.index(SECRET) + len(SECRET) + 5].encode(),
        headers={"content-type": "application/json"},
    )


CUT_OFF_CASES: list[tuple[type[Any], dict[str, Any]]] = [
    (LexAPISearchLegislationTool, {"query": "q", "as_of": "2099-06-01"}),
    (LexAPIReadDocumentTool, {"celex": "32099R0001"}),
    (LexAPILegislationAsOfTool, {"celex": "32099R0001", "as_of": "2099-06-01"}),
]


def everything_raised(error: BaseException) -> str:
    parts, seen = [], set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(repr(current) + str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


@pytest.mark.parametrize(("cls", "args"), CUT_OFF_CASES)
async def test_a_cut_off_answer_never_reaches_the_model(
    cls: type[Any], args: dict[str, Any]
) -> None:
    routes = {
        p: cut_off for p in ("/legislation/semantic", "/documentContent", "/legislation/as-of")
    }
    client, async_client = offline_clients(Stub(routes=routes))  # type: ignore[arg-type]
    tool = cls(client=client, async_client=async_client)
    with pytest.raises(ToolException) as info:
        tool.invoke(args)
    assert str(info.value) == UNREADABLE
    assert SECRET not in everything_raised(info.value)
    with pytest.raises(ToolException) as info:
        await tool.ainvoke(args)
    assert str(info.value) == UNREADABLE
    assert SECRET not in everything_raised(info.value)
    handled = cls(client=client, handle_tool_error=True)
    message = handled.invoke({"type": "tool_call", "id": "c", "name": handled.name, "args": args})
    assert SECRET not in str(message.content) + repr(getattr(message, "artifact", None))


def test_the_retriever_still_raises_the_sdk_error() -> None:
    client, _ = offline_clients(Stub(routes={"/legislation/semantic": cut_off}))
    with pytest.raises(InvalidResponseError) as info:
        LexAPILegislationRetriever(client=client).invoke("q")
    assert SECRET not in str(info.value)
    assert SECRET not in everything_raised(info.value)


def wrong_shape(kind: str) -> Any:
    def reply(request: httpx.Request) -> httpx.Response:
        if kind == "empty":
            return httpx.Response(200, content=b"", headers={"content-type": "application/json"})
        if kind == "results-not-a-list":
            return httpx.Response(200, json={"success": True, "results": SECRET})
        if kind == "success-false":
            return httpx.Response(200, json={"success": False})
        body = {"success": False, "error": {"code": "UPSTREAM_ERROR", "message": SECRET}}
        return httpx.Response(200, json=body)

    return reply


SEARCH_SHAPES = ["empty", "results-not-a-list", "success-false", "success-false-envelope"]
ALL_SHAPES = ["empty", "success-false", "success-false-envelope"]
PATHS = ("/legislation/semantic", "/search/semantic", "/documentContent", "/legislation/as-of")


def shape_clients(kind: str) -> tuple[Any, Any]:
    return offline_clients(Stub(routes={p: wrong_shape(kind) for p in PATHS}))


@pytest.mark.parametrize("kind", SEARCH_SHAPES)
@pytest.mark.parametrize("cls", [LexAPILegislationRetriever, LexAPICaseLawRetriever])
async def test_retrievers_raise_on_a_wrong_shape_200(cls: type[Any], kind: str) -> None:
    client, async_client = shape_clients(kind)
    expected = LexAPIError if kind == "success-false-envelope" else InvalidResponseError
    with pytest.raises(expected) as info:
        cls(client=client).invoke("q")
    assert info.value.status == 200
    with pytest.raises(expected):
        await cls(async_client=async_client).ainvoke("q")


@pytest.mark.parametrize(
    ("cls", "args", "kind"),
    [(c, a, k) for c, a in CUT_OFF_CASES for k in ALL_SHAPES]
    + [(LexAPISearchLegislationTool, {"query": "q"}, "results-not-a-list")],
)
async def test_tools_give_the_fixed_text_on_a_wrong_shape_200(
    cls: type[Any], args: dict[str, Any], kind: str
) -> None:
    client, async_client = shape_clients(kind)
    tool = cls(client=client, async_client=async_client)
    for run in ("sync", "async"):
        with pytest.raises(ToolException) as info:
            if run == "sync":
                tool.invoke(args)
            else:
                await tool.ainvoke(args)
        assert str(info.value) == UNREADABLE
        assert SECRET not in everything_raised(info.value)
    handled = cls(client=client, handle_tool_error=True)
    message = handled.invoke({"type": "tool_call", "id": "c", "name": handled.name, "args": args})
    assert message.content == UNREADABLE


def test_any_error_on_a_2xx_status_gets_the_fixed_text() -> None:
    error = LexAPIError(f"unexpected shape: {SECRET}", code="SOMETHING", status=204)
    assert str(tool_error(error)) == (
        "LexAPI error INVALID_RESPONSE (HTTP 204): The answer could not be read."
    )


# ── output strings ──────────────────────────────────────────────────────


def search_once(body: Any, **kwargs: Any) -> str:
    client, _ = offline_clients(stub_for("/legislation/semantic", body))
    tool = LexAPISearchLegislationTool(client=client, **kwargs)
    args = {"query": "q", "as_of": "2099-06-01"}
    return str(
        tool.invoke({"type": "tool_call", "id": "c", "name": tool.name, "args": args}).content
    )


def test_hint_is_appended() -> None:
    body = load("legislation-semantic.mixed")
    body["hint"] = "Low similarity; try rephrasing."
    content = search_once(body)
    assert content.endswith("\n\nHint from LexAPI: Low similarity; try rephrasing.")
    assert "Hint from LexAPI" not in search_once(load("legislation-semantic.mixed"))


def test_drop_note_then_hint() -> None:
    body = load("legislation-semantic.unresolved")
    body["hint"] = "Rephrase."
    assert search_once(body, unresolved="drop") == (
        "LexAPI returned no matching articles.\n\n"
        "[1 hit(s) left out: LexAPI did not establish their text for 2099-06-01.]\n\n"
        "Hint from LexAPI: Rephrase."
    )


def test_drop_note_without_a_known_date() -> None:
    report = RetrievalReport(requested=5, returned=0, dropped=2)
    assert LexAPISearchLegislationTool._content([], report) == (
        "LexAPI returned no matching articles.\n\n"
        "[2 hit(s) left out: LexAPI did not establish their text for the date asked.]"
    )


def test_retry_after_on_any_error() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        body = {"success": False, "error": {"code": "SERVICE_UNAVAILABLE", "message": "Busy"}}
        return httpx.Response(503, json=body, headers={"Retry-After": "12"})

    client, _ = offline_clients(Stub(routes={"/legislation/semantic": unavailable}))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert str(info.value) == (
        "LexAPI error SERVICE_UNAVAILABLE (HTTP 503): Busy. Retry after 12 s."
    )


# ── construction ────────────────────────────────────────────────────────

ALL: list[type[Any]] = [
    LexAPILegislationRetriever,
    LexAPICaseLawRetriever,
    LexAPISearchLegislationTool,
    LexAPIReadDocumentTool,
    LexAPILegislationAsOfTool,
]


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("extra", [{"api_key": TEST_KEY}, {"base_url": "http://127.0.0.1:9"}])
def test_clients_and_key_or_url_together_are_refused(cls: type[Any], extra: dict[str, Any]) -> None:
    client, async_client = offline_clients(stub_for("/x", {}))
    for given in ({"client": client}, {"async_client": async_client}):
        with pytest.raises(
            ValueError, match="Pass either client/async_client or api_key/base_url, not both."
        ):
            cls(**given, **extra)


@pytest.mark.parametrize("cls", [LexAPIReadDocumentTool, LexAPILegislationAsOfTool])
def test_language_field_is_the_default_for_the_call(cls: type[Any]) -> None:
    path = "/documentContent" if cls is LexAPIReadDocumentTool else "/legislation/as-of"
    name = "document.read" if cls is LexAPIReadDocumentTool else "as-of.resolved"
    stub = stub_for(path, load(name))
    client, _ = offline_clients(stub)
    tool = cls(client=client, language="fr")
    args: dict[str, Any] = {"celex": "32099R0001"}
    if cls is LexAPILegislationAsOfTool:
        args["as_of"] = "2099-06-01"
    tool.invoke(args)
    tool.invoke({**args, "language": "de"})
    assert [body.get("language") for body in stub.bodies()] == ["fr", "de"]


@pytest.mark.parametrize(
    "cls", [LexAPISearchLegislationTool, LexAPIReadDocumentTool, LexAPILegislationAsOfTool]
)
def test_response_format_is_fixed(cls: type[Any]) -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    with pytest.raises(ValidationError, match="content_and_artifact"):
        cls(client=client, response_format="content")


# ── use after close ─────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [LexAPILegislationRetriever, LexAPICaseLawRetriever])
async def test_retrievers_refuse_use_after_close(cls: type[Any]) -> None:
    for closer in ("close", "aclose"):
        r = cls()
        if closer == "close":
            r.close()
        else:
            await r.aclose()
        with pytest.raises(ClientClosedError, match=f"{cls.__name__} is closed"):
            r.invoke("q")
        with pytest.raises(ClientClosedError, match=f"{cls.__name__} is closed"):
            await r.ainvoke("q")


@pytest.mark.parametrize(("cls", "args"), CUT_OFF_CASES)
async def test_tools_refuse_use_after_close(cls: type[Any], args: dict[str, Any]) -> None:
    tool = cls()
    await tool.aclose()
    with pytest.raises(ToolException, match=f"{cls.__name__} is closed"):
        tool.invoke(args)
    with pytest.raises(ToolException, match=f"{cls.__name__} is closed"):
        await tool.ainvoke(args)
