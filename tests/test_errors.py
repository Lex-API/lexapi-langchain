"""Errors are raised, never turned into an empty result."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException
from lexapi import (
    AuthenticationError,
    CreditsExhaustedError,
    LexAPIError,
    PaymentPastDueError,
    RateLimitedError,
)

from conftest import TEST_KEY, Stub, load, offline_clients
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

ERRORS = [
    ("error-402", 402, CreditsExhaustedError, "CREDITS_EXHAUSTED"),
    ("error-403-past-due", 403, PaymentPastDueError, "PAYMENT_PAST_DUE"),
    ("error-429", 429, RateLimitedError, "RATE_LIMITED"),
    ("error-401", 401, AuthenticationError, "UNAUTHORIZED"),
]
PATHS = ["/legislation/semantic", "/search/semantic", "/documentContent", "/legislation/as-of"]


def failing(name: str, status: int) -> Stub:
    return Stub(routes={path: (status, load(name)) for path in PATHS})


RETRIEVERS: list[type[Any]] = [LexAPILegislationRetriever, LexAPICaseLawRetriever]
TOOLS: list[tuple[type[Any], dict[str, Any]]] = [
    (LexAPISearchLegislationTool, {"query": "q"}),
    (LexAPIReadDocumentTool, {"celex": "32099R0001"}),
    (LexAPILegislationAsOfTool, {"celex": "32099R0001", "as_of": "2099-06-01"}),
]


@pytest.mark.parametrize(("name", "status", "error", "code"), ERRORS)
@pytest.mark.parametrize("cls", RETRIEVERS)
def test_retrievers_raise_sdk_errors(
    cls: type[Any], name: str, status: int, error: type[LexAPIError], code: str
) -> None:
    client, _ = offline_clients(failing(name, status))
    with pytest.raises(error) as info:
        cls(client=client).invoke("q")
    assert info.value.code == code


@pytest.mark.parametrize(("name", "status", "error", "code"), ERRORS)
@pytest.mark.parametrize("cls", RETRIEVERS)
async def test_retrievers_raise_sdk_errors_async(
    cls: type[Any], name: str, status: int, error: type[LexAPIError], code: str
) -> None:
    _, async_client = offline_clients(failing(name, status))
    with pytest.raises(error):
        await cls(async_client=async_client).ainvoke("q")


@pytest.mark.parametrize(("name", "status", "error", "code"), ERRORS)
@pytest.mark.parametrize(("cls", "args"), TOOLS)
def test_tools_raise_tool_exception_with_code_and_message(
    cls: type[Any],
    args: dict[str, Any],
    name: str,
    status: int,
    error: type[LexAPIError],
    code: str,
) -> None:
    client, _ = offline_clients(failing(name, status))
    with pytest.raises(ToolException) as info:
        cls(client=client).invoke(args)
    text = str(info.value)
    assert text.startswith(f"LexAPI error {code} (HTTP {status}): ")
    assert isinstance(info.value.__cause__, error)
    assert info.value.__cause__.message in text
    assert TEST_KEY not in text


@pytest.mark.parametrize(("name", "status", "error", "code"), ERRORS)
@pytest.mark.parametrize(("cls", "args"), TOOLS)
async def test_tools_raise_tool_exception_async(
    cls: type[Any],
    args: dict[str, Any],
    name: str,
    status: int,
    error: type[LexAPIError],
    code: str,
) -> None:
    _, async_client = offline_clients(failing(name, status))
    with pytest.raises(ToolException, match=f"LexAPI error {code}"):
        await cls(async_client=async_client).ainvoke(args)


def test_tool_error_details() -> None:
    client, _ = offline_clients(failing("error-402", 402))
    with pytest.raises(ToolException, match="Credits reset at 2099-07-01T00:00:00.000Z"):
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    client, _ = offline_clients(failing("error-403-past-due", 403))
    with pytest.raises(ToolException, match="Billing: https://lex-api.com/dashboard#billing"):
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    client, _ = offline_clients(failing("error-429", 429))
    with pytest.raises(ToolException, match="Retry after 7 s"):
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})


def test_a_key_in_an_error_message_is_masked() -> None:
    body = {"success": False, "error": {"code": "UNAUTHORIZED", "message": f"bad key {TEST_KEY}"}}
    client, _ = offline_clients(Stub(routes={"/legislation/semantic": (401, body)}))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert TEST_KEY not in str(info.value)
    assert "lex_***" in str(info.value)


def test_handle_tool_error_turns_it_into_a_message_when_asked() -> None:
    client, _ = offline_clients(failing("error-402", 402))
    tool = LexAPISearchLegislationTool(client=client, handle_tool_error=True)
    assert str(tool.invoke({"query": "q"})).startswith("LexAPI error CREDITS_EXHAUSTED")
