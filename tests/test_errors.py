"""Errors are raised, never turned into an empty result."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException
from lexapi import (
    AuthenticationError,
    CreditsExhaustedError,
    InvalidResponseError,
    LexAPIError,
    PaymentPastDueError,
    RateLimitedError,
    TierForbiddenError,
)

import langchain_lexapi
import langchain_lexapi.tools as tools_module
from conftest import TEST_KEY, Stub, load, offline_clients
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
    tool_error,
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


@pytest.mark.parametrize(
    ("name", "status", "expected"),
    [
        (
            "error-429",
            429,
            "LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 7 s.",
        ),
        (
            "error-402",
            402,
            "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. "
            "Credits reset at 2099-07-01T00:00:00.000Z.",
        ),
        (
            "error-403-past-due",
            403,
            "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription "
            "payment failed, so API requests are refused. "
            "Access ended at 2099-06-25T09:14:03.000Z. "
            "Billing: https://lex-api.com/dashboard#billing",
        ),
        ("error-401", 401, "LexAPI error UNAUTHORIZED (HTTP 401): Invalid API key."),
    ],
)
def test_tool_error_text_is_sentences(name: str, status: int, expected: str) -> None:
    client, _ = offline_clients(failing(name, status))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert str(info.value) == expected


def test_terminal_punctuation_is_not_doubled() -> None:
    body = {"success": False, "error": {"code": "RATE_LIMITED", "message": "Slow down!"}}
    client, _ = offline_clients(Stub(routes={"/legislation/semantic": (429, body)}))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert str(info.value) == "LexAPI error RATE_LIMITED (HTTP 429): Slow down! Retry after 7 s."


# ── payment past due, quota 429s, stalled iterators ─────────────────────

QUOTA_TYPED = {
    "success": False,
    "error": {"code": "QUOTA_EXHAUSTED", "message": "Daily quota exhausted"},
    "resetsAt": "2099-07-02T00:00:00.000Z",
}
QUOTA_LEGACY = {
    "error": "Daily limit exceeded",
    "code": "QUOTA_EXHAUSTED",
    "message": "You have used all 100 calls for today",
    "usage": {"resetsAt": "2099-07-02T00:00:00.000Z"},
}


def search_error(status: int, body: Any) -> ToolException:
    """The tool's error for one answer (the stub adds ``Retry-After: 7`` to a 429)."""
    client, _ = offline_clients(Stub(routes={"/legislation/semantic": (status, body)}))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    return info.value


def test_payment_past_due_is_a_tier_error_and_gives_no_upgrade_or_retry_hint() -> None:
    client, _ = offline_clients(failing("error-403-past-due", 403))
    with pytest.raises(TierForbiddenError) as info:
        LexAPILegislationRetriever(client=client).invoke("q")
    assert isinstance(info.value, PaymentPastDueError)
    text = str(tool_error(info.value))
    assert "upgrade" not in text.lower()
    assert "retry" not in text.lower()


def test_payment_past_due_on_a_429_is_not_told_to_retry() -> None:
    body = {"success": False, "error": {"code": "PAYMENT_PAST_DUE", "message": "Payment past due"}}
    error = search_error(429, body)
    assert isinstance(error.__cause__, PaymentPastDueError)
    assert error.__cause__.retry_after == 7  # the header was read, and is not passed on
    assert str(error) == "LexAPI error PAYMENT_PAST_DUE (HTTP 429): Payment past due."


def test_payment_past_due_without_details_names_no_billing_link() -> None:
    body = {"success": False, "code": "PAYMENT_PAST_DUE", "error": "Payment past due"}
    error = search_error(403, body)
    assert str(error) == "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due."


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            QUOTA_TYPED,
            "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exhausted. "
            "Quota resets at 2099-07-02T00:00:00.000Z.",
        ),
        (
            QUOTA_LEGACY,
            "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily limit exceeded: You have used all "
            "100 calls for today. Quota resets at 2099-07-02T00:00:00.000Z.",
        ),
    ],
)
def test_an_exhausted_quota_is_not_told_to_retry(body: Any, expected: str) -> None:
    error = search_error(429, body)
    assert isinstance(error.__cause__, RateLimitedError)
    assert error.__cause__.quota_exhausted
    assert str(error) == expected


def test_the_quota_flag_decides_even_with_a_wait_on_the_error() -> None:
    error = RateLimitedError("Quota used up", code="QUOTA_EXHAUSTED", status=429, retry_after=7)
    assert error.quota_exhausted
    assert str(tool_error(error)) == "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Quota used up."


def test_a_rate_limit_keeps_its_wait() -> None:
    error = search_error(429, load("error-429"))
    assert isinstance(error.__cause__, RateLimitedError)
    assert not error.__cause__.quota_exhausted
    assert str(error).endswith(" Retry after 7 s.")


def test_a_stalled_iterator_gives_the_unreadable_answer_text() -> None:
    stalled_page = {
        "success": True,
        "celex": "32099R0001",
        "citedBy": [{"id": "32099R0002", "edges": [{}]}],
        "offset": 999,
        "limit": 1,
    }
    client, _ = offline_clients(
        Stub(routes={"/citations/cited-by/32099R0001": (200, stalled_page)})
    )
    with pytest.raises(InvalidResponseError) as info:
        list(client.citations.iter_cited_by("32099R0001", page_size=1))
    assert str(tool_error(info.value)) == (
        "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."
    )


def test_tool_error_is_public_and_is_the_tools_text() -> None:
    assert langchain_lexapi.tool_error is tools_module.tool_error
    error = search_error(402, load("error-402"))
    assert isinstance(error.__cause__, CreditsExhaustedError)
    public = langchain_lexapi.tool_error(error.__cause__)
    assert isinstance(public, ToolException)
    assert str(public) == str(error)
