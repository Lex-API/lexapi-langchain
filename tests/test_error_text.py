"""The text of a tool's error: its fixed format, and a wait only where
LexAPI's client would retry."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException
from lexapi import (
    CreditsExhaustedError,
    InvalidResponseError,
    LexAPI,
    LexAPIConnectionError,
    LexAPIError,
    PaymentPastDueError,
    RateLimitedError,
    TierForbiddenError,
)

import langchain_lexapi
import langchain_lexapi.tools as tools_module
from conftest import LOOPBACK, TEST_KEY, Stub, assert_local_base_url, load, offline_clients
from langchain_lexapi import LexAPILegislationRetriever, LexAPISearchLegislationTool, tool_error

PATH = "/legislation/semantic"
RESET = "2099-07-02T00:00:00.000Z"
BILLING = "https://lex-api.com/dashboard#billing"
APP = {"X-LexAPI-Version": "2.30.0"}


def reply(status: int, body: Any, headers: dict[str, str] | None = None) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers or {})

    return handle


def text_for(status: int, body: Any, headers: dict[str, str] | None = None) -> str:
    """The search tool's error text for one answer (no retries)."""
    client, _ = offline_clients(Stub(routes={PATH: reply(status, body, headers)}))
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    assert isinstance(info.value.__cause__, LexAPIError)
    assert str(tool_error(info.value.__cause__)) == str(info.value)
    return str(info.value)


def typed(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}, **extra}


PAST_DUE = load("error-403-past-due")
QUOTA_TYPED = typed("QUOTA_EXHAUSTED", "Daily quota exhausted", resetsAt=RESET)
QUOTA_LEGACY_CODE = {
    "error": "Daily limit exceeded",
    "code": "QUOTA_EXHAUSTED",
    "message": "You have used all 100 calls for today",
    "usage": {"resetsAt": RESET},
}
# Before API 2.4.0: no code (so RATE_LIMITED), a usage block and no wait.
QUOTA_LEGACY_SHAPE = {"error": "Daily quota exceeded", "usage": {"resetsAt": RESET}}

CASES = [
    (
        "credits",
        402,
        load("error-402"),
        None,
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. "
        "Credits reset at 2099-07-01T00:00:00.000Z.",
    ),
    (
        "credits with Retry-After",
        402,
        load("error-402"),
        {"Retry-After": "7"},
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. "
        "Credits reset at 2099-07-01T00:00:00.000Z.",
    ),
    (
        "quota, typed",
        429,
        QUOTA_TYPED,
        {"Retry-After": "7"},
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exhausted. This is an exhausted "
        f"quota, not a rate limit. Quota resets at {RESET}.",
    ),
    (
        "quota, typed, no reset time",
        429,
        typed("QUOTA_EXHAUSTED", "Daily quota exhausted"),
        None,
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exhausted. This is an exhausted "
        "quota, not a rate limit.",
    ),
    (
        "quota, legacy code",
        429,
        QUOTA_LEGACY_CODE,
        None,
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily limit exceeded: You have used all 100 "
        f"calls for today. This is an exhausted quota, not a rate limit. Quota resets at {RESET}.",
    ),
    (
        "quota, legacy code, no reset time",
        429,
        {k: v for k, v in QUOTA_LEGACY_CODE.items() if k != "usage"},
        None,
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily limit exceeded: You have used all 100 "
        "calls for today. This is an exhausted quota, not a rate limit.",
    ),
    (
        "quota, legacy shape (code RATE_LIMITED)",
        429,
        QUOTA_LEGACY_SHAPE,
        None,
        "LexAPI error RATE_LIMITED (HTTP 429): Daily quota exceeded. This is an exhausted "
        f"quota, not a rate limit. Quota resets at {RESET}.",
    ),
    (
        "past due, full body",
        403,
        PAST_DUE,
        None,
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment "
        "failed, so API requests are refused. Access ended at 2099-06-25T09:14:03.000Z. "
        f"Billing: {BILLING}",
    ),
    (
        "past due, bare",
        403,
        {"success": False, "code": "PAYMENT_PAST_DUE", "error": "Payment past due"},
        None,
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due.",
    ),
    (
        "past due on a 429 with Retry-After",
        429,
        typed("PAYMENT_PAST_DUE", "Payment past due"),
        {"Retry-After": "7"},
        "LexAPI error PAYMENT_PAST_DUE (HTTP 429): Payment past due.",
    ),
    (
        "past due, link already in the message",
        403,
        {**PAST_DUE, "message": f"Pay the open invoice at {BILLING}"},
        None,
        f"LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Pay the open invoice at "
        f"{BILLING}. Access ended at 2099-06-25T09:14:03.000Z.",
    ),
    (
        "rate limit",
        429,
        load("error-429"),
        {"Retry-After": "7"},
        "LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 7 s.",
    ),
    (
        "rate limit, fractional wait",
        429,
        typed("RATE_LIMITED", "Slow down"),
        {"Retry-After": "2.5"},
        "LexAPI error RATE_LIMITED (HTTP 429): Slow down. Retry after 2.5 s.",
    ),
    (
        "rate limit, zero wait",
        429,
        typed("RATE_LIMITED", "Slow down"),
        {"Retry-After": "0"},
        "LexAPI error RATE_LIMITED (HTTP 429): Slow down.",
    ),
    (
        "rate limit, no wait",
        429,
        typed("RATE_LIMITED", "Slow down"),
        None,
        "LexAPI error RATE_LIMITED (HTTP 429): Slow down.",
    ),
    (
        "503 with Retry-After",
        503,
        typed("SERVICE_UNAVAILABLE", "Busy"),
        {"Retry-After": "12", **APP},
        "LexAPI error SERVICE_UNAVAILABLE (HTTP 503): Busy.",
    ),
    (
        "404 with Retry-After",
        404,
        typed("NOT_FOUND", "No such document"),
        {"Retry-After": "12"},
        "LexAPI error NOT_FOUND (HTTP 404): No such document.",
    ),
    (
        "national allowance",
        402,
        typed(
            "NATIONAL_ALLOWANCE_EXHAUSTED",
            "National allowance used up",
            details={"resetsAt": RESET, "upgradeUrl": "https://lex-api.com/pricing"},
        ),
        {"Retry-After": "7"},
        "LexAPI error NATIONAL_ALLOWANCE_EXHAUSTED (HTTP 402): National allowance used up.",
    ),
    (
        "multi-line message",
        400,
        typed("INVALID_PARAMS", "Bad request:\n  query\tis\r\ntoo long\x00 here"),
        None,
        "LexAPI error INVALID_PARAMS (HTTP 400): Bad request: query is too long here.",
    ),
    (
        "message ending in a colon",
        400,
        typed("INVALID_PARAMS", "Rejected fields:"),
        None,
        "LexAPI error INVALID_PARAMS (HTTP 400): Rejected fields.",
    ),
    (
        "message of only a colon",
        400,
        typed("INVALID_PARAMS", " : "),
        None,
        "LexAPI error INVALID_PARAMS (HTTP 400).",
    ),
    (
        "blank message",
        400,
        typed("INVALID_PARAMS", "\n\t "),
        None,
        "LexAPI error INVALID_PARAMS (HTTP 400).",
    ),
]


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_error_text(status: int, body: Any, headers: Any, expected: str) -> None:
    assert text_for(status, body, headers) == expected


def test_the_quota_sentences_follow_the_flag_not_the_code() -> None:
    client, _ = offline_clients(Stub(routes={PATH: reply(429, QUOTA_LEGACY_SHAPE)}))
    with pytest.raises(RateLimitedError) as info:
        LexAPILegislationRetriever(client=client).invoke("q")
    assert info.value.code == "RATE_LIMITED"
    assert info.value.quota_exhausted
    assert "This is an exhausted quota, not a rate limit." in str(tool_error(info.value))


def test_a_past_due_error_is_a_tier_error() -> None:
    client, _ = offline_clients(Stub(routes={PATH: reply(403, PAST_DUE)}))
    with pytest.raises(TierForbiddenError) as info:
        LexAPILegislationRetriever(client=client).invoke("q")
    assert isinstance(info.value, PaymentPastDueError)
    assert "upgrade" not in str(tool_error(info.value)).lower()


# ── hand-built errors ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("wait", "printed"),
    [
        (7, "7"),
        (7.0, "7"),
        (2.5, "2.5"),
        (1_000_000.0, "1000000"),
        (0.25, "0.25"),
        (0, None),
        (0.0, None),
        (-3.0, None),
        (float("inf"), None),
        (float("-inf"), None),
        (float("nan"), None),
        (True, None),
    ],
)
def test_a_wait_is_a_plain_positive_decimal(wait: Any, printed: str | None) -> None:
    error = RateLimitedError("Slow down", status=429, retry_after=wait)
    text = str(tool_error(error))
    if printed is None:
        assert text == "LexAPI error RATE_LIMITED (HTTP 429): Slow down."
    else:
        assert text == f"LexAPI error RATE_LIMITED (HTTP 429): Slow down. Retry after {printed} s."


def test_no_wait_with_an_exhausted_quota_even_when_the_error_has_one() -> None:
    error = RateLimitedError("Quota used up", code="QUOTA_EXHAUSTED", status=429, retry_after=7)
    assert str(tool_error(error)) == (
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Quota used up. "
        "This is an exhausted quota, not a rate limit."
    )


def test_relayed_values_are_capped() -> None:
    long_url = "https://lex-api.com/billing?" + "x" * 2_000_000
    body = {
        **PAST_DUE,
        "message": "m" * 1500,
        "accessEndedAt": "t" * 300,
        "billingUrl": long_url,
    }
    text = text_for(403, body)
    assert len(text) < 2_000
    assert text == (
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: "
        + ("m" * 1500)[: 1000 - len("Payment past due: ")]
        + "…. Access ended at "
        + "t" * 200
        + "…. Billing: "
        + long_url[:200]
        + "…"
    )


def test_blank_relayed_values_are_absent() -> None:
    body = {**PAST_DUE, "accessEndedAt": " \n\t", "billingUrl": "\r\n"}
    assert text_for(403, body) == (
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment "
        "failed, so API requests are refused."
    )
    error = CreditsExhaustedError("Credit pool exhausted", status=402, resets_at="\x00 \x1f")
    assert str(tool_error(error)) == (
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted."
    )


def test_a_transport_error_has_no_status() -> None:
    error = LexAPIConnectionError("Connection error: refused", code="CONNECTION_ERROR")
    assert str(tool_error(error)) == "LexAPI error CONNECTION_ERROR: Connection error: refused."


def test_keys_are_still_masked() -> None:
    text = text_for(401, typed("UNAUTHORIZED", f"bad key {TEST_KEY}\nretry"))
    assert text == "LexAPI error UNAUTHORIZED (HTTP 401): bad key lex_*** retry."


@pytest.mark.parametrize("value", [ValueError("x"), "error", None, 429])
def test_anything_but_a_lexapi_error_is_a_type_error(value: Any) -> None:
    with pytest.raises(TypeError) as info:
        tool_error(value)
    assert str(info.value) == (
        f"tool_error expects a lexapi.LexAPIError, got {type(value).__name__}"
    )


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


def test_tool_error_is_public_and_the_private_name_still_works() -> None:
    assert langchain_lexapi.tool_error is tools_module.tool_error


# ── a wait is printed exactly when LexAPI's client retries ──────────────

RETRY_TABLE = [
    (429, typed("RATE_LIMITED", "Slow down"), {"Retry-After": "7"}),
    (429, typed("RATE_LIMITED", "Slow down"), {"Retry-After": "7", **APP}),
    (429, typed("RATE_LIMITED", "Slow down"), {}),
    (429, typed("SOMETHING_NEW", "Busy"), {"Retry-After": "7"}),
    (429, QUOTA_TYPED, {"Retry-After": "7"}),
    (429, QUOTA_LEGACY_CODE, {}),
    (429, QUOTA_LEGACY_SHAPE, {}),
    (429, typed("PAYMENT_PAST_DUE", "Payment past due"), {"Retry-After": "7"}),
    (403, PAST_DUE, {"Retry-After": "7"}),
    (402, load("error-402"), {"Retry-After": "7"}),
    (402, typed("NATIONAL_ALLOWANCE_EXHAUSTED", "Used up"), {"Retry-After": "7"}),
    (401, typed("UNAUTHORIZED", "Invalid API key"), {"Retry-After": "7"}),
    (403, typed("TIER_FORBIDDEN", "Plan"), {"Retry-After": "7"}),
    (404, typed("NOT_FOUND", "No such document"), {"Retry-After": "7"}),
    (500, typed("INTERNAL", "Oops"), {"Retry-After": "7", **APP}),
    (502, typed("UPSTREAM_ERROR", "Upstream"), {"Retry-After": "7", **APP}),
    (503, typed("SERVICE_UNAVAILABLE", "Busy"), {"Retry-After": "7", **APP}),
    (503, {"success": False, "partialReason": "TIMEOUT"}, {"Retry-After": "7", **APP}),
    (504, typed("TIMEOUT", "Timed out"), {"Retry-After": "7", **APP}),
    (520, typed("UPSTREAM_ERROR", "Proxy"), {"Retry-After": "7"}),
    (521, typed("UPSTREAM_ERROR", "Down"), {"Retry-After": "7"}),
    (522, typed("UPSTREAM_ERROR", "Timed out"), {}),
    (523, {"success": False, "code": "PAYMENT_PAST_DUE"}, {"Retry-After": "7"}),
    (524, typed("TIMEOUT", "Proxy timeout"), {"Retry-After": "7"}),
]


@pytest.mark.parametrize(("status", "body", "headers"), RETRY_TABLE)
def test_a_wait_is_printed_exactly_when_the_sdk_retries(
    monkeypatch: pytest.MonkeyPatch, status: int, body: Any, headers: dict[str, str]
) -> None:
    """The tool's call with a real client allowed one retry, sleeps patched:
    "Retry after" in the text if and only if the client tried again."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)  # the module LexAPI sleeps with
    stub = Stub(routes={PATH: reply(status, body, headers)})
    assert_local_base_url(LOOPBACK)
    client = LexAPI(
        TEST_KEY, base_url=LOOPBACK, transport=httpx.MockTransport(stub.handle), max_retries=1
    )
    with pytest.raises(ToolException) as info:
        LexAPISearchLegislationTool(client=client).invoke({"query": "q"})
    retried = len(stub.requests) == 2
    printed = "Retry after" in str(info.value)
    cause = info.value.__cause__
    assert isinstance(cause, LexAPIError)
    # A wait is printed only when the client retried, and whenever it retried
    # with a wait from the server.
    assert printed <= retried, str(info.value)
    if retried and cause.retry_after:
        assert printed, str(info.value)
