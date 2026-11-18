"""langchain-tests standard suites, run offline.

The integration suites need no model: their "integration" calls go to the
mock transport, which answers with as many synthetic hits as the request's
``limit``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import RetrieversIntegrationTests, ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from conftest import SYN, TEST_KEY, Stub, load, offline_clients
from langchain_lexapi import (
    LexAPICaseLawRetriever,
    LexAPILegislationAsOfTool,
    LexAPILegislationRetriever,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)


def sized_stub() -> Stub:
    """Answers each search with exactly ``limit`` distinct hits."""

    def legislation(request: httpx.Request) -> httpx.Response:
        limit = json.loads(request.content)["limit"]
        hits = [
            {"law_id": f"32099R{n:04d}", "article_ref": "art_1", "score": 0.9, "text": SYN + str(n)}
            for n in range(limit)
        ]
        return httpx.Response(200, json={"success": True, "results": hits})

    def case_law(request: httpx.Request) -> httpx.Response:
        limit = json.loads(request.content)["limit"]
        hits = [
            {"score": 0.9, "text": SYN + str(n), "metadata": {"celex_id": f"62099CJ{n:04d}"}}
            for n in range(limit)
        ]
        return httpx.Response(200, json={"success": True, "results": hits})

    return Stub(
        routes={
            "/legislation/semantic": legislation,
            "/search/semantic": case_law,
            "/documentContent": (200, load("document.read")),
            "/legislation/as-of": (200, load("as-of.resolved")),
        }
    )


def clients() -> dict[str, Any]:
    client, async_client = offline_clients(sized_stub())
    return {"client": client, "async_client": async_client}


class TestLegislationRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return LexAPILegislationRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return clients()

    @property
    def retriever_query_example(self) -> str:
        return "transfer of personal data"


class TestCaseLawRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return LexAPICaseLawRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return clients()

    @property
    def retriever_query_example(self) -> str:
        return "right to be forgotten"


TOOL_CASES: list[tuple[type[BaseTool], dict[str, Any]]] = [
    (LexAPISearchLegislationTool, {"query": "transfer of personal data", "as_of": "2099-06-01"}),
    (LexAPIReadDocumentTool, {"celex": "32099R0001", "article": "art_1", "language": "en"}),
    (
        LexAPILegislationAsOfTool,
        {"celex": "32099R0001", "as_of": "2099-06-01", "provision": "art_17"},
    ),
]


class _ToolCase:
    cls: type[BaseTool]
    example: dict[str, Any]

    @property
    def tool_constructor(self) -> type[BaseTool]:
        return self.cls

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return clients()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return dict(self.example)

    @property
    def init_from_env_params(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return ({"LEXAPI_API_KEY": TEST_KEY}, {}, {})


def _suite(base: type[Any], cls: type[BaseTool], example: dict[str, Any]) -> type[Any]:
    return type(
        f"Test{cls.__name__}{base.__name__}", (_ToolCase, base), {"cls": cls, "example": example}
    )


for _cls, _example in TOOL_CASES:
    for _base in (ToolsUnitTests, ToolsIntegrationTests):
        _test = _suite(_base, _cls, _example)
        globals()[_test.__name__] = _test


def test_every_suite_is_collected() -> None:
    names = [name for name in globals() if name.startswith("Test")]
    assert len(names) == 2 + 2 * len(TOOL_CASES)


@pytest.mark.parametrize(("cls", "example"), TOOL_CASES)
def test_examples_validate_against_the_schemas(
    cls: type[BaseTool], example: dict[str, Any]
) -> None:
    tool = cls(**clients())
    assert tool.get_input_schema()(**example)
