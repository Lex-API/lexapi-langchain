"""Tool names, descriptions and argument schemas are pinned: a change to what
the model is told must be a deliberate change to the snapshot."""

from __future__ import annotations

import re

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

import langchain_lexapi
from conftest import load, offline_clients, stub_for
from langchain_lexapi import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

TOOLS: list[type[BaseTool]] = [
    LexAPISearchLegislationTool,
    LexAPIReadDocumentTool,
    LexAPILegislationAsOfTool,
]
FORBIDDEN = re.compile(
    r"\b(complian\w*|complies|comply|legal advice|in force|guarantee\w*|accurate|authoritative)\b",
    re.IGNORECASE,
)


def test_tool_schemas_match_the_snapshot() -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    actual = []
    for cls in TOOLS:
        tool = cls(client=client)
        actual.append(
            {
                "name": tool.name,
                "description": tool.description,
                "response_format": tool.response_format,
                "openai": convert_to_openai_tool(tool),
            }
        )
    assert actual == load("tool-schemas")


def test_public_api() -> None:
    assert sorted(langchain_lexapi.__all__) == [
        "LexAPICaseLawRetriever",
        "LexAPILegislationAsOfTool",
        "LexAPILegislationRetriever",
        "LexAPIReadDocumentTool",
        "LexAPISearchLegislationTool",
        "METADATA_ALIASES",
        "REPORT_EVENT",
        "RetrievalReport",
        "__version__",
    ]
    assert langchain_lexapi.__version__ == "0.1.0"


def test_no_compliance_or_in_force_claims_in_tool_text() -> None:
    for entry in load("tool-schemas"):
        text = entry["description"] + str(entry["openai"])
        assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
