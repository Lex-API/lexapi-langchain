"""Tool names, descriptions and argument schemas are pinned: a change to what
the model is told must be a deliberate change to the snapshot."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

import langchain_lexapi
from conftest import load, offline_clients, stub_for
from langchain_lexapi import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

FORBIDDEN = re.compile(
    r"\b(complian\w*|complies|comply|legal advice|in force|guarantee\w*|accurate|authoritative)\b",
    re.IGNORECASE,
)


CONFIGS: list[tuple[type[BaseTool], dict[str, Any]]] = [
    (LexAPISearchLegislationTool, {"unresolved": "withhold"}),
    (LexAPISearchLegislationTool, {"unresolved": "drop"}),
    (LexAPISearchLegislationTool, {"unresolved": "keep_labelled"}),
    (LexAPIReadDocumentTool, {}),
    (LexAPIReadDocumentTool, {"max_chars": 5000}),
    (LexAPIReadDocumentTool, {"max_chars": None}),
    (LexAPILegislationAsOfTool, {}),
]


def snapshot() -> list[dict[str, Any]]:
    client, _ = offline_clients(stub_for("/x", {}))
    out = []
    for cls, kwargs in CONFIGS:
        tool = cls(client=client, **kwargs)
        out.append(
            {
                "config": {"class": cls.__name__, **kwargs},
                "name": tool.name,
                "description": tool.description,
                "response_format": tool.response_format,
                "openai": convert_to_openai_tool(tool),
            }
        )
    return out


def test_tool_schemas_match_the_snapshot() -> None:
    assert snapshot() == load("tool-schemas")


def test_search_description_follows_the_policy() -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    texts = {
        policy: LexAPISearchLegislationTool(client=client, unresolved=policy).description
        for policy in ("withhold", "drop", "keep_labelled")
    }
    assert "comes back as a notice without text" in texts["withhold"]
    assert "is left out, and the output says how many" in texts["drop"]
    assert "you get its latest text, under a label" in texts["keep_labelled"]
    assert len(set(texts.values())) == 3
    assert texts["withhold"] == LexAPISearchLegislationTool(client=client).description


def test_an_explicit_description_is_kept() -> None:
    client, _ = offline_clients(stub_for("/x", {}))
    tool = LexAPISearchLegislationTool(client=client, unresolved="drop", description="Mine.")
    assert tool.description == "Mine."


def test_public_api() -> None:
    assert sorted(langchain_lexapi.__all__) == [
        "ClientClosedError",
        "InvalidArgumentError",
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
