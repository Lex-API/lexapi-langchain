"""Generated bodies: under ``withhold`` and ``drop`` no word of an unresolved
hit's text or snippet reaches any Document, any metadata value, any tool
output or any report.

Every unresolved hit's text and snippet are made of tokens that occur
nowhere else in the body, so finding any of them is a leak."""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest

from conftest import offline_clients, stub_for
from langchain_lexapi import LexAPILegislationRetriever, LexAPISearchLegislationTool

CASES = 520
AS_OF = "2099-06-01"
REASONS = ["fetch_timeout", "amendment_not_consolidated", "not_yet_in_force", None]


def token(rng: random.Random, tag: str) -> str:
    return f"{tag}{rng.randrange(10**9):09d}"


def make_hit(rng: random.Random, n: int, secrets: set[str]) -> dict[str, Any]:
    celex = f"3209{rng.randrange(10)}R{n:04d}"
    hit: dict[str, Any] = {
        "law_id": celex,
        "article_ref": f"art_{n}",
        "score": round(rng.random(), 3),
    }
    if rng.random() < 0.3:
        hit["law_title"] = f"Synthetic act {n}"
    if rng.random() < 0.8:
        hit["language"] = rng.choice(["en", "de", "fr"])
    if rng.random() < 0.3:
        hit["consolidated"] = True
        hit["law_id"] = f"0{celex[1:]}-20990101"
        if rng.random() < 0.7:
            hit["baseCelex"] = celex
    state = rng.choice(["resolved", "unresolved", "no_flag", "wrong_date", "string_true"])
    text_words = [token(rng, "W") for _ in range(rng.randrange(1, 6))]
    snippet_words = [token(rng, "S") for _ in range(rng.randrange(0, 3))]
    if rng.random() < 0.9:
        hit["text"] = " ".join(text_words)
    if snippet_words:
        hit["snippet"] = " ".join(snippet_words)
    if state == "resolved":
        hit.update(asOf=AS_OF, asOfResolved=True)
        if rng.random() < 0.5:
            hit["asOfVersion"] = {"celex": hit["law_id"], "kind": "as_published"}
        return hit
    if state == "unresolved":
        hit.update(asOf=AS_OF, asOfResolved=False)
        reason = rng.choice(REASONS)
        if reason:
            hit["asOfReason"] = reason
    elif state == "no_flag":
        if rng.random() < 0.5:
            hit["asOf"] = AS_OF
    elif state == "wrong_date":
        hit.update(asOf="2000-01-01", asOfResolved=True)
    else:
        hit.update(asOf=AS_OF, asOfResolved="true")
    secrets.update(text_words if "text" in hit else [])
    secrets.update(snippet_words)
    return hit


def make_body(seed: int) -> tuple[dict[str, Any], set[str]]:
    rng = random.Random(seed)
    secrets: set[str] = set()
    hits = [make_hit(rng, n, secrets) for n in range(rng.randrange(0, 7))]
    if rng.random() < 0.1:
        hits.append({"article_ref": "art_0", "text": "uncited " + token(rng, "U")})
    body: dict[str, Any] = {"success": True, "results": hits}
    if rng.random() < 0.6:
        body["asOf"] = AS_OF
    if rng.random() < 0.5:
        body["credits"] = {"operation_weight": rng.randrange(5, 21), "remaining": 100}
    return body, secrets


def everything_said(docs: list[Any], report: Any, tool_content: str, artifact: Any) -> str:
    parts = [tool_content, repr(artifact), repr(dataclasses.asdict(report))]
    for doc in docs:
        parts.append(doc.page_content)
        parts.append(repr(doc.metadata))
        parts.append(str(doc.id))
    return "\n".join(parts)


def test_generator_produces_unresolved_hits() -> None:
    assert sum(bool(make_body(seed)[1]) for seed in range(CASES)) > CASES // 2


@pytest.mark.parametrize("unresolved", ["withhold", "drop"])
def test_no_unresolved_text_leaks(unresolved: str) -> None:
    checked = 0
    for seed in range(CASES):
        body, secrets = make_body(seed)
        stub = stub_for("/legislation/semantic", body)
        client, _ = offline_clients(stub)
        retriever = LexAPILegislationRetriever(client=client, as_of=AS_OF, unresolved=unresolved)
        docs, report = retriever.invoke_with_report("q")
        tool = LexAPISearchLegislationTool(client=client, unresolved=unresolved)
        message = tool.invoke(
            {
                "type": "tool_call",
                "id": "c",
                "name": tool.name,
                "args": {"query": "q", "as_of": AS_OF},
            }
        )
        said = everything_said(docs, report, str(message.content), message.artifact)
        leaked = [word for word in secrets if word in said]
        assert not leaked, f"seed {seed}: {leaked}"
        if unresolved == "drop":
            assert not any(d.metadata["text_withheld"] for d in docs)
        checked += 1
    assert checked >= 500


def test_keep_labelled_is_the_only_way_to_get_the_latest_text() -> None:
    """Control: the generated secrets do reach the output under keep_labelled,
    so the leak test above can see a leak."""
    seen = 0
    for seed in range(60):
        body, secrets = make_body(seed)
        client, _ = offline_clients(stub_for("/legislation/semantic", body))
        docs = LexAPILegislationRetriever(
            client=client, as_of=AS_OF, unresolved="keep_labelled"
        ).invoke("q")
        content = "\n".join(d.page_content for d in docs)
        seen += any(word in content for word in secrets if word.startswith("W"))
    assert seen > 0
