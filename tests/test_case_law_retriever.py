"""LexAPICaseLawRetriever against synthetic responses."""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from lexapi import passages
from lexapi.models import SemanticSearchResponse
from pydantic import ValidationError

from conftest import SYN, Stub, load, offline_clients, stub_for
from langchain_lexapi import LexAPICaseLawRetriever
from langchain_lexapi._documents import cut_text

PATH = "/search/semantic"


def retriever(name: str, **kwargs: Any) -> tuple[LexAPICaseLawRetriever, Stub]:
    stub = stub_for(PATH, load(name))
    client, async_client = offline_clients(stub)
    return LexAPICaseLawRetriever(client=client, async_client=async_client, **kwargs), stub


def test_request_always_asks_for_text_and_sends_filters() -> None:
    r, stub = retriever(
        "case-law.with-text",
        k=3,
        court="Court of Justice",
        date_from=datetime.date(2099, 1, 1),
        date_to="2099-12-31",
        language="fr",
    )
    r.invoke("q")
    assert stub.bodies() == [
        {
            "query": "q",
            "limit": 3,
            "include": ["text"],
            "language": "fr",
            "filters": {
                "court": "Court of Justice",
                "date_from": "2099-01-01",
                "date_to": "2099-12-31",
            },
        }
    ]
    assert "hyde" not in stub.bodies()[0]


def test_no_filters_sends_none() -> None:
    r, stub = retriever("case-law.with-text")
    r.invoke("q", k=2)
    assert stub.bodies() == [{"query": "q", "limit": 2, "include": ["text"]}]


def test_documents_carry_the_decision_text_and_citation() -> None:
    r, _ = retriever("case-law.with-text")
    docs = r.invoke("q")
    expected = passages.from_case_law_search(SemanticSearchResponse(load("case-law.with-text")))
    assert [d.page_content for d in docs] == [passages.format_passage(p) for p in expected.passages]
    first = docs[0]
    assert first.page_content.startswith(SYN + "Judgment.")
    assert "ECLI:EU:C:2099:1, para. 93" in first.page_content
    assert first.metadata["kind"] == "case_law"
    assert first.metadata["paragraph_number"] == 93
    assert first.id == "lexapi:62099CJ9901:93::en"


def test_snippet_and_summary_never_stand_in_for_text() -> None:
    for name in ("case-law.with-text", "case-law.without-text"):
        r, _ = retriever(name)
        for doc in r.invoke("q"):
            blob = doc.page_content + repr(doc.metadata)
            assert "SNIPPET-MUST-NOT-APPEAR" not in blob
            assert "SUMMARY-MUST-NOT-APPEAR" not in blob


def test_hit_without_text_is_a_notice() -> None:
    r, _ = retriever("case-law.without-text")
    docs = r.invoke("q")
    assert docs[0].page_content.startswith("[No text returned for CELEX 62099CJ9901.]")
    assert docs[0].metadata["text_status"] == "absent"


def test_max_chars_is_off_by_default() -> None:
    body = load("case-law.with-text")
    long_text = SYN + "\n\n".join(f"{n} " + "y" * 500 for n in range(1, 200))
    body["results"][0]["text"] = long_text
    client, _ = offline_clients(stub_for(PATH, body))
    r = LexAPICaseLawRetriever(client=client)
    assert r.max_chars is None
    docs, report = r.invoke_with_report("q")
    assert report.truncated == 0
    assert docs[0].page_content.startswith(long_text + "\nSource: ")
    assert "Text cut" not in docs[0].page_content


def test_max_chars_cuts_at_a_paragraph_break_and_says_so() -> None:
    body = load("case-law.with-text")
    long_text = SYN + "\n\n".join(f"{n} " + "x" * 150 for n in range(1, 30))
    body["results"][0]["text"] = long_text
    stub = stub_for(PATH, body)
    client, _ = offline_clients(stub)
    r = LexAPICaseLawRetriever(client=client, max_chars=1000)
    docs, report = r.invoke_with_report("q")
    text = docs[0].page_content
    shown, rest = text.split("\n[Text cut by langchain-lexapi: ", 1)
    assert len(shown) <= 1000
    assert long_text.startswith(shown)
    assert long_text[len(shown) : len(shown) + 2] == "\n\n"
    assert rest.startswith(f"{len(shown)} of {len(long_text)} characters shown.")
    assert text.endswith(docs[0].metadata["citation"])
    assert report.truncated == 1
    assert docs[1].page_content.startswith(SYN + "Order of the Court.")


def test_cut_text_falls_back_to_line_then_space_then_hard_cut() -> None:
    assert cut_text("short", 10) is None
    line = cut_text("aaaa\nbbbb\ncccc", 12)
    assert line is not None and line.startswith("aaaa\nbbbb\n[Text cut")
    word = cut_text("aaaa bbbb cccc", 12)
    assert word is not None and word.startswith("aaaa bbbb\n[Text cut")
    hard = cut_text("a" * 30, 12)
    assert hard is not None and hard.startswith("a" * 12 + "\n[Text cut")


def test_max_chars_floor() -> None:
    with pytest.raises(ValidationError):
        retriever("case-law.with-text", max_chars=10)


def test_k_up_to_100() -> None:
    r, stub = retriever("case-law.with-text", k=100)
    r.invoke("q")
    assert stub.bodies()[0]["limit"] == 100
    with pytest.raises(ValidationError):
        retriever("case-law.with-text", k=101)
    with pytest.raises(TypeError):
        r.invoke("q", as_of="2099-01-01")


def test_duplicate_decision_hits_collapse() -> None:
    body = load("case-law.with-text")
    body["results"].append(dict(body["results"][0], score=0.5))
    stub = stub_for(PATH, body)
    client, _ = offline_clients(stub)
    docs, report = LexAPICaseLawRetriever(client=client).invoke_with_report("q")
    assert len(docs) == 2
    assert report.duplicates == 1
    assert docs[0].metadata["score"] == 0.83
