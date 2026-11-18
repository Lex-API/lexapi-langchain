"""LexAPILegislationRetriever against synthetic responses."""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from lexapi import passages
from lexapi.models import SemanticLegislationResponse
from pydantic import ValidationError

from conftest import SYN, Stub, load, offline_clients, stub_for
from langchain_lexapi import LexAPILegislationRetriever

PATH = "/legislation/semantic"


def retriever(name: str, **kwargs: Any) -> tuple[LexAPILegislationRetriever, Stub]:
    stub = stub_for(PATH, load(name))
    client, async_client = offline_clients(stub)
    return LexAPILegislationRetriever(client=client, async_client=async_client, **kwargs), stub


def core(name: str, **kwargs: Any) -> passages.PassageResult:
    return passages.from_legislation_search(SemanticLegislationResponse(load(name)), **kwargs)


# ── request ─────────────────────────────────────────────────────────────


def test_request_body_without_date() -> None:
    r, stub = retriever("legislation-semantic.no-date")
    r.invoke("right to erasure")
    assert stub.bodies() == [{"query": "right to erasure", "limit": 5, "include": ["text"]}]
    assert stub.requests[0].headers["x-api-key"]


def test_request_body_with_date_language_and_floor() -> None:
    r, stub = retriever(
        "legislation-semantic.mixed", k=7, as_of="2099-06-01", language="de", min_score=0.4
    )
    r.invoke("q")
    assert stub.bodies() == [
        {
            "query": "q",
            "limit": 7,
            "include": ["text"],
            "language": "de",
            "min_score": 0.4,
            "asOf": "2099-06-01",
        }
    ]


def test_per_call_overrides_for_k_and_as_of() -> None:
    r, stub = retriever("legislation-semantic.mixed", as_of="2000-01-01")
    r.invoke("q", k=2, as_of=datetime.date(2099, 6, 1))
    assert stub.bodies()[0]["limit"] == 2
    assert stub.bodies()[0]["asOf"] == "2099-06-01"


@pytest.mark.parametrize("k", [0, 51, -1, True, "3", 2.0])
def test_bad_k_override_is_refused_before_any_request(k: Any) -> None:
    r, stub = retriever("legislation-semantic.mixed")
    with pytest.raises(ValueError, match="k must be"):
        r.invoke("q", k=k)
    assert stub.requests == []


@pytest.mark.parametrize("k", [0, 51])
def test_bad_k_in_constructor(k: int) -> None:
    with pytest.raises(ValidationError):
        retriever("legislation-semantic.mixed", k=k)


def test_unknown_override_is_a_type_error() -> None:
    r, stub = retriever("legislation-semantic.mixed")
    with pytest.raises(TypeError, match="unexpected keyword"):
        r.invoke("q", top_k=3)
    assert stub.requests == []


def test_bad_as_of_is_refused_before_any_request() -> None:
    r, stub = retriever("legislation-semantic.mixed")
    with pytest.raises(ValidationError):
        r.invoke("q", as_of="2099-02-30")
    assert stub.requests == []


def test_k_is_never_raised_to_fill_the_result() -> None:
    """Withheld, dropped and filtered hits do not make the retriever ask for more."""
    r, stub = retriever(
        "legislation-semantic.mixed",
        k=4,
        as_of="2099-06-01",
        unresolved="drop",
        celex="32099R0001",
    )
    docs = r.invoke("q")
    assert len(docs) == 1
    assert len(stub.requests) == 1
    assert stub.bodies()[0]["limit"] == 4


# ── content ─────────────────────────────────────────────────────────────


def test_page_content_is_format_passage_by_default() -> None:
    r, _ = retriever("legislation-semantic.mixed", as_of="2099-06-01")
    docs = r.invoke("q")
    expected = core("legislation-semantic.mixed", as_of="2099-06-01")
    assert [d.page_content for d in docs] == [passages.format_passage(p) for p in expected.passages]
    assert all("\nSource: CELEX " in d.page_content for d in docs)


def test_include_citation_in_content_false_uses_content() -> None:
    r, _ = retriever(
        "legislation-semantic.mixed", as_of="2099-06-01", include_citation_in_content=False
    )
    docs = r.invoke("q")
    expected = core("legislation-semantic.mixed", as_of="2099-06-01")
    assert [d.page_content for d in docs] == [p.content() for p in expected.passages]
    assert all("Source:" not in d.page_content for d in docs)


def test_consolidated_hit_names_the_act_to_cite() -> None:
    r, _ = retriever("legislation-semantic.consolidated", as_of="2099-06-01")
    (doc,) = r.invoke("q")
    m = doc.metadata
    assert m["consolidated"] is True
    assert m["celex"] == "02099R0001-20990101"
    assert m["cite_celex"] == "32099R0001"
    assert "not the authentic text" in m["consolidated_note"]
    assert "consolidated text 02099R0001-20990101 (no legal force)" in doc.page_content
    assert m["source"] == m["url"]


def test_no_date_asked_means_no_date_keys_or_wording() -> None:
    r, _ = retriever("legislation-semantic.no-date")
    docs = r.invoke("q")
    assert len(docs) == 2
    for doc in docs:
        assert not [key for key in doc.metadata if key.startswith("as_of")]
        assert "established" not in doc.page_content
        assert "in force" not in doc.page_content


def test_resolved_hit_has_text_and_version() -> None:
    r, _ = retriever("legislation-semantic.as-published", as_of="2099-06-01")
    (doc,) = r.invoke("q")
    assert doc.page_content.startswith(SYN + "Article 5 as published.")
    assert doc.metadata["as_of_resolved"] is True
    assert doc.metadata["version_kind"] == "as_published"
    assert "version for 2099-06-01: 32099R0002" in doc.page_content


def test_unresolved_hit_is_a_notice_by_default() -> None:
    r, _ = retriever("legislation-semantic.unresolved", as_of="2099-06-01")
    (doc,) = r.invoke("q")
    assert doc.page_content.startswith("[No text: CELEX 32099L0003 art_12 is not established")
    assert "fetch_timeout" in doc.page_content
    assert "MUST-NOT-LEAK" not in doc.page_content
    assert "MUST-NOT-LEAK" not in repr(doc.metadata)
    assert doc.metadata["text_withheld"] is True
    assert doc.metadata["text_status"] == "withheld"
    assert doc.metadata["score"] == 0.75


def test_as_of_is_passed_to_the_core_even_if_the_hit_does_not_echo_it() -> None:
    """A hit that says resolved but without echoing a date is not trusted
    for the date that was asked."""
    body = load("legislation-semantic.as-published")
    for hit in body["results"]:
        hit.pop("asOf")
    body.pop("asOf")
    stub = stub_for(PATH, body)
    client, _ = offline_clients(stub)
    r = LexAPILegislationRetriever(client=client, as_of="2099-06-01")
    (doc,) = r.invoke("q")
    assert doc.metadata["text_withheld"] is True
    assert "Article 5 as published" not in doc.page_content


def test_drop_policy() -> None:
    r, _ = retriever("legislation-semantic.mixed", as_of="2099-06-01", unresolved="drop")
    docs, report = r.invoke_with_report("q")
    assert len(docs) == 3
    assert report.dropped == 1
    assert all("MUST-NOT-LEAK" not in d.page_content for d in docs)


def test_keep_labelled_policy_labels_the_latest_text() -> None:
    r, _ = retriever(
        "legislation-semantic.unresolved", as_of="2099-06-01", unresolved="keep_labelled"
    )
    (doc,) = r.invoke("q")
    assert doc.page_content.startswith("[Latest indexed text of CELEX 32099L0003 art_12; not est")
    assert "UNRESOLVED-LATEST-TEXT-MUST-NOT-LEAK" in doc.page_content
    assert doc.metadata["text_status"] == "not_established_for_date"


def test_bad_unresolved_policy_is_refused() -> None:
    with pytest.raises(ValidationError):
        retriever("legislation-semantic.mixed", unresolved="latest_labelled")


def test_emit_withheld_false_leaves_notices_out_and_counts_them() -> None:
    r, _ = retriever("legislation-semantic.mixed", as_of="2099-06-01", emit_withheld=False)
    docs, report = r.invoke_with_report("q")
    assert len(docs) == 3
    assert not any(d.metadata["text_withheld"] for d in docs)
    assert report.withheld == 1
    assert report.withheld_not_emitted == 1


def test_celex_restriction_keeps_the_act_and_its_consolidated_texts() -> None:
    r, stub = retriever("legislation-semantic.mixed", as_of="2099-06-01", celex="CELEX:32099R0001")
    docs, report = r.invoke_with_report("q")
    assert [d.metadata["celex"] for d in docs] == ["02099R0001-20990101"]
    assert report.filtered_out == 3
    assert stub.bodies()[0]["limit"] == 5


def test_celex_restriction_with_several_acts() -> None:
    r, _ = retriever(
        "legislation-semantic.mixed", as_of="2099-06-01", celex=["32099R0002", "32099R0009"]
    )
    docs = r.invoke("q")
    assert [d.metadata["celex"] for d in docs] == ["32099R0002", "32099R0009"]


@pytest.mark.parametrize("celex", ["", [], ["  "]])
def test_empty_celex_restriction_is_refused(celex: Any) -> None:
    with pytest.raises(ValidationError):
        retriever("legislation-semantic.mixed", celex=celex)


def test_empty_response_is_an_empty_list() -> None:
    r, _ = retriever("legislation-semantic.empty")
    docs, report = r.invoke_with_report("q")
    assert docs == []
    assert report.returned == 0
    assert report.units_charged == 5


def test_duplicate_ids_are_collapsed_text_wins_over_notice() -> None:
    body = load("legislation-semantic.mixed")
    unresolved = dict(body["results"][1])
    unresolved.update({"asOfResolved": False, "asOfReason": "fetch_timeout"})
    unresolved.pop("asOfVersion")
    body["results"].insert(1, unresolved)  # same id as the resolved hit after it
    stub = stub_for(PATH, body)
    client, _ = offline_clients(stub)
    r = LexAPILegislationRetriever(client=client, as_of="2099-06-01")
    docs, report = r.invoke_with_report("q")
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)) == 4
    second = docs[1]
    assert second.metadata["text_withheld"] is False
    assert report.duplicates == 1


def test_scores_pass_through_unchanged() -> None:
    r, _ = retriever("legislation-semantic.mixed", as_of="2099-06-01")
    raw = [hit["score"] for hit in load("legislation-semantic.mixed")["results"]]
    assert [d.metadata["score"] for d in r.invoke("q")] == raw


def test_report_counts_and_credits() -> None:
    r, _ = retriever("legislation-semantic.mixed", as_of="2099-06-01")
    docs, report = r.invoke_with_report("q")
    assert report.requested == 5
    assert report.returned == len(docs) == 4
    assert report.withheld == 1
    assert report.as_of == "2099-06-01"
    assert (report.units_charged, report.credits_remaining) == (5, 995)


def test_only_an_async_client_refuses_sync_calls() -> None:
    stub = stub_for(PATH, load("legislation-semantic.mixed"))
    _, async_client = offline_clients(stub)
    r = LexAPILegislationRetriever(async_client=async_client)
    with pytest.raises(ValueError, match="only an async_client"):
        r.invoke("q")


async def test_only_a_sync_client_serves_async_calls_in_an_executor() -> None:
    stub = stub_for(PATH, load("legislation-semantic.mixed"))
    client, _ = offline_clients(stub)
    r = LexAPILegislationRetriever(client=client, as_of="2099-06-01")
    docs = await r.ainvoke("q")
    assert len(docs) == 4
    assert len(stub.requests) == 1
