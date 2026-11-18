"""The Document mapping: content, metadata and id come from lexapi.passages;
the only addition is the documented ``source`` alias. Sync equals async."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from lexapi import passages
from lexapi.models import SemanticLegislationResponse, SemanticSearchResponse

from conftest import load, offline_clients, stub_for
from langchain_lexapi import (
    METADATA_ALIASES,
    LexAPICaseLawRetriever,
    LexAPILegislationRetriever,
)

LEGISLATION = [
    ("legislation-semantic.as-published", "2099-06-01"),
    ("legislation-semantic.consolidated", "2099-06-01"),
    ("legislation-semantic.unresolved", "2099-06-01"),
    ("legislation-semantic.mixed", "2099-06-01"),
    ("legislation-semantic.no-date", None),
    ("legislation-semantic.empty", None),
]
POLICIES = ["withhold", "drop", "keep_labelled"]


def expected_metadata(p: passages.Passage) -> dict[str, Any]:
    metadata: dict[str, Any] = p.metadata()
    if p.url:
        metadata["source"] = p.url
    return metadata


def legislation_docs(
    name: str, as_of: str | None, **kwargs: Any
) -> tuple[list[Document], passages.PassageResult]:
    stub = stub_for("/legislation/semantic", load(name))
    client, _ = offline_clients(stub)
    docs = LexAPILegislationRetriever(client=client, as_of=as_of, **kwargs).invoke("q")
    result = passages.from_legislation_search(
        SemanticLegislationResponse(load(name)),
        as_of=as_of,
        unresolved=kwargs.get("unresolved", "withhold"),
    )
    return docs, result


def test_aliases_are_documented() -> None:
    assert METADATA_ALIASES == ("source",)


@pytest.mark.parametrize(("name", "as_of"), LEGISLATION)
@pytest.mark.parametrize("unresolved", POLICIES)
def test_legislation_documents_match_the_core(
    name: str, as_of: str | None, unresolved: str
) -> None:
    docs, result = legislation_docs(name, as_of, unresolved=unresolved)
    assert len(docs) == len(result.passages)
    for doc, p in zip(docs, result.passages, strict=True):
        assert doc.page_content == passages.format_passage(p)
        assert doc.metadata == expected_metadata(p)
        assert set(doc.metadata) - set(p.metadata()) <= set(METADATA_ALIASES)
        assert doc.id == p.id == doc.metadata["id"]
        for value in doc.metadata.values():
            assert type(value) in (str, int, float, bool)


@pytest.mark.parametrize("name", ["case-law.with-text", "case-law.without-text"])
def test_case_law_documents_match_the_core(name: str) -> None:
    stub = stub_for("/search/semantic", load(name))
    client, _ = offline_clients(stub)
    docs = LexAPICaseLawRetriever(client=client).invoke("q")
    result = passages.from_case_law_search(SemanticSearchResponse(load(name)))
    for doc, p in zip(docs, result.passages, strict=True):
        assert doc.page_content == passages.format_passage(p)
        assert doc.metadata == expected_metadata(p)
        assert doc.id == p.id


def test_metadata_never_holds_text_or_snippet() -> None:
    body = load("legislation-semantic.mixed")
    texts = [hit["text"] for hit in body["results"]] + [hit["snippet"] for hit in body["results"]]
    docs, _ = legislation_docs(
        "legislation-semantic.mixed", "2099-06-01", unresolved="keep_labelled"
    )
    for doc in docs:
        for value in doc.metadata.values():
            assert value not in texts


def test_ids_are_stable_across_calls_and_differ_between_dates() -> None:
    first, _ = legislation_docs("legislation-semantic.mixed", "2099-06-01")
    second, _ = legislation_docs("legislation-semantic.mixed", "2099-06-01")
    assert [d.id for d in first] == [d.id for d in second]
    assert first[0].id == "lexapi:02099R0001-20990101:art_17:2099-06-01:en"
    undated, _ = legislation_docs("legislation-semantic.no-date", None)
    assert undated[0].id == "lexapi:02099R0001-20990101:art_17::en"


SYNC_ASYNC = [
    (LexAPILegislationRetriever, "/legislation/semantic", name, {"as_of": as_of})
    for name, as_of in LEGISLATION
] + [
    (LexAPICaseLawRetriever, "/search/semantic", "case-law.with-text", {"max_chars": 200}),
    (LexAPICaseLawRetriever, "/search/semantic", "case-law.without-text", {}),
]


@pytest.mark.parametrize(("cls", "path", "name", "kwargs"), SYNC_ASYNC)
async def test_sync_equals_async(cls: type[Any], path: str, name: str, kwargs: Any) -> None:
    stub = stub_for(path, load(name))
    client, async_client = offline_clients(stub)
    r = cls(client=client, async_client=async_client, **kwargs)
    sync_docs, sync_report = r.invoke_with_report("q")
    async_docs, async_report = await r.ainvoke_with_report("q")
    assert sync_docs == async_docs
    assert [d.id for d in sync_docs] == [d.id for d in async_docs]
    assert sync_report == async_report
    first, second = stub.bodies()
    assert first == second
