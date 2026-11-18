"""De-duplication ranking, the report's returned counts, and cut_text."""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest
from lexapi.passages import Passage, PassageResult

from conftest import SYN, load, offline_clients, stub_for
from langchain_lexapi import LexAPILegislationRetriever
from langchain_lexapi._documents import cut_text, to_documents

LABELLED = "LATEST-LABELLED-TEXT"
RESOLVED = "RESOLVED-TEXT"


def same_article(state: str, text: str) -> dict[str, Any]:
    """Hits that share one passage id (same act, article, date, language)."""
    hit: dict[str, Any] = {
        "law_id": "32099R0005",
        "article_ref": "art_5",
        "language": "en",
        "score": 0.5,
        "text": SYN + text,
        "asOf": "2099-06-01",
    }
    if state == "resolved":
        hit["asOfResolved"] = True
    else:
        hit.update(asOfResolved=False, asOfReason="fetch_timeout")
    return hit


def run(hits: list[dict[str, Any]], **kwargs: Any) -> tuple[list[Any], Any]:
    body = load("legislation-semantic.mixed")
    body["results"] = [body["results"][0], *hits, body["results"][3]]
    client, _ = offline_clients(stub_for("/legislation/semantic", body))
    retriever = LexAPILegislationRetriever(client=client, as_of="2099-06-01", **kwargs)
    return retriever.invoke_with_report("q")


def test_resolved_text_replaces_a_labelled_latest_text_in_place() -> None:
    docs, report = run(
        [same_article("unresolved", LABELLED), same_article("resolved", RESOLVED)],
        unresolved="keep_labelled",
    )
    assert len(docs) == 3
    middle = docs[1]
    assert middle.page_content.startswith(SYN + RESOLVED)
    assert middle.metadata["text_status"] == "provided"
    assert report.duplicates == 1
    assert (report.kept_labelled, report.returned_labelled) == (1, 0)


def test_a_later_labelled_text_does_not_replace_a_resolved_text() -> None:
    docs, report = run(
        [same_article("resolved", RESOLVED), same_article("unresolved", LABELLED)],
        unresolved="keep_labelled",
    )
    assert docs[1].page_content.startswith(SYN + RESOLVED)
    assert LABELLED not in "".join(d.page_content for d in docs)
    assert report.duplicates == 1


def test_resolved_text_replaces_a_notice_in_place() -> None:
    docs, report = run([same_article("unresolved", LABELLED), same_article("resolved", RESOLVED)])
    assert docs[1].page_content.startswith(SYN + RESOLVED)
    assert (report.withheld, report.returned_withheld) == (1, 0)
    assert LABELLED not in "".join(d.page_content for d in docs)


def test_equal_rank_keeps_the_first() -> None:
    first = same_article("resolved", RESOLVED)
    second = same_article("resolved", "SECOND-TEXT")
    second["score"] = 0.99
    docs, report = run([first, second])
    assert docs[1].page_content.startswith(SYN + RESOLVED)
    assert docs[1].metadata["score"] == 0.5
    assert report.duplicates == 1


def test_returned_withheld_counts_the_notices_returned() -> None:
    _, report = LexAPILegislationRetriever(
        client=offline_clients(
            stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
        )[0],
        as_of="2099-06-01",
    ).invoke_with_report("q")
    assert (report.withheld, report.returned_withheld, report.withheld_not_emitted) == (1, 1, 0)
    client, _ = offline_clients(
        stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    )
    _, report = LexAPILegislationRetriever(
        client=client, as_of="2099-06-01", emit_withheld=False
    ).invoke_with_report("q")
    assert (report.withheld, report.returned_withheld, report.withheld_not_emitted) == (1, 0, 1)


def test_returned_labelled_counts_the_labelled_texts_returned() -> None:
    client, _ = offline_clients(
        stub_for("/legislation/semantic", load("legislation-semantic.mixed"))
    )
    _, report = LexAPILegislationRetriever(
        client=client, as_of="2099-06-01", unresolved="keep_labelled"
    ).invoke_with_report("q")
    assert (report.kept_labelled, report.returned_labelled, report.returned_withheld) == (1, 1, 0)


# ── cut_text ────────────────────────────────────────────────────────────

NOTE = "\n[Text cut by langchain-lexapi: "


@pytest.mark.parametrize(
    "text",
    [
        "\n\n\n" + "x" * 300,
        " " * 50 + "y" * 300,
        "\n\n" + "a" * 250 + "\n\n" + "b" * 100,
        "a\n\n" + "b" * 400,
        " \n \n " + "c" * 500,
    ],
)
def test_cut_text_never_shows_nothing(text: str) -> None:
    cut, was_cut = cut_text(text, 200)
    assert was_cut
    kept, note = cut.split(NOTE, 1)
    assert kept.strip()
    assert len(kept) <= 200
    assert text.lstrip().startswith(kept)
    assert note == (
        f"{len(kept)} of {len(text)} characters shown. The full text is at the source.]"
    )


def test_a_separator_right_at_the_limit_is_used() -> None:
    # The space at index 10 (just past the 10 kept characters) is used, not
    # the earlier one at index 4.
    cut, was_cut = cut_text("aaaa bbbbb cccc", 10)
    assert was_cut and cut.startswith("aaaa bbbbb" + NOTE)
    assert cut_text("aaaa\nbbbbb\ncccc", 10)[0].startswith("aaaa\nbbbbb" + NOTE)


def test_leading_whitespace_is_stripped_when_the_rest_fits() -> None:
    text = "\n\n   " + "x" * 100
    assert cut_text(text, 100) == ("x" * 100, False)
    assert cut_text("x" * 100, 100) == ("x" * 100, False)


def test_cut_text_property() -> None:
    rng = random.Random(7)
    alphabet = ["a", "b", " ", "\n", "\n\n", "\t"]
    for _ in range(3000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 400)))
        limit = rng.randrange(1, 300)
        cut, was_cut = cut_text(text, limit)
        if not was_cut:
            assert NOTE not in cut
            assert cut in (text, text.lstrip())
            assert len(cut) <= limit
            continue
        kept, note = cut.split(NOTE, 1)
        assert kept.strip(), repr(text)
        assert 1 <= len(kept) <= limit
        assert text.lstrip().startswith(kept)
        assert note.startswith(f"{len(kept)} of {len(text)} characters shown.")


def passage(status: str, text: str | None) -> Passage:
    return Passage(
        kind="legislation",
        celex="32099R0005",
        text=text,
        text_status=status,  # type: ignore[arg-type]
        article_ref="art_5",
        as_of="2099-06-01",
    )


@pytest.mark.parametrize(
    ("first", "second", "winner"),
    [
        (("withheld", None), ("not_established_for_date", "L"), "L"),
        (("not_established_for_date", "L"), ("withheld", None), "L"),
        (("absent", None), ("not_established_for_date", "L"), "L"),
        (("not_established_for_date", "L"), ("provided", "P"), "P"),
        (("provided", "P"), ("not_established_for_date", "L"), "P"),
        (("withheld", None), ("provided", "P"), "P"),
        (("provided", "P"), ("provided", "Q"), "P"),
        (("withheld", None), ("absent", None), "[No text: "),
    ],
)
def test_rank_order(
    first: tuple[str, str | None], second: tuple[str, str | None], winner: str
) -> None:
    other = dataclasses.replace(passage("provided", "OTHER"), article_ref="art_9")
    result = PassageResult(passages=(passage(*first), other, passage(*second)))
    docs, report, _ = to_documents(result, requested=None, include_citation_in_content=False)
    assert len(docs) == 2
    assert docs[0].page_content.startswith(winner) or docs[0].page_content.endswith(winner)
    assert docs[1].page_content == "OTHER"
    assert report.duplicates == 1
