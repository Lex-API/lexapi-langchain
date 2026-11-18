"""Passages to LangChain documents.

All text and metadata come from :mod:`lexapi.passages`. This module adds only
LangChain's ``source`` alias, the optional cut of a long case-law text, and
de-duplication by passage id.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from langchain_core.documents import Document
from lexapi.passages import Passage, PassageResult, format_passage

#: Name of the callback event that carries a :class:`RetrievalReport`.
REPORT_EVENT = "lexapi_retrieval_report"

#: Metadata keys this package adds to :meth:`lexapi.passages.Passage.metadata`.
#: ``source`` is LangChain's conventional key for where a document came from;
#: it is the passage's ``url`` and is present only when ``url`` is.
METADATA_ALIASES = ("source",)


@dataclass(frozen=True)
class RetrievalReport:
    """What one retriever or tool call did. Immutable; one per call.

    ``withheld``, ``kept_labelled``, ``dropped``, ``uncitable`` and
    ``filtered_out`` are the :class:`lexapi.passages.PassageResult` counters
    of the call, counted by the core before this package's de-duplication,
    ``emit_withheld`` and read budget. What was actually returned is in
    ``returned``, ``returned_withheld`` and ``returned_labelled``.
    ``units_charged`` and ``credits_remaining`` are what the response
    reported (``None`` when it reported nothing); they are per call, never
    per document.
    """

    #: Hits asked for (the ``limit`` sent); ``None`` for calls without one.
    requested: int | None
    #: Documents returned.
    returned: int
    #: The date asked for, ``YYYY-MM-DD``; ``None`` when no date was asked.
    as_of: str | None = None
    #: Core count: passages whose text was withheld as not established for
    #: ``as_of`` (before de-duplication).
    withheld: int = 0
    #: Core count: passages kept with the latest indexed text
    #: (``unresolved="keep_labelled"``, before de-duplication).
    kept_labelled: int = 0
    #: Core count: hits left out by ``unresolved="drop"``.
    dropped: int = 0
    #: Core count: hits left out because the API sent no CELEX number.
    uncitable: int = 0
    #: Core count: passages removed by the ``celex`` restriction.
    filtered_out: int = 0
    #: Returned documents that are a withheld-text notice.
    returned_withheld: int = 0
    #: Returned documents that carry a labelled latest text.
    returned_labelled: int = 0
    #: Withheld passages not returned because ``emit_withheld=False``.
    withheld_not_emitted: int = 0
    #: Passages not returned because another passage with the same id ranked
    #: at least as high (see :func:`to_documents`).
    duplicates: int = 0
    #: Returned documents whose text was cut to ``max_chars``.
    truncated: int = 0
    #: Passages not returned because the read tool's ``max_chars`` budget was
    #: used up.
    not_shown: int = 0
    #: Credits the call cost, as the response reported them.
    units_charged: int | None = None
    #: Credits left, as the response reported them.
    credits_remaining: int | None = None
    #: The response's low-confidence hint, when it has one.
    hint: str | None = None


def passage_metadata(passage: Passage) -> dict[str, str | int | float | bool]:
    """``passage.metadata()`` plus the :data:`METADATA_ALIASES`."""
    metadata = passage.metadata()
    if passage.url:
        metadata["source"] = passage.url
    return metadata


def passage_to_document(passage: Passage, *, include_citation_in_content: bool = True) -> Document:
    """One passage as a ``Document``.

    ``page_content`` is :func:`lexapi.passages.format_passage` (the text or
    notice, then a ``Source:`` line), because LangChain's document prompts
    show only ``page_content``. With ``include_citation_in_content=False`` it
    is :meth:`~lexapi.passages.Passage.content` alone.
    """
    content = format_passage(passage) if include_citation_in_content else passage.content()
    return Document(page_content=content, metadata=passage_metadata(passage), id=passage.id)


def cut_text(text: str, max_chars: int) -> str | None:
    """``text`` cut to at most ``max_chars`` characters, followed by a note
    saying it was cut; ``None`` when the text fits.

    Leading whitespace is dropped first. The cut falls at the last blank
    line, else the last line break, else the last space, provided at least
    one character comes before it; otherwise it is a hard cut at
    ``max_chars``. The note's counts are characters of ``text``."""
    if len(text) <= max_chars:
        return None
    body = text.lstrip()
    if len(body) <= max_chars:
        return None
    # ``body`` starts with a non-space character, so a cut at a separator
    # after position 0 keeps at least that character.
    window = body[:max_chars]
    kept = ""
    for separator in ("\n\n", "\n", " "):
        at = window.rfind(separator)
        if at > 0:
            kept = window[:at].rstrip()
            break
    if not kept:
        kept = window.rstrip()
    return (
        f"{kept}\n[Text cut by langchain-lexapi: {len(kept)} of {len(text)} characters "
        "shown. The full text is at the source.]"
    )


def truncate_passage(passage: Passage, max_chars: int | None) -> tuple[Passage, bool]:
    """The passage with its text cut to ``max_chars``, and whether it was cut.

    Only ``text`` changes; citation, id and metadata stay those of the
    passage."""
    if max_chars is None or passage.text is None:
        return passage, False
    cut = cut_text(passage.text, max_chars)
    if cut is None:
        return passage, False
    return dataclasses.replace(passage, text=cut), True


def rank(passage: Passage) -> int:
    """De-duplication rank: a text from the API (2) over a labelled latest
    text (1) over a notice or no text (0)."""
    if passage.text_status == "provided":
        return 2
    if passage.text_status == "not_established_for_date":
        return 1
    return 0


def to_documents(
    result: PassageResult,
    *,
    requested: int | None,
    include_citation_in_content: bool = True,
    emit_withheld: bool = True,
    max_chars: int | None = None,
    budget: int | None = None,
) -> tuple[list[Document], RetrievalReport, list[Passage]]:
    """The documents of one call, in the API's order, its report, and the
    passages left out by ``budget``.

    De-duplication by passage id: a passage whose id was already seen
    replaces the earlier one only if it ranks strictly higher (:func:`rank`);
    otherwise it is left out. The document stays at the position of the
    first passage with that id.

    ``max_chars`` cuts each text (:func:`cut_text`). ``budget`` bounds the
    sum of the passages' texts and notices (``Passage.content()``, after the
    cut): passages after the first that would go over it are not returned.
    Source lines are not counted."""
    kept: dict[str, Passage] = {}
    duplicates = 0
    not_emitted = 0
    for passage in result.passages:
        if passage.text_withheld and not emit_withheld:
            not_emitted += 1
            continue
        earlier = kept.get(passage.id)
        if earlier is not None:
            duplicates += 1
            if rank(passage) <= rank(earlier):
                continue
        kept[passage.id] = passage
    shown: list[Passage] = []
    omitted: list[Passage] = []
    truncated = 0
    used = 0
    for original in kept.values():
        passage, was_cut = truncate_passage(original, max_chars)
        size = len(passage.content())
        if omitted or (budget is not None and shown and used + size > budget):
            omitted.append(original)
            continue
        shown.append(passage)
        used += size
        truncated += was_cut
    documents = [
        passage_to_document(p, include_citation_in_content=include_citation_in_content)
        for p in shown
    ]
    report = RetrievalReport(
        requested=requested,
        returned=len(documents),
        as_of=result.as_of,
        withheld=result.withheld,
        kept_labelled=result.kept_labelled,
        dropped=result.dropped,
        uncitable=result.uncitable,
        filtered_out=result.filtered_out,
        returned_withheld=sum(p.text_withheld for p in shown),
        returned_labelled=sum(p.text_status == "not_established_for_date" for p in shown),
        withheld_not_emitted=not_emitted,
        duplicates=duplicates,
        truncated=truncated,
        not_shown=len(omitted),
        units_charged=result.units_charged,
        credits_remaining=result.credits_remaining,
        hint=result.hint,
    )
    return documents, report, omitted
