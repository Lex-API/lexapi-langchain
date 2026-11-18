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

    The counters ``withheld``, ``kept_labelled``, ``dropped``, ``uncitable``
    and ``filtered_out`` are the :class:`lexapi.passages.PassageResult`
    counters of the call. ``units_charged`` and ``credits_remaining`` are
    what the response reported (``None`` when it reported nothing); they are
    per call, never per document.
    """

    #: Hits asked for (the ``limit`` sent); ``None`` for calls without one.
    requested: int | None
    #: Documents returned.
    returned: int
    #: The date asked for, ``YYYY-MM-DD``; ``None`` when no date was asked.
    as_of: str | None = None
    #: Passages without text because the text is not established for ``as_of``.
    withheld: int = 0
    #: Passages kept with the latest indexed text (``unresolved="keep_labelled"``).
    kept_labelled: int = 0
    #: Hits left out by ``unresolved="drop"``.
    dropped: int = 0
    #: Hits left out because the API sent no CELEX number for them.
    uncitable: int = 0
    #: Passages removed by the ``celex`` restriction.
    filtered_out: int = 0
    #: Withheld passages not returned because ``emit_withheld=False``.
    withheld_not_emitted: int = 0
    #: Passages not returned because an earlier one had the same id.
    duplicates: int = 0
    #: Documents whose text was cut to ``max_chars``.
    truncated: int = 0
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
    """``text`` cut to at most ``max_chars`` characters at the last paragraph
    break (a blank line, else a line break, else a space), followed by a
    note saying it was cut. ``None`` when the text fits."""
    if len(text) <= max_chars:
        return None
    window = text[:max_chars]
    for separator in ("\n\n", "\n", " "):
        at = window.rfind(separator)
        if at > 0:
            window = window[:at]
            break
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


def to_documents(
    result: PassageResult,
    *,
    requested: int | None,
    include_citation_in_content: bool = True,
    emit_withheld: bool = True,
    max_chars: int | None = None,
) -> tuple[list[Document], RetrievalReport]:
    """The documents of one call, in the API's order, and its report.

    A passage whose id was already seen is left out, except that a passage
    with text replaces an earlier one without text, so a notice never stands
    in for a text."""
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
            if earlier.has_text or not passage.has_text:
                continue
        kept[passage.id] = passage
    documents: list[Document] = []
    truncated = 0
    for passage in kept.values():
        passage, was_cut = truncate_passage(passage, max_chars)
        truncated += was_cut
        documents.append(
            passage_to_document(passage, include_citation_in_content=include_citation_in_content)
        )
    report = RetrievalReport(
        requested=requested,
        returned=len(documents),
        as_of=result.as_of,
        withheld=result.withheld,
        kept_labelled=result.kept_labelled,
        dropped=result.dropped,
        uncitable=result.uncitable,
        filtered_out=result.filtered_out,
        withheld_not_emitted=not_emitted,
        duplicates=duplicates,
        truncated=truncated,
        units_charged=result.units_charged,
        credits_remaining=result.credits_remaining,
        hint=result.hint,
    )
    return documents, report
