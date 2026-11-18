"""LangChain tools over LexAPI: search EU legislation, read a document or one
article, and read a provision for a date.

Each tool returns ``(content, artifact)``: ``content`` is the text for the
model, built by :mod:`lexapi.passages` (texts or notices, each with a
``Source:`` citation line); ``artifact`` holds the passages' metadata and
the call's :class:`~langchain_lexapi.RetrievalReport` as plain dicts. An API
error is raised as :class:`~langchain_core.tools.ToolException` carrying the
API's error code and message.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from lexapi import CreditsExhaustedError, LexAPIError, PaymentPastDueError, RateLimitedError
from lexapi import passages as lexapi_passages
from lexapi.models import DocumentResponse, LegislationAsOfResponse
from lexapi.passages import PassageResult, UnresolvedPolicy
from pydantic import BaseModel, Field

from langchain_lexapi._clients import LexAPIClientFields
from langchain_lexapi._documents import RetrievalReport, to_documents
from langchain_lexapi.retrievers import LexAPILegislationRetriever

__all__ = [
    "LexAPILegislationAsOfTool",
    "LexAPIReadDocumentTool",
    "LexAPISearchLegislationTool",
]

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
_KEY_PATTERN = re.compile(r"lex_[A-Za-z0-9_\-]+")

Artifact = dict[str, Any]


def tool_error(error: LexAPIError) -> ToolException:
    """A ``ToolException`` naming the API's error code and message. Anything
    shaped like an API key is masked."""
    parts = [f"LexAPI error {error.code}"]
    if error.status is not None:
        parts[0] += f" (HTTP {error.status})"
    parts.append(f": {error.message}")
    if isinstance(error, CreditsExhaustedError) and error.resets_at:
        parts.append(f" Credits reset at {error.resets_at}.")
    if isinstance(error, PaymentPastDueError) and error.billing_url:
        parts.append(f" Billing: {error.billing_url}")
    if isinstance(error, RateLimitedError) and error.retry_after is not None:
        parts.append(f" Retry after {error.retry_after:g} s.")
    return ToolException(_KEY_PATTERN.sub("lex_***", "".join(parts)))


def _artifact(documents: list[Document], report: RetrievalReport) -> Artifact:
    return {
        "passages": [dict(document.metadata) for document in documents],
        "report": dataclasses.asdict(report),
    }


def _join(documents: list[Document]) -> str:
    return "\n\n".join(f"[{n}] {doc.page_content}" for n, doc in enumerate(documents, 1))


class _LexAPITool(LexAPIClientFields, BaseTool):
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"


# ── search ──────────────────────────────────────────────────────────────


class SearchLegislationInput(BaseModel):
    """Arguments of ``lexapi_search_legislation``."""

    query: str = Field(description="What to look for, in plain words.", min_length=1)
    as_of: str | None = Field(
        default=None,
        pattern=_DATE_PATTERN,
        description=(
            "Optional date, YYYY-MM-DD: return the text for that date. Leave it out for "
            "the latest text."
        ),
    )


class LexAPISearchLegislationTool(_LexAPITool):
    """Search EU legislation by meaning; one passage per matching article.

    Runs :class:`~langchain_lexapi.LexAPILegislationRetriever` with this
    tool's settings, so the same rules apply: ``k`` hits are asked for, a
    text not established for ``as_of`` is replaced by a notice, errors are
    raised. Cost: 5 credits per search; with ``as_of``, 0 to 5 credits for
    each version fetched (at most 3 per search by default).
    """

    name: str = "lexapi_search_legislation"
    description: str = (
        "Search EU legislation (EUR-Lex) by meaning. Returns matching articles, each "
        "followed by a Source line with its CELEX citation. With as_of (YYYY-MM-DD) it "
        "asks for the text for that date; an article whose text LexAPI did not establish "
        "for that date comes back as a notice without text. Consolidated texts have no "
        "legal force; the Source line names the act to cite when LexAPI knows it."
    )
    args_schema: type[BaseModel] = SearchLegislationInput

    #: Hits to ask for per search.
    k: int = Field(default=5, ge=1, le=50)
    #: Language of the texts (ISO 639-1); ``None`` is the API's default.
    language: str | None = None
    #: Similarity floor; ``None`` is the API's default floor.
    min_score: float | None = None
    #: See :attr:`LexAPILegislationRetriever.unresolved`.
    unresolved: UnresolvedPolicy = "withhold"

    def _retriever(self) -> LexAPILegislationRetriever:
        return LexAPILegislationRetriever(
            client=self._sync,
            async_client=self._async,
            k=self.k,
            language=self.language,
            min_score=self.min_score,
            unresolved=self.unresolved,
        )

    @staticmethod
    def _content(documents: list[Document], report: RetrievalReport) -> str:
        if documents:
            return _join(documents)
        note = "LexAPI found no matching legislation for this query."
        if report.dropped:
            note += f" {report.dropped} hit(s) without a text for the date were left out."
        return note

    def _run(
        self,
        query: str,
        as_of: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        config: RunnableConfig = {"callbacks": run_manager.get_child() if run_manager else None}
        overrides = {"as_of": as_of} if as_of is not None else {}
        try:
            documents, report = self._retriever().invoke_with_report(query, config, **overrides)
        except LexAPIError as error:
            raise tool_error(error) from error
        except ValueError as error:
            raise ToolException(f"Invalid argument: {error}") from error
        return self._content(documents, report), _artifact(documents, report)

    async def _arun(
        self,
        query: str,
        as_of: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        config: RunnableConfig = {"callbacks": run_manager.get_child() if run_manager else None}
        overrides = {"as_of": as_of} if as_of is not None else {}
        try:
            documents, report = await self._retriever().ainvoke_with_report(
                query, config, **overrides
            )
        except LexAPIError as error:
            raise tool_error(error) from error
        except ValueError as error:
            raise ToolException(f"Invalid argument: {error}") from error
        return self._content(documents, report), _artifact(documents, report)


# ── read a document ─────────────────────────────────────────────────────


class ReadDocumentInput(BaseModel):
    """Arguments of ``lexapi_read_document``."""

    celex: str = Field(
        description="CELEX number of the document, for example 32016R0679.", min_length=1
    )
    article: str | None = Field(
        default=None,
        description="Optional article to read: art_17, 17 or 'Article 17'. Leave it out to "
        "read the document's articles from the start.",
    )
    language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description="Optional language of the text (ISO 639-1, for example en).",
    )


class LexAPIReadDocumentTool(_LexAPITool):
    """Read a document by CELEX number: one article, or the articles from
    the start.

    Without ``article`` the output holds whole articles up to ``max_chars``
    characters and then lists the articles not shown, so the model can ask
    for one. A single text longer than ``max_chars`` is cut at a paragraph
    break, with a note. Annexes and recitals are not included. Cost: 5
    credits, settled at 1 when LexAPI already holds the document and 4 when
    it is read from the Publications Office.
    """

    name: str = "lexapi_read_document"
    description: str = (
        "Read an EU document by CELEX number: one article (article=art_17) or the "
        "document's articles from the start. Each text is followed by a Source line with "
        "its CELEX citation. Consolidated texts (CELEX starting with 0) have no legal "
        "force: cite the act as published in the Official Journal."
    )
    args_schema: type[BaseModel] = ReadDocumentInput

    #: Most characters of text to return per call; ``None`` returns all.
    max_chars: int | None = Field(default=20_000, ge=200)

    @staticmethod
    def _request(article: str | None, language: str | None) -> dict[str, Any]:
        include = ["metadata", "articles"] if article else ["metadata", "articles", "fullText"]
        return {"include": include, "article_id": article, "language": language}

    def _render(
        self, response: DocumentResponse, celex: str, article: str | None
    ) -> tuple[str, Artifact]:
        result: PassageResult = lexapi_passages.from_document(response)
        documents, report = to_documents(result, requested=None, max_chars=self.max_chars)
        if not documents:
            what = f"article {article} of CELEX {celex}" if article else f"CELEX {celex}"
            return f"LexAPI returned no text for {what}.", _artifact(documents, report)
        shown: list[Document] = []
        used = 0
        for document in documents:
            size = len(document.page_content)
            if shown and self.max_chars is not None and used + size > self.max_chars:
                break
            shown.append(document)
            used += size
        content = _join(shown)
        rest = documents[len(shown) :]
        if rest:
            refs = ", ".join(str(doc.metadata.get("article_ref", "?")) for doc in rest)
            content += (
                f"\n\n[Not shown: {len(rest)} more article(s) of CELEX {celex}: {refs}. "
                "Read one with the article argument.]"
            )
        return content, _artifact(shown, dataclasses.replace(report, returned=len(shown)))

    def _run(
        self,
        celex: str,
        article: str | None = None,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        try:
            response = self._sync_client().get_document(celex, **self._request(article, language))
        except LexAPIError as error:
            raise tool_error(error) from error
        return self._render(response, celex, article)

    async def _arun(
        self,
        celex: str,
        article: str | None = None,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        if self._async is None:
            fallback: tuple[str, Artifact] = await super()._arun(
                celex, article, language, run_manager=run_manager
            )
            return fallback
        try:
            response = await self._async.get_document(celex, **self._request(article, language))
        except LexAPIError as error:
            raise tool_error(error) from error
        return self._render(response, celex, article)


# ── a provision for a date ──────────────────────────────────────────────


class LegislationAsOfInput(BaseModel):
    """Arguments of ``lexapi_legislation_as_of``."""

    celex: str = Field(
        description="CELEX number of an EU regulation, directive or decision, for example "
        "32016R0679.",
        min_length=1,
    )
    as_of: str = Field(pattern=_DATE_PATTERN, description="The date, YYYY-MM-DD.")
    provision: str | None = Field(
        default=None,
        description="Optional article (art_17), annex (anx_III) or recitals (rct_26-28). "
        "Leave it out to get only the version for the date.",
    )
    language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}$",
        description="Optional language of the text (ISO 639-1, for example en).",
    )


class LexAPILegislationAsOfTool(_LexAPITool):
    """The version of an EU act listed for a date and, optionally, one
    provision of it.

    When LexAPI cannot establish the version or the text for the date, the
    output is a notice with the reason and no text. The answer does not say
    whether an individual provision applied on that date. Cost: 1 credit
    when no text is fetched; 4 or 5 when the provision's text is fetched (1
    if that fetch finds no text).
    """

    name: str = "lexapi_legislation_as_of"
    description: str = (
        "Get the version of an EU act listed for a date (YYYY-MM-DD) and, optionally, one "
        "provision of it (art_17, anx_III, rct_26-28). If LexAPI cannot establish the text "
        "for that date, you get a notice with the reason and no text. It does not say "
        "whether a provision applied on that date. Consolidated texts have no legal force; "
        "the Source line names the act to cite when LexAPI knows it."
    )
    args_schema: type[BaseModel] = LegislationAsOfInput

    @staticmethod
    def _render(response: LegislationAsOfResponse, provision: str | None) -> tuple[str, Artifact]:
        result = lexapi_passages.from_legislation_as_of(response, provision=provision)
        documents, report = to_documents(result, requested=None)
        if not documents:
            return "LexAPI's answer named no document.", _artifact(documents, report)
        return documents[0].page_content, _artifact(documents, report)

    def _run(
        self,
        celex: str,
        as_of: str,
        provision: str | None = None,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        try:
            response = self._sync_client().legislation_as_of(
                celex, as_of, provision=provision, language=language
            )
        except LexAPIError as error:
            raise tool_error(error) from error
        return self._render(response, provision)

    async def _arun(
        self,
        celex: str,
        as_of: str,
        provision: str | None = None,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        if self._async is None:
            fallback: tuple[str, Artifact] = await super()._arun(
                celex, as_of, provision, language, run_manager=run_manager
            )
            return fallback
        try:
            response = await self._async.legislation_as_of(
                celex, as_of, provision=provision, language=language
            )
        except LexAPIError as error:
            raise tool_error(error) from error
        return self._render(response, provision)
