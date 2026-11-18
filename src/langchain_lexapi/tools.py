"""LangChain tools over LexAPI: search EU legislation, read a document or one
article, and read a provision for a date.

Each tool returns ``(content, artifact)``: ``content`` is the text for the
model, built by :mod:`lexapi.passages` (texts or notices, each with a
``Source:`` citation line); ``artifact`` holds the passages' metadata and
the call's :class:`~langchain_lexapi.RetrievalReport` as plain dicts. An API
error is raised as :class:`~langchain_core.tools.ToolException` carrying the
API's error code and message; an answer that could not be read gets a fixed
message.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any, Literal, NoReturn, Self

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from lexapi import (
    AsyncLexAPI,
    CreditsExhaustedError,
    InvalidResponseError,
    LexAPIError,
    PaymentPastDueError,
)
from lexapi import passages as lexapi_passages
from lexapi.models import DocumentResponse, LegislationAsOfResponse
from lexapi.passages import PassageResult, UnresolvedPolicy
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from langchain_lexapi._clients import ClientClosedError, LexAPIClientFields
from langchain_lexapi._documents import RetrievalReport, to_documents
from langchain_lexapi.retrievers import (
    InvalidArgumentError,
    LexAPILegislationRetriever,
    _date_argument,
)

__all__ = [
    "LexAPILegislationAsOfTool",
    "LexAPIReadDocumentTool",
    "LexAPISearchLegislationTool",
]

_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
_KEY_PATTERN = re.compile(r"lex_[A-Za-z0-9_\-]+")

Artifact = dict[str, Any]


def _unreadable(error: LexAPIError) -> bool:
    """An answer that came back but could not be read."""
    status = error.status
    return isinstance(error, InvalidResponseError) or (status is not None and 200 <= status < 300)


def tool_error(error: LexAPIError) -> ToolException:
    """A ``ToolException`` naming the API's error code and message. Anything
    shaped like an API key is masked.

    An answer that could not be read (``InvalidResponseError``, or any error
    on a 2xx status) gets a fixed message, so what the model reads does not
    depend on the SDK's message for it."""
    status = error.status
    if _unreadable(error):
        where = f" (HTTP {status})" if status is not None else ""
        return ToolException(f"LexAPI error INVALID_RESPONSE{where}: The answer could not be read.")
    parts = [f"LexAPI error {error.code}"]
    if status is not None:
        parts[0] += f" (HTTP {status})"
    message = error.message.strip()
    if message and message[-1] not in ".!?":
        message += "."
    parts.append(f": {message}")
    if isinstance(error, CreditsExhaustedError) and error.resets_at:
        parts.append(f" Credits reset at {error.resets_at}.")
    if isinstance(error, PaymentPastDueError) and error.billing_url:
        parts.append(f" Billing: {error.billing_url}")
    if error.retry_after is not None:
        parts.append(f" Retry after {error.retry_after:g} s.")
    return ToolException(_KEY_PATTERN.sub("lex_***", "".join(parts)))


def _check_celex(celex: str) -> None:
    if not isinstance(celex, str) or not celex.strip():
        raise InvalidArgumentError("celex must not be empty")


def _unreadable_answer() -> InvalidResponseError:
    """For a 2xx answer that LexAPI's client accepted but that fails one of
    this package's additional checks. The status is not known at this
    point, so the fixed text names none."""
    return InvalidResponseError("The answer does not have the documented shape.")


def _is_object_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _document_answer_readable(response: DocumentResponse) -> bool:
    """A document read must carry a ``document`` object with a non-empty
    ``celex`` string; its ``content``, when present and not null, an object;
    ``content.fullText``, when present, a string; and ``content.articles``,
    when present, a list of objects. LexAPI's client checks the documented
    fields itself; these checks are additional (it accepts ``document: {}``,
    for example)."""
    document = response.get("document")
    if not isinstance(document, dict):
        return False
    celex = document.get("celex")
    if not isinstance(celex, str) or not celex.strip():
        return False
    content = document.get("content")
    if content is None:
        return True
    if not isinstance(content, dict):
        return False
    full_text = content.get("fullText")
    if full_text is not None and not isinstance(full_text, str):
        return False
    articles = content.get("articles")
    return articles is None or _is_object_list(articles)


def _as_of_answer_readable(response: LegislationAsOfResponse) -> bool:
    """An as-of answer must name its document (a non-blank CELEX string)
    and say whether it is resolved (a boolean). LexAPI's client checks the
    documented fields itself; these checks are additional (it accepts a
    whitespace-only ``document``, for example)."""
    document = response.get("document")
    return (
        isinstance(document, str)
        and bool(document.strip())
        and isinstance(response.get("asOfResolved"), bool)
    )


def _cause(error: Exception) -> Exception | None:
    """The exception to chain: none for an answer that could not be read, so
    the tool's error carries only its fixed message."""
    if isinstance(error, LexAPIError) and _unreadable(error):
        return None
    return error


def _fail(error: Exception) -> NoReturn:
    """Raise the tool's ``ToolException`` for ``error``; any other error is
    raised as itself. Called outside the ``except`` block, so an unreadable
    answer's SDK error is neither the cause nor the context of what is
    raised."""
    if not isinstance(error, _TOOL_ERRORS):
        raise error  # not a tool error (a RuntimeError that did not race close)
    failure = _raise_for(error)
    cause = _cause(error)
    if cause is None:
        raise failure
    raise failure from cause


def _raise_for(error: Exception) -> ToolException:
    if isinstance(error, LexAPIError):
        return tool_error(error)
    if isinstance(error, InvalidArgumentError):
        return ToolException(f"Invalid argument: {error}")
    return ToolException(str(error))  # ClientClosedError


#: Errors a tool turns into a ``ToolException``. Anything else (a
#: configuration error, say) is raised as itself.
_TOOL_ERRORS = (LexAPIError, InvalidArgumentError, ClientClosedError)


_SEARCH_LEAD = (
    "Search EU legislation (EUR-Lex) by meaning. Returns matching articles, each followed by "
    "a Source line with its CELEX citation. With as_of (YYYY-MM-DD) it asks for the text for "
    "that date; "
)
_SEARCH_DATE = {
    "withhold": "an article whose text LexAPI did not establish for that date comes back as a "
    "notice without text.",
    "drop": "an article whose text LexAPI did not establish for that date is left out, and the "
    "output says how many were left out.",
    "keep_labelled": "for an article whose text LexAPI did not establish for that date you get "
    "its latest text, under a label saying it is not established for that date.",
}
_SEARCH_TAIL = (
    " Consolidated texts have no legal force; the Source line names the act to cite when "
    "LexAPI knows it."
)


def search_description(unresolved: UnresolvedPolicy) -> str:
    """The description of ``lexapi_search_legislation`` for a policy."""
    return _SEARCH_LEAD + _SEARCH_DATE[unresolved] + _SEARCH_TAIL


_GENERATED_DESCRIPTIONS = frozenset(search_description(p) for p in _SEARCH_DATE)  # type: ignore[arg-type]


_READ_BASE = (
    "Read an EU document by its CELEX number: one article (article argument), or its articles "
    "in order. Each passage is followed by a Source line with its CELEX citation. Consolidated "
    "texts (CELEX starting with 0) have no legal force: cite the act as published in the "
    "Official Journal."
)
_READ_BUDGET = re.compile(
    re.escape(_READ_BASE)
    + r" The output is limited by a character budget of \d+ characters of text; the articles "
    r"not shown are listed by reference\."
)


def read_description(max_chars: int | None) -> str:
    """The description of ``lexapi_read_document`` for a budget."""
    if max_chars is None:
        return _READ_BASE
    return (
        f"{_READ_BASE} The output is limited by a character budget of {max_chars} characters "
        "of text; the articles not shown are listed by reference."
    )


def _artifact(documents: list[Document], report: RetrievalReport) -> Artifact:
    return {
        "passages": [dict(document.metadata) for document in documents],
        "report": dataclasses.asdict(report),
    }


def _join(documents: list[Document]) -> str:
    return "\n\n".join(f"[{n}] {doc.page_content}" for n, doc in enumerate(documents, 1))


class _LexAPITool(LexAPIClientFields, BaseTool):
    # LangChain's base classes ignore unknown arguments; these refuse them.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    @field_validator("response_format")
    @classmethod
    def _only_content_and_artifact(cls, value: str) -> str:
        if value != "content_and_artifact":
            raise ValueError("response_format must be 'content_and_artifact'")
        return value

    # A tool whose description follows its settings returns it here; the
    # settings it follows are frozen, and a copy regenerates it.
    def _description_text(self) -> str | None:
        return None

    @classmethod
    def _is_generated_text(cls, text: str) -> bool:
        return False

    def _generated_description(self) -> bool:
        return "description" not in self.model_fields_set or self._is_generated_text(
            self.description
        )

    def _set_generated_description(self) -> None:
        text = self._description_text()
        if text is not None:
            self.description = text
            self.__pydantic_fields_set__.discard("description")

    @model_validator(mode="after")
    def _describe(self) -> Self:
        if self._description_text() is not None and self._generated_description():
            self._set_generated_description()
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy (see :meth:`LexAPIClientFields.__copy__`); a generated
        description is regenerated for the copy's settings."""
        update = dict(update or {})
        if "description" in update:
            generated = self._is_generated_text(update["description"])
        else:
            generated = self._generated_description()
        copy = super().model_copy(update=update, deep=deep)
        if generated:
            copy._set_generated_description()
        return copy


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
    tool's settings, so the same rules apply: ``k`` hits are asked for; a
    text not established for ``as_of`` is, by default
    (``unresolved="withhold"``), replaced by a notice, left out with
    ``"drop"`` (the output says how many), or kept under a label with
    ``"keep_labelled"``; errors are raised. The description follows
    ``unresolved``, which cannot be changed after construction (use
    ``model_copy(update={"unresolved": ...})``). Cost: 5 credits per search;
    with ``as_of``, 0 to 5 credits for each version fetched (at most 3 per
    search).
    """

    name: str = "lexapi_search_legislation"
    #: Built from ``unresolved`` unless another text is given; see
    #: :func:`search_description`.
    description: str = ""
    args_schema: type[BaseModel] = SearchLegislationInput

    #: Hits to ask for per search.
    k: int = Field(default=5, ge=1, le=50, strict=True)
    #: Language of the texts (ISO 639-1); ``None`` is the API's default.
    language: str | None = None
    #: Similarity floor; ``None`` is the API's default floor.
    min_score: float | None = None
    #: See :attr:`LexAPILegislationRetriever.unresolved`. Frozen, so the
    #: description cannot go stale.
    unresolved: UnresolvedPolicy = Field(default="withhold", frozen=True)

    def _description_text(self) -> str | None:
        return search_description(self.unresolved)

    @classmethod
    def _is_generated_text(cls, text: str) -> bool:
        return text in _GENERATED_DESCRIPTIONS

    def _retriever(self, async_client: AsyncLexAPI | None) -> LexAPILegislationRetriever:
        return LexAPILegislationRetriever(
            client=self._sync,
            async_client=async_client,
            k=self.k,
            language=self.language,
            min_score=self.min_score,
            unresolved=self.unresolved,
        )

    @staticmethod
    def _content(documents: list[Document], report: RetrievalReport) -> str:
        content = _join(documents) if documents else "LexAPI returned no matching articles."
        if report.dropped:
            content += (
                f"\n\n[{report.dropped} hit(s) left out: LexAPI did not establish their "
                f"text for {report.as_of or 'the date asked'}.]"
            )
        if report.hint:
            content += f"\n\nHint from LexAPI: {report.hint}"
        return content

    def _run(
        self,
        query: str,
        as_of: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        config: RunnableConfig = {"callbacks": run_manager.get_child() if run_manager else None}
        overrides = {"as_of": as_of} if as_of is not None else {}
        failure: Exception | None = None
        try:
            self._sync_client()  # names this tool when only an async client was given
            documents, report = self._retriever(self._async).invoke_with_report(
                query, config, **overrides
            )
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
        return self._content(documents, report), _artifact(documents, report)

    async def _arun(
        self,
        query: str,
        as_of: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        config: RunnableConfig = {"callbacks": run_manager.get_child() if run_manager else None}
        overrides = {"as_of": as_of} if as_of is not None else {}
        failure: Exception | None = None
        try:
            documents, report = await self._retriever(self._async_client()).ainvoke_with_report(
                query, config, **overrides
            )
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
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

    ``max_chars`` bounds the text: the sum of the articles' texts (and
    notices) in one output, in whole articles. Once an article does not fit
    what is left, it and every later article are listed by reference
    instead, so the model can ask for one. Only the first article is cut,
    when it alone is longer than ``max_chars``: at a paragraph break when one
    falls in the second half of the limit, otherwise at a line break or a
    space there, otherwise at the limit, with a note; nothing else is then
    shown. The numbering, the cut
    note, the ``Source:`` lines and the list of articles not shown come on
    top of the bound. The description states the budget; ``max_chars`` is
    frozen so the description stays true.
    Annexes and recitals are not included. Cost: 5 credits, settled at 1
    when LexAPI already holds the document and 4 when it is read from the
    Publications Office.
    """

    name: str = "lexapi_read_document"
    #: Built from ``max_chars`` unless another text is given; see
    #: :func:`read_description`.
    description: str = ""
    args_schema: type[BaseModel] = ReadDocumentInput

    #: Most characters of text (articles' texts and notices, without Source
    #: lines) to return per call; ``None`` returns all. Frozen, so the
    #: description cannot go stale.
    max_chars: int | None = Field(default=20_000, ge=200, frozen=True)

    def _description_text(self) -> str | None:
        return read_description(self.max_chars)

    @classmethod
    def _is_generated_text(cls, text: str) -> bool:
        return text == _READ_BASE or _READ_BUDGET.fullmatch(text) is not None

    #: Language used when a call does not name one (ISO 639-1); ``None`` is
    #: the API's default.
    language: str | None = None

    def _request(self, article: str | None, language: str | None) -> dict[str, Any]:
        include = ["metadata", "articles"] if article else ["metadata", "articles", "fullText"]
        return {"include": include, "article_id": article, "language": language or self.language}

    def _render(
        self, response: DocumentResponse, celex: str, article: str | None
    ) -> tuple[str, Artifact]:
        if not _document_answer_readable(response):
            _fail(_unreadable_answer())
        result: PassageResult = lexapi_passages.from_document(response)
        documents, report, rest = to_documents(
            result, requested=None, max_chars=self.max_chars, budget=self.max_chars
        )
        if not documents:
            what = f"article {article} of CELEX {celex}" if article else f"CELEX {celex}"
            return f"LexAPI returned no text for {what}.", _artifact(documents, report)
        content = _join(documents)
        if rest:
            refs = ", ".join(p.article_ref or "?" for p in rest)
            content += (
                f"\n\n[Not shown: {len(rest)} more article(s) of CELEX {celex}: {refs}. "
                "Read one with the article argument.]"
            )
        return content, _artifact(documents, report)

    def _run(
        self,
        celex: str,
        article: str | None = None,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        failure: Exception | None = None
        try:
            _check_celex(celex)
            response = self._sync_client().get_document(celex, **self._request(article, language))
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
        return self._render(response, celex, article)

    async def _arun(
        self,
        celex: str,
        article: str | None = None,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        failure: Exception | None = None
        try:
            client = self._async_client()
        except ClientClosedError as error:
            failure = error
        if failure is not None:
            _fail(failure)
        if client is None:
            fallback: tuple[str, Artifact] = await super()._arun(
                celex, article, language, run_manager=run_manager
            )
            return fallback
        failure = None
        try:
            _check_celex(celex)
            response = await client.get_document(celex, **self._request(article, language))
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
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

    #: Language used when a call does not name one (ISO 639-1); ``None`` is
    #: the API's default.
    language: str | None = None

    def _request(self, as_of: str, provision: str | None, language: str | None) -> dict[str, Any]:
        date = _date_argument(as_of)
        if date is None:
            raise InvalidArgumentError("as_of is required")
        return {
            "as_of": date.isoformat(),
            "provision": provision,
            "language": language or self.language,
        }

    @staticmethod
    def _render(
        response: LegislationAsOfResponse, celex: str, provision: str | None
    ) -> tuple[str, Artifact]:
        if not _as_of_answer_readable(response):
            _fail(_unreadable_answer())
        result = lexapi_passages.from_legislation_as_of(response, provision=provision)
        documents, report, _ = to_documents(result, requested=None)
        if not documents:  # a readable answer always names its document
            _fail(_unreadable_answer())
        return documents[0].page_content, _artifact(documents, report)

    def _run(
        self,
        celex: str,
        as_of: str,
        provision: str | None = None,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        failure: Exception | None = None
        try:
            _check_celex(celex)
            request = self._request(as_of, provision, language)
            response = self._sync_client().legislation_as_of(celex, **request)
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
        return self._render(response, celex, provision)

    async def _arun(
        self,
        celex: str,
        as_of: str,
        provision: str | None = None,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, Artifact]:
        failure: Exception | None = None
        try:
            _check_celex(celex)
            request = self._request(as_of, provision, language)
            client = self._async_client()
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
        if client is None:
            fallback: tuple[str, Artifact] = await super()._arun(
                celex, as_of, provision, language, run_manager=run_manager
            )
            return fallback
        failure = None
        try:
            response = await client.legislation_as_of(celex, **request)
        except _TOOL_ERRORS as error:
            failure = error
        except RuntimeError as error:
            failure = self._closed_race(error)
        if failure is not None:
            _fail(failure)
        return self._render(response, celex, provision)
