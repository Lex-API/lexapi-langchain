"""Retrievers for EU legislation and CJEU case law from LexAPI.

Every document is built by :mod:`lexapi.passages`: its text (or a notice
when there is none) followed by a ``Source:`` citation line, and the
passage's citation metadata. Errors from the SDK are raised, never turned
into an empty list.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, ClassVar

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForRetrieverRun,
    CallbackManager,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor
from lexapi import passages
from lexapi.passages import PassageResult, UnresolvedPolicy
from pydantic import ConfigDict, Field, field_validator

from langchain_lexapi._clients import LexAPIClientFields
from langchain_lexapi._documents import REPORT_EVENT, RetrievalReport, to_documents
from langchain_lexapi._report import caught, config_with_catcher

__all__ = ["LexAPICaseLawRetriever", "LexAPILegislationRetriever"]

_DATE_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class InvalidArgumentError(ValueError):
    """A per-call argument (``k`` or ``as_of``) that is not valid. Raised
    before any request is sent."""


def _date_argument(value: Any) -> datetime.date | None:
    """A ``datetime.date`` (not a ``datetime``), a ``YYYY-MM-DD`` string, or
    ``None``. Anything else, including a time part, is refused."""
    if value is None:
        return None
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str) and _DATE_TEXT.match(value):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            pass
    raise InvalidArgumentError(f"as_of must be a date in the form YYYY-MM-DD; got {value!r}")


def _iso(value: datetime.date | None) -> str | None:
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    return value.isoformat() if value is not None else None


def _run_handlers(run: CallbackManagerForRetrieverRun) -> CallbackManager:
    """A manager for every handler of the retriever's run, so the report
    event reaches the run's own handlers as well as inherited ones."""
    return CallbackManager(
        handlers=run.handlers,
        inheritable_handlers=run.inheritable_handlers,
        parent_run_id=run.parent_run_id,
        tags=run.tags,
        inheritable_tags=run.inheritable_tags,
        metadata=run.metadata,
        inheritable_metadata=run.inheritable_metadata,
    )


def _async_run_handlers(run: AsyncCallbackManagerForRetrieverRun) -> AsyncCallbackManager:
    return AsyncCallbackManager(
        handlers=run.handlers,
        inheritable_handlers=run.inheritable_handlers,
        parent_run_id=run.parent_run_id,
        tags=run.tags,
        inheritable_tags=run.inheritable_tags,
        metadata=run.metadata,
        inheritable_metadata=run.inheritable_metadata,
    )


class _LexAPIRetriever(LexAPIClientFields, BaseRetriever):
    """Shared call flow: one API call, conversion by ``lexapi.passages``,
    one report event per call."""

    # LangChain's base classes ignore unknown arguments; these refuse them.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    #: Keyword arguments ``invoke`` accepts as per-call overrides.
    _overrides: ClassVar[frozenset[str]] = frozenset({"k"})
    #: Largest ``k`` the endpoint accepts.
    _max_k: ClassVar[int] = 50
    k: int

    #: Put the citation (``Source:`` line) into ``page_content``. LangChain's
    #: document prompts show only ``page_content``, so leave this on unless
    #: your prompt adds ``metadata["citation"]`` itself.
    include_citation_in_content: bool = True

    def _check_overrides(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = {key: value for key, value in kwargs.items() if key != "verbose"}
        unknown = set(kwargs) - self._overrides
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.invoke() got unexpected keyword arguments: "
                f"{', '.join(sorted(unknown))}"
            )
        if "k" in kwargs:
            k = kwargs["k"]
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self._max_k:
                raise InvalidArgumentError(
                    f"k must be an integer from 1 to {self._max_k}; got {k!r}"
                )
        kwargs.setdefault("k", self.k)
        return kwargs

    # Subclasses: build the request and convert the response.
    def _call(self, query: str, options: dict[str, Any]) -> PassageResult:
        raise NotImplementedError

    async def _acall(self, query: str, options: dict[str, Any]) -> PassageResult:
        raise NotImplementedError

    def _conversion(self) -> dict[str, Any]:
        """Keyword arguments for ``to_documents``."""
        return {"include_citation_in_content": self.include_citation_in_content}

    def _convert(
        self, result: PassageResult, options: dict[str, Any]
    ) -> tuple[list[Document], RetrievalReport]:
        documents, report, _ = to_documents(result, requested=options["k"], **self._conversion())
        return documents, report

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        options = self._check_overrides(kwargs)
        try:
            result = self._call(query, options)
        except RuntimeError as error:
            raise self._closed_race(error) from None
        documents, report = self._convert(result, options)
        _run_handlers(run_manager).on_custom_event(REPORT_EVENT, report, run_id=run_manager.run_id)
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        options = self._check_overrides(kwargs)
        try:
            if self._async_client() is not None:
                result = await self._acall(query, options)
            else:
                result = await run_in_executor(None, self._call, query, options)
        except RuntimeError as error:
            raise self._closed_race(error) from None
        documents, report = self._convert(result, options)
        await _async_run_handlers(run_manager).on_custom_event(
            REPORT_EVENT, report, run_id=run_manager.run_id
        )
        return documents

    def invoke_with_report(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[list[Document], RetrievalReport]:
        """``invoke``, and the :class:`RetrievalReport` of this call (credits
        charged and remaining, withheld and dropped counts). Safe to call
        from several threads at once: the report travels with the call's own
        callbacks, not on the retriever."""
        config, catcher = config_with_catcher(config)
        documents = self.invoke(input, config, **kwargs)
        return documents, caught(catcher)

    async def ainvoke_with_report(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[list[Document], RetrievalReport]:
        """Async :meth:`invoke_with_report`."""
        config, catcher = config_with_catcher(config)
        documents = await self.ainvoke(input, config, **kwargs)
        return documents, caught(catcher)


class LexAPILegislationRetriever(_LexAPIRetriever):
    """EU legislation, article by article, from LexAPI's semantic search.

    Each hit becomes one ``Document``. ``k`` is the number of hits asked
    for; fewer documents can come back (no hits above the relevance floor,
    withheld texts left out, a ``celex`` restriction). The retriever never
    asks for more hits to make up the number.

    With ``as_of`` the API answers each hit with the version listed for that
    date. A hit it could not establish for the date still carries its latest
    text in the API response. By default (``unresolved="withhold"``) that
    text is not passed on: the hit becomes a document whose content is a
    notice saying no text is established for the date and why, with
    ``metadata["text_withheld"] = True``. Do not index such documents in a
    vector store. ``"drop"`` leaves such hits out; ``"keep_labelled"``
    passes the latest text on under a label saying it is not established
    for the date.

    Cost: 5 credits per search. With ``as_of``, each version LexAPI does not
    hold yet is fetched (at most 3 per search), at 0 to 5 credits
    each. See the API reference for current prices.

    Per-call overrides: ``invoke(query, k=3, as_of="2019-06-01")``.
    """

    _overrides: ClassVar[frozenset[str]] = frozenset({"k", "as_of"})

    #: Hits to ask for (the API's ``limit``); plans can cap it lower.
    k: int = Field(default=5, ge=1, le=50, strict=True)
    #: Ask for the text for this date (``YYYY-MM-DD`` or a ``datetime.date``).
    as_of: datetime.date | None = None

    @field_validator("as_of", mode="before")
    @classmethod
    def _as_of_is_a_date(cls, value: Any) -> datetime.date | None:
        return _date_argument(value)

    #: Language of the texts (ISO 639-1); ``None`` is the API's default.
    language: str | None = None
    #: Similarity floor; ``None`` is the API's default floor.
    min_score: float | None = None
    #: Keep only passages of these acts (CELEX numbers). Applied to the hits
    #: after the search, because the API cannot limit a search to one act:
    #: the search still asks for ``k`` hits and costs the same, and the
    #: result can be shorter than ``k`` or empty.
    celex: str | list[str] | None = None
    #: What to do with a hit whose text is not established for ``as_of``:
    #: ``"withhold"`` (a notice instead of the text, the default), ``"drop"``
    #: (leave it out) or ``"keep_labelled"`` (the latest indexed text, under
    #: a label saying it is not established for the date).
    unresolved: UnresolvedPolicy = "withhold"
    #: Return withheld passages as notice documents. ``False`` leaves them
    #: out; the report still counts them.
    emit_withheld: bool = True

    @field_validator("celex")
    @classmethod
    def _celex_not_empty(cls, value: str | list[str] | None) -> str | list[str] | None:
        if value is not None:
            # Same check the core applies, at construction time.
            passages.restrict_to_celex(PassageResult(passages=()), value)
        return value

    def _check_overrides(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        options = super()._check_overrides(kwargs)
        if "as_of" in options:
            options["as_of"] = _date_argument(options["as_of"])
        else:
            options["as_of"] = self.as_of
        return options

    def _request(self, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "limit": options["k"],
            "include_text": True,
            "language": self.language,
            "min_score": self.min_score,
            "as_of": _iso(options["as_of"]),
        }

    def _passages(self, response: Any, options: dict[str, Any]) -> PassageResult:
        result = passages.from_legislation_search(
            response, unresolved=self.unresolved, as_of=_iso(options["as_of"])
        )
        if self.celex is not None:
            result = passages.restrict_to_celex(result, self.celex)
        return result

    def _call(self, query: str, options: dict[str, Any]) -> PassageResult:
        response = self._sync_client().semantic_legislation_search(query, **self._request(options))
        return self._passages(response, options)

    async def _acall(self, query: str, options: dict[str, Any]) -> PassageResult:
        client = self._async_client()
        assert client is not None
        response = await client.semantic_legislation_search(query, **self._request(options))
        return self._passages(response, options)

    def _conversion(self) -> dict[str, Any]:
        return {**super()._conversion(), "emit_withheld": self.emit_withheld}


class LexAPICaseLawRetriever(_LexAPIRetriever):
    """CJEU case law from LexAPI's semantic search.

    The search is sent with ``include_text=True``. A hit's text is the whole
    decision, not the matching paragraph, so documents can be long; set
    ``max_chars`` to bound each text: it is cut at a paragraph break when
    one falls in the second half of the limit, otherwise at a line break or
    a space there, otherwise at the limit, with a note saying it was cut
    (the note and the ``Source:`` line come on top). A hit without text gets
    a notice, never its snippet or summary.

    ``k`` is the number of hits asked for; fewer documents can come back.
    Cost: 5 credits per search. See the API reference for current prices.
    """

    _max_k: ClassVar[int] = 100

    #: Hits to ask for (the API's ``limit``); plans can cap it lower.
    k: int = Field(default=5, ge=1, le=100, strict=True)
    #: Language of the texts (ISO 639-1); ``None`` is the API's default.
    language: str | None = None
    #: Similarity floor; ``None`` is the API's default floor.
    min_score: float | None = None
    #: Only this court (sent as ``filters.court``).
    court: str | None = None
    #: Only decisions on or after this date (``filters.date_from``).
    date_from: datetime.date | None = None
    #: Only decisions on or before this date (``filters.date_to``).
    date_to: datetime.date | None = None
    #: Bound each text to this many characters: cut at a paragraph break
    #: when one falls in the second half of the limit, otherwise at a line
    #: break or a space there, otherwise at the limit, with a note saying so.
    #: ``None`` (the default) passes whole texts.
    max_chars: int | None = Field(default=None, ge=200)

    def _request(self, options: dict[str, Any]) -> dict[str, Any]:
        filters = {
            key: value
            for key, value in (
                ("court", self.court),
                ("date_from", _iso(self.date_from)),
                ("date_to", _iso(self.date_to)),
            )
            if value is not None
        }
        return {
            "limit": options["k"],
            "include_text": True,
            "language": self.language,
            "min_score": self.min_score,
            "filters": filters or None,
        }

    def _call(self, query: str, options: dict[str, Any]) -> PassageResult:
        response = self._sync_client().semantic_search(query, **self._request(options))
        return passages.from_case_law_search(response)

    async def _acall(self, query: str, options: dict[str, Any]) -> PassageResult:
        client = self._async_client()
        assert client is not None
        response = await client.semantic_search(query, **self._request(options))
        return passages.from_case_law_search(response)

    def _conversion(self) -> dict[str, Any]:
        return {**super()._conversion(), "max_chars": self.max_chars}
