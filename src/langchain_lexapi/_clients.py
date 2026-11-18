"""Client fields shared by the retrievers and tools.

Each retriever or tool holds its own clients: the ones passed in, or ones it
builds from ``api_key`` (or ``LEXAPI_API_KEY``). Nothing is built at import
time and no client is shared between instances. Clients an instance built
are closed by its :meth:`~LexAPIClientFields.close` /
:meth:`~LexAPIClientFields.aclose`; clients passed in are never closed by it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from lexapi import AsyncLexAPI, LexAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from langchain_lexapi._version import __version__

#: Appended to the SDK's User-Agent by the clients this package builds.
USER_AGENT_SUFFIX = f"langchain-lexapi/{__version__}"


class ClientClosedError(RuntimeError):
    """The retriever or tool was used after ``close()`` / ``aclose()``."""


class _BuiltClients:
    """Shared by an object that built its clients and every copy of it:
    whether the builder has closed them."""

    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False


_CONNECTION_FIELDS = frozenset({"client", "async_client", "api_key", "base_url"})


class LexAPIClientFields(BaseModel):
    """Connection settings.

    Pass ``client`` and/or ``async_client`` to use your own SDK clients (you
    then control their lifetime), or ``api_key`` (else ``LEXAPI_API_KEY``)
    and optionally ``base_url`` to have them built for you: the sync client
    at construction, the async one on first async use. Passing clients
    together with ``api_key`` or ``base_url`` is refused. After ``close()``
    or ``aclose()`` the object cannot be used again. A ``model_copy`` shares
    the clients but never closes them; only the object that built them does,
    and a copy used after that raises :class:`ClientClosedError`. The key is never
    shown in ``repr`` and never serialised. Unknown constructor arguments
    raise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    #: A ``lexapi.LexAPI`` client, used for sync calls.
    client: LexAPI | None = Field(default=None, exclude=True, repr=False)
    #: A ``lexapi.AsyncLexAPI`` client, used for async calls.
    async_client: AsyncLexAPI | None = Field(default=None, exclude=True, repr=False)
    #: API key for the clients this object builds. Defaults to the
    #: ``LEXAPI_API_KEY`` environment variable (read by the SDK).
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    #: Base URL for the clients this object builds; ``None`` is the SDK's
    #: default (the LexAPI service).
    base_url: str | None = None

    _sync: LexAPI | None = PrivateAttr(default=None)
    _async: AsyncLexAPI | None = PrivateAttr(default=None)
    _owns_clients: bool = PrivateAttr(default=False)
    _closed: bool = PrivateAttr(default=False)
    _built: _BuiltClients = PrivateAttr(default_factory=_BuiltClients)

    @model_validator(mode="after")
    def _build_clients(self) -> Self:
        given = self.client is not None or self.async_client is not None
        if given and (self.api_key is not None or self.base_url is not None):
            raise ValueError("Pass either client/async_client or api_key/base_url, not both.")
        self._sync = self.client
        self._async = self.async_client
        if self.client is None and self.async_client is None:
            self._owns_clients = True
            self._sync = LexAPI(self._key(), **self._client_kwargs())
        return self

    def _key(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"user_agent_suffix": USER_AGENT_SUFFIX}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return kwargs

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A shallow copy that shares this object's clients (see
        :meth:`__copy__`). Connection settings cannot be changed in a copy,
        and a deep copy is refused."""
        changed = sorted(set(update or {}) & _CONNECTION_FIELDS)
        if changed:
            raise ValueError(
                "Connection settings cannot be changed in a copy; "
                f"create a new {type(self).__name__}."
            )
        return super().model_copy(update=update, deep=deep)

    def __copy__(self) -> Self:
        """A copy that shares this object's clients. The copy never closes
        them; once the object that built them closes them, the copy raises
        :class:`ClientClosedError`."""
        copy = super().__copy__()
        copy._sync, copy._async = self._sync, self._async
        copy._built = self._built
        copy._owns_clients = False
        copy._closed = False
        return copy

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        raise TypeError(self._no_deep_copy())

    def __getstate__(self) -> dict[Any, Any]:
        raise TypeError(self._no_deep_copy())

    def _no_deep_copy(self) -> str:
        return f"{type(self).__name__} holds an open client and cannot be deep-copied or pickled."

    def _is_closed(self) -> bool:
        return self._closed or self._built.closed

    def _closed_error(self) -> ClientClosedError:
        return ClientClosedError(f"{type(self).__name__} is closed; create a new one.")

    def _closed_race(self, error: RuntimeError) -> Exception:
        """The error to raise for a ``RuntimeError`` from a call: a call that
        raced ``close()`` and hit httpx's closed client becomes
        :class:`ClientClosedError`; anything else stays as it is."""
        if self._is_closed() and "client has been closed" in str(error):
            return self._closed_error()
        return error

    def _check_open(self) -> None:
        if self._is_closed():
            raise self._closed_error()

    def _sync_client(self) -> LexAPI:
        self._check_open()
        if self._sync is None:
            raise ValueError(
                f"{type(self).__name__} was given only an async_client; "
                "use the async methods or pass client= as well"
            )
        return self._sync

    def _async_client(self) -> AsyncLexAPI | None:
        """The async client: the one passed in, one built now (when this
        object builds its own clients), or ``None`` when only a sync client
        was passed in."""
        self._check_open()
        if self._async is None and self._owns_clients:
            self._async = AsyncLexAPI(self._key(), **self._client_kwargs())
        return self._async

    def close(self) -> None:
        """Close the sync client this object built. A client passed in is
        left open. An async client it built needs :meth:`aclose`. The object
        cannot be used afterwards."""
        self._closed = True
        if not self._owns_clients:
            return
        self._built.closed = True
        if self._sync is not None:
            self._sync.close()

    async def aclose(self) -> None:
        """Close the sync and async clients this object built. Clients passed
        in are left open. The object cannot be used afterwards."""
        self._closed = True
        if not self._owns_clients:
            return
        self._built.closed = True
        if self._async is not None:
            await self._async.close()
        if self._sync is not None:
            self._sync.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
