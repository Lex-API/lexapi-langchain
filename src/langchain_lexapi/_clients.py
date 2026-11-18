"""Client fields shared by the retrievers and tools.

Each retriever or tool holds its own clients: the ones passed in, or ones it
builds from ``api_key`` (or ``LEXAPI_API_KEY``). Nothing is built at import
time and no client is shared between instances.
"""

from __future__ import annotations

from typing import Any

from lexapi import AsyncLexAPI, LexAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from langchain_lexapi._version import __version__

#: Appended to the SDK's User-Agent by the clients this package builds.
USER_AGENT_SUFFIX = f"langchain-lexapi/{__version__}"


class LexAPIClientFields(BaseModel):
    """Connection settings.

    Pass ``client`` and/or ``async_client`` to use your own SDK clients, or
    ``api_key`` (else ``LEXAPI_API_KEY``) and optionally ``base_url`` to have
    both built for you. The key is never shown in ``repr`` and never
    serialised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @model_validator(mode="after")
    def _build_clients(self) -> LexAPIClientFields:
        self._sync = self.client
        self._async = self.async_client
        if self.client is None and self.async_client is None:
            kwargs: dict[str, Any] = {"user_agent_suffix": USER_AGENT_SUFFIX}
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            key = self.api_key.get_secret_value() if self.api_key is not None else None
            self._sync = LexAPI(key, **kwargs)
            self._async = AsyncLexAPI(key, **kwargs)
        return self

    def _sync_client(self) -> LexAPI:
        if self._sync is None:
            raise ValueError(
                f"{type(self).__name__} was given only an async_client; "
                "use the async methods or pass client= as well"
            )
        return self._sync
