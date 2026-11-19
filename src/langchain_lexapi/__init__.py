"""LangChain retrievers and tools for EU law from LexAPI.

Documents and tool outputs are built by the SDK's :mod:`lexapi.passages`:
every text comes with its CELEX citation. A text that LexAPI did not
establish for the date asked is, by default (``unresolved="withhold"``),
replaced by a notice; ``"drop"`` leaves it out and ``"keep_labelled"`` keeps
the latest text under a label saying it is not established for the date.
"""

from langchain_lexapi._clients import ClientClosedError
from langchain_lexapi._documents import METADATA_ALIASES, REPORT_EVENT, RetrievalReport
from langchain_lexapi._version import __version__
from langchain_lexapi.retrievers import (
    InvalidArgumentError,
    LexAPICaseLawRetriever,
    LexAPILegislationRetriever,
)
from langchain_lexapi.tools import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
    tool_error,
)

__all__ = [
    "METADATA_ALIASES",
    "REPORT_EVENT",
    "ClientClosedError",
    "InvalidArgumentError",
    "LexAPICaseLawRetriever",
    "LexAPILegislationAsOfTool",
    "LexAPILegislationRetriever",
    "LexAPIReadDocumentTool",
    "LexAPISearchLegislationTool",
    "RetrievalReport",
    "__version__",
    "tool_error",
]
