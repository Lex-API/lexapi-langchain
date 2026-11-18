"""LangChain retrievers and tools for EU law from LexAPI.

Documents and tool outputs are built by the SDK's :mod:`lexapi.passages`:
every text comes with its CELEX citation, and a text that LexAPI did not
establish for the date asked is replaced by a notice.
"""

from langchain_lexapi._documents import METADATA_ALIASES, REPORT_EVENT, RetrievalReport
from langchain_lexapi._version import __version__
from langchain_lexapi.retrievers import LexAPICaseLawRetriever, LexAPILegislationRetriever
from langchain_lexapi.tools import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

__all__ = [
    "METADATA_ALIASES",
    "REPORT_EVENT",
    "LexAPICaseLawRetriever",
    "LexAPILegislationAsOfTool",
    "LexAPILegislationRetriever",
    "LexAPIReadDocumentTool",
    "LexAPISearchLegislationTool",
    "RetrievalReport",
    "__version__",
]
