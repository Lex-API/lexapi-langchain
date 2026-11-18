# langchain-lexapi

LangChain retrievers and tools for EU law from [LexAPI](https://lex-api.com): EU legislation
article by article, CJEU case law, and the text of an act for a date, each with its CELEX
citation.

- `LexAPILegislationRetriever`: semantic search over EU legislation, optionally for a date (`as_of`).
- `LexAPICaseLawRetriever`: semantic search over CJEU case law.
- Tools: `lexapi_search_legislation`, `lexapi_read_document` (a document or one article) and
  `lexapi_legislation_as_of` (the version of an act listed for a date, and one provision of it).

Documents and tool outputs are built by the SDK's `lexapi.passages` module, so this package
applies the same citation and date rules as the [Python SDK](https://github.com/Lex-API/lexapi-python).

## Install

Version 0.1.0 is not on PyPI yet. Install it from this repository:

```bash
pip install "langchain-lexapi @ git+https://github.com/Lex-API/lexapi-langchain"
```

It needs Python 3.11 or later, `langchain-core` 1.4 or later and the LexAPI Python SDK. Get an
API key from the [LexAPI dashboard](https://lex-api.com/dashboard) and set `LEXAPI_API_KEY`, or
pass `api_key=`.

## Examples

Legislation, for a date. Each document's `page_content` is the text followed by a `Source:` line
with the citation, so a stock LangChain prompt shows the model where the text comes from:

```python
from langchain_lexapi import LexAPILegislationRetriever

retriever = LexAPILegislationRetriever(k=3, as_of="2019-06-01")
docs = retriever.invoke("right to erasure")
for doc in docs:
    print(doc.page_content)
    # No cite_celex when a consolidated text does not name its act.
    print(doc.metadata.get("cite_celex"), doc.metadata["text_withheld"])
```

Case law. A hit's text is the whole decision; `max_chars` bounds it: the text is cut at a paragraph
break when one is near, otherwise at a line break, a space or the limit, and a note says so. The
note and the `Source:` line come on top of the bound:

```python
from langchain_lexapi import LexAPICaseLawRetriever

retriever = LexAPICaseLawRetriever(k=2, max_chars=4000, court="Court of Justice")
docs, report = retriever.invoke_with_report("right to be forgotten")
print(len(docs), "documents;", report.units_charged, "credits charged")
```

Tools, for an agent or a tool-calling model. Each returns the text for the model and, as the
artifact, the passages' metadata and the call's report:

```python
from langchain_lexapi import (
    LexAPILegislationAsOfTool,
    LexAPIReadDocumentTool,
    LexAPISearchLegislationTool,
)

tools = [LexAPISearchLegislationTool(), LexAPIReadDocumentTool(), LexAPILegislationAsOfTool()]
as_of = tools[2]
message = as_of.invoke(
    {
        "type": "tool_call",
        "id": "call-1",
        "name": as_of.name,
        "args": {"celex": "32016R0679", "as_of": "2019-06-01", "provision": "art_17"},
    }
)
print(message.content)
print(message.artifact["report"]["units_charged"])
```

`lexapi_read_document` returns at most `max_chars` (default 20,000) characters of text per call:
whole articles, a single longer text cut at a paragraph break when one is near (otherwise at a line
break, a space or the limit), then a list of the articles not shown so the model can ask for one.
The numbering, the cut note, the `Source:` lines and that list come on top of the bound. The tool's report counts `truncated` (texts cut) and `not_shown`.

Pass your own SDK clients with `client=` (a `lexapi.LexAPI`) and `async_client=` (a
`lexapi.AsyncLexAPI`) to control timeouts, retries and their lifetime: this package never closes a
client you pass in. Without them, each retriever or tool builds its own clients from `api_key` (or
`LEXAPI_API_KEY`) and `base_url`: the sync client at construction, the async client on first async
use. Close those with `close()` (sync client) or `aclose()` (both), or use the object as a context
manager (`with` / `async with`). With only `client=`, async calls run the sync client in an
executor; with only `async_client=`, sync calls are refused. Unknown constructor arguments raise a
validation error.

## What a document holds

- `page_content`: `lexapi.passages.format_passage(passage)`: the text, or a one-line notice when
  there is no text, then `Source: <citation>`. With `include_citation_in_content=False` it is the
  text or notice alone.
- `metadata`: `passage.metadata()` unchanged (flat strings, numbers and booleans; never the text),
  plus `source`, LangChain's usual key, set to the passage's `url` when it has one. Always present:
  `kind`, `celex`, `text_status`, `text_withheld`, `consolidated`, `citation`, `id`. Present when
  known: `cite_celex`, `article_ref`, `article_number`, `article_title`, `title`, `language`,
  `url`, `eli`, `score`, `base_celex`, `consolidated_as_of`, `consolidated_note`, `as_of`,
  `as_of_resolved`, `as_of_reason`, `version_celex`, `version_kind`, `version_date`,
  `version_valid_until`, `ecli`, `case_number`, `court`, `decision_date`, `document_type`,
  `paragraph_number`.
- `id`: `passage.id`, `lexapi:<celex>:<part>:<as_of>:<language>`. It is the same on every call that
  returns the same passage. Within one result, passages with the same id are returned once. They
  rank: a text from the API, then a latest text under a label (`keep_labelled`), then a notice or
  no text. A later passage replaces the earlier one only if it ranks strictly higher, and the
  document keeps the position of the first.
- `metadata["score"]` is the API's score, unchanged.

## The text for a date

- With `as_of`, the API answers each hit with the version listed for that date. A hit it could not
  establish for the date still carries its latest text in the API response. By default
  (`unresolved="withhold"`) this package does not pass that text on; `"drop"` leaves the hit out,
  and `"keep_labelled"` passes the latest text on under a label saying it is not established for
  the date.
- `unresolved="withhold"` (the default) returns such a hit as a document whose content is a notice
  with the reason, and `metadata["text_withheld"] = True`. `"drop"` leaves it out.
  `"keep_labelled"` returns the latest text under a label saying it is not established for the
  date. `emit_withheld=False` leaves the notices out.
- Do not index documents with `metadata["text_withheld"] = True` in a vector store: they hold a
  notice, not a text.
- `lexapi_legislation_as_of` returns a notice with the reason, and no text, when LexAPI cannot
  establish the version or the text for the date. It does not say whether a provision applied on
  that date.

## `k` and the `celex` restriction

`k` is the number of hits asked for. Fewer documents can come back: the API's relevance floor,
withheld or dropped texts, de-duplication. The retrievers never ask for more hits to make up the
number. `celex=` keeps only passages of the given acts (and their consolidated texts). The API
cannot limit a search to one act, so this filters the `k` hits after the search: it costs the same
as the search without it, and the result can be shorter than `k` or empty. Raise `k` if you need
more candidates; the search costs the same, but with `as_of` more hits can mean more version
fetches.

## Credits

Each call draws credits from your LexAPI plan. As documented by the API today:

- Legislation search: 5 credits. With `as_of`, each version LexAPI does not hold yet is fetched (at
  most 3 per search), at 0 to 5 credits each.
- Case-law search: 5 credits.
- `lexapi_read_document`: 5 credits, settled at 1 when LexAPI already holds the document and 4 when
  it is read from the Publications Office.
- `lexapi_legislation_as_of`: 1 credit when no text is fetched; 4 or 5 when the provision's text is
  fetched; 1 if that fetch finds no text.

See the [API reference](https://lex-api.com/docs) for current prices. To see what a call cost, use
`invoke_with_report` / `ainvoke_with_report` on a retriever (it returns the documents and a
`RetrievalReport` with `units_charged`, `credits_remaining`, the core's withheld, labelled, dropped
and filtered counts, and what was returned: `returned`, `returned_withheld`, `returned_labelled`),
read `artifact["report"]` of a tool message, or handle the `lexapi_retrieval_report`
callback event, which every retriever call sends on its own run. The report is per call and is not
stored on the retriever, so concurrent calls do not mix.

## Errors

An error is never turned into an empty list. A result can still be empty without an error: no hit
above the relevance floor, a `celex` restriction that matched nothing, hits left out by
`unresolved="drop"` or `emit_withheld=False`, or hits without a CELEX number.

- Retrievers raise the SDK's exceptions unchanged, with the API's own message (nothing is masked
  or reworded): `lexapi.CreditsExhaustedError` (402),
  `lexapi.PaymentPastDueError` (403), `lexapi.RateLimitedError` (429, after the SDK's own
  retries), `lexapi.AuthenticationError` (401), and the rest of `lexapi.LexAPIError`.
- Tools raise `langchain_core.tools.ToolException` with the API's error code and message, for
  example `LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 7 s.`, with
  anything shaped like an API key masked. An answer that could not be read gives
  `LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.` and quotes nothing from
  it. An argument that matches the tool's schema but is an impossible date (`2099-02-30`) gives
  `Invalid argument: ...` before any request; an argument that does not match the tool's schema
  (`as_of="1 June 2099"`) raises the framework's validation error. Use after `close()` gives
  `... is closed`. A configuration error (such as a tool given only an `async_client` and called
  synchronously) is raised as itself. Set `handle_tool_error=True` on a tool
  to hand that message to the model instead of raising.

The API key is never logged, never in `repr()` and never serialised.

## What this package does not do

- LexAPI provides source texts and citations. It does not assess whether anyone complies with the
  law, and neither does this package. It is not legal advice.
- Consolidated texts have no legal force: they are documentation. Cite the act as published in the
  Official Journal (`metadata["cite_celex"]`).
- A text is passed on as the text for a date only when LexAPI established it for that date. For any
  other hit, by default (`unresolved="withhold"`) this package returns a notice with the reason
  LexAPI gave, not the text; `"drop"` leaves the hit out, and `"keep_labelled"` returns the latest
  text under a label saying it is not established for the date.
- It does not call a language model, build prompts or chains, embed texts or cache results.

## Tests

The tests run offline against synthetic fixtures: no request reaches LexAPI, a model provider or
LangSmith. They do not cover the live service.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
