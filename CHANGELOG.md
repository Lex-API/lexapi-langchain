# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-22

### Added

- `LexAPILegislationRetriever`: semantic search over EU legislation, one document per article,
  with `as_of` for the text for a date, the `unresolved` policies `withhold` (default), `drop` and
  `keep_labelled`, `emit_withheld`, and a `celex` restriction applied after the search.
- `LexAPICaseLawRetriever`: semantic search over CJEU case law with the decision's text, the
  `court`, `date_from` and `date_to` filters, and an optional `max_chars` cut at a paragraph break.
- Tools `lexapi_search_legislation`, `lexapi_read_document` and `lexapi_legislation_as_of`,
  returning content and an artifact (passage metadata and the call's report).
- `RetrievalReport`, delivered per call by `invoke_with_report` / `ainvoke_with_report`, by the
  `lexapi_retrieval_report` callback event and in tool artifacts.
- Documents, metadata and ids are built by the SDK's `lexapi.passages`; the only metadata key added
  is `source`.
- Sync and async, with the SDK's sync and async clients. Clients built by an instance are closed
  by its `close()` / `aclose()` or context manager; clients passed in are never closed. The async
  client is built on first async use.
- De-duplication by passage id: a text from the API ranks over a labelled latest text, which ranks
  over a notice; a later passage replaces an earlier one only if it ranks strictly higher.
- `InvalidArgumentError` for a bad per-call `k` or `as_of`, also checked by
  `lexapi_legislation_as_of` before any request. Unknown constructor arguments raise, and so do
  clients passed together with `api_key` / `base_url`. Use after `close()` / `aclose()` raises.
- An answer that could not be read gives tools a fixed error text that quotes nothing from it,
  including a document read or as-of answer that parsed but lacks its documented shape.
- The read tool needs a document with a `celex` string and a string `fullText` when present.
- The read tool's budget is whole articles; only the first may be cut. Its description states the
  budget, and `max_chars` is frozen.
- `as_of` with a time part and `k` given as a string, float or boolean are refused.
- Copies cannot change connection settings; deep copies and pickles are refused. A call that
  races `close()` raises `ClientClosedError`.
- `ClientClosedError` is exported. A `model_copy` shares its source's clients and never closes
  them; it raises `ClientClosedError` once the source has closed the clients it built.
- The search tool's description follows its `unresolved` policy, which is frozen; `model_copy`
  regenerates it.
- `tool_error` is exported: it turns a `lexapi.LexAPIError` into the `ToolException` this
  package's tools raise, with the same text, for tools of your own. Anything else raises
  `TypeError`. It can also be imported as `langchain_lexapi.tools.tool_error`.
- Tool error text has a fixed form: `LexAPI error CODE (HTTP n): <message>.`, then, each only when
  it applies and in this order, `Credits reset at <t>.`, `This is an exhausted quota, not a rate
  limit.`, `Quota resets at <t>.`, `Access ended at <t>.`, `Billing: <url>` (not when the message
  already holds the link) and `Retry after <n> s.`.
- `Retry after <n> s.` is given only for an error LexAPI's client retries (a 429 rate limit, or a
  521/522/523), as a plain decimal above 0; never for a failed payment, an exhausted quota, a 402,
  another 4xx or a 5xx.
- The server's strings in the text are put on one line and capped (the message at 1,000
  characters, a time or link at 200); a blank one is left out, and a message ending in `:` ends in
  `.`.
- Requires `lexapi-client` 0.5.0 or later (below 0.6). A failed payment (`PaymentPastDueError`, a
  subclass of `TierForbiddenError`) and an exhausted quota are never retried. Answers that cannot
  be read (wrong shape, non-strict JSON such as `NaN`/`Infinity` or nesting deeper than 512
  levels, an iterator whose server stops advancing) make retrievers raise
  `lexapi.InvalidResponseError` and give tools the text `LexAPI error INVALID_RESPONSE (HTTP 200):
  The answer could not be read.` This package's additional checks give that text without the
  status.
- Read tool budget: once the first article alone is over the budget, nothing else is shown, and
  the cut note is not counted; the articles shown are always a prefix of the document's.
- Empty or whitespace-only `query`/`celex`: tools give `Invalid argument: …`, retrievers raise
  `InvalidArgumentError`. A whitespace-only `article` (read tool) or `provision` (as-of tool) is
  refused too; an absent one is "not given". All before any request.
- The search tool given only an async client and called synchronously names itself in its error.
