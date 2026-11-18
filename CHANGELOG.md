# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- `lexapi-client` pinned to `19da4dd`, which checks document and as-of answers against their
  documented shape. Such answers (no `document`, say) now give the tools'
  `LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.`; the text without the
  status remains for this package's additional checks.

## [0.1.0] - not released yet

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
