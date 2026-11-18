# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- `InvalidArgumentError` for a bad per-call `k` or `as_of`. Unknown constructor arguments raise.
