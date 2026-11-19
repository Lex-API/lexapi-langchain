"""The README's Python examples run, against the offline stub."""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
from typing import Any

import lexapi
import pytest
from langchain_core.tools import ToolException

import langchain_lexapi._clients as clients_module
from conftest import Stub, load, offline_clients

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)


def legislation_with_unnamed_act() -> Any:
    """The mixed fixture plus a consolidated hit whose act is not named, so
    it has no ``cite_celex``."""
    body = load("legislation-semantic.mixed")
    unnamed = dict(body["results"][0])
    unnamed.pop("baseCelex")
    unnamed["article_ref"] = "art_18"
    body["results"].insert(0, unnamed)
    return body


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Stub:
    """Clients the package builds for itself answer from the stub."""
    stub = Stub(
        routes={
            "/legislation/semantic": (200, legislation_with_unnamed_act()),
            "/search/semantic": (200, load("case-law.with-text")),
            "/documentContent": (200, load("document.read")),
            "/legislation/as-of": (200, load("as-of.resolved")),
        }
    )

    def sync_client(*args: Any, **kwargs: Any) -> Any:
        return offline_clients(stub)[0]

    def async_client(*args: Any, **kwargs: Any) -> Any:
        return offline_clients(stub)[1]

    monkeypatch.setattr(clients_module, "LexAPI", sync_client)
    monkeypatch.setattr(clients_module, "AsyncLexAPI", async_client)
    return stub


def test_readme_has_four_examples() -> None:
    assert len(BLOCKS) == 4


@pytest.mark.parametrize("index", range(3))
def test_readme_example_runs(stub: Stub, index: int) -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(BLOCKS[index], f"README example {index + 1}", "exec"), {})
    assert out.getvalue()
    assert stub.requests, "the example made no request"
    assert "MUST-NOT-LEAK" not in out.getvalue()
    if index == 0:
        assert out.getvalue().splitlines()[2].startswith("None ")  # no cite_celex


CITED_BY = "/citations/cited-by/32099R0001"


def test_readme_tool_error_example(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``@tool`` example: a client the example builds, answering from the stub."""
    monkeypatch.setattr(lexapi, "LexAPI", lambda *args, **kwargs: offline_clients(stub)[0])
    namespace: dict[str, Any] = {}
    exec(compile(BLOCKS[3], "README example 4", "exec"), namespace)
    cited_by = namespace["lexapi_cited_by"]

    page = {
        "success": True,
        "targetDocument": "32099R0001",
        "citedBy": [{"id": "32099R0002", "citations": []}, {"id": "32099R0003", "citations": []}],
    }
    stub.routes[CITED_BY] = (200, page)
    assert cited_by.invoke({"celex": "32099R0001"}) == "32099R0002\n32099R0003"

    stub.routes[CITED_BY] = (402, load("error-402"))
    with pytest.raises(ToolException) as info:
        cited_by.invoke({"celex": "32099R0001"})
    assert str(info.value) == (
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. "
        "Credits reset at 2099-07-01T00:00:00.000Z."
    )
    assert isinstance(info.value.__cause__, lexapi.CreditsExhaustedError)
    assert [r.url.path for r in stub.requests] == ["/api/v1" + CITED_BY] * 2
