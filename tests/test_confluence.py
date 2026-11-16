import httpx
import pytest

from ragu.config import ROOT_PAGE_ID, ConfluenceCloudConfig
from ragu.confluence import ConfluenceSource, clean_html
from ragu.confluence_cloud import ConfluenceCloudSource
from ragu.errors import AppError


def page(page_id, ancestors):
    return {
        "id": page_id,
        "title": "Project page",
        "space": {"key": "GDPID"},
        "ancestors": [{"id": item} for item in ancestors],
        "version": {"number": 1, "when": "2026-11-10"},
        "body": {"storage": {"value": "<h1>Retries</h1><p>Use backoff.</p>"}},
    }


def test_source_rejects_pages_outside_the_root_tree(env):
    runtime, *_ = env
    source = ConfluenceSource(
        runtime.settings,
        httpx.MockTransport(lambda request: httpx.Response(200, json=page("999", ["888"]))),
    )
    with pytest.raises(AppError, match="outside"):
        source.page("999", "personal-token")
    source.client.close()


def test_source_crawls_only_children_and_preserves_partial_failure(env):
    runtime, *_ = env
    requested = []

    def handle(request):
        requested.append(request.url.path)
        assert request.url.host == "confluence.weareplanet.com"
        assert request.headers["authorization"] == "Bearer personal-token"
        if request.url.path == f"/rest/api/content/{ROOT_PAGE_ID}":
            return httpx.Response(200, json=page(ROOT_PAGE_ID, []))
        if request.url.path == f"/rest/api/content/{ROOT_PAGE_ID}/child/page":
            return httpx.Response(
                200,
                json={
                    "results": [page("123", [ROOT_PAGE_ID])],
                    "_links": {},
                },
            )
        return httpx.Response(200, json={"results": [], "_links": {}})

    source = ConfluenceSource(runtime.settings, httpx.MockTransport(handle))
    pages = source.crawl("personal-token", lambda: None)
    assert {item.id for item in pages} == {ROOT_PAGE_ID, "123"}
    assert requested == [
        f"/rest/api/content/{ROOT_PAGE_ID}",
        f"/rest/api/content/{ROOT_PAGE_ID}/child/page",
        "/rest/api/content/123/child/page",
    ]
    source.client.close()
    source = ConfluenceSource(
        runtime.settings,
        httpx.MockTransport(lambda request: httpx.Response(200, json=page(ROOT_PAGE_ID, []))),
    )
    with pytest.raises(AppError, match="Incomplete"):
        source.crawl("personal-token", lambda: None)
    source.client.close()


def test_source_does_not_follow_redirects_or_accept_unavailable_pages(env):
    runtime, *_ = env
    for status in [302, 401, 403, 404, 429, 500]:
        source = ConfluenceSource(
            runtime.settings,
            httpx.MockTransport(
                lambda request, code=status: httpx.Response(
                    code, headers={"Location": "https://attacker.example/"}
                )
            ),
        )
        with pytest.raises(AppError):
            source.page(ROOT_PAGE_ID, "personal-token")
        source.client.close()


@pytest.mark.parametrize("operation", ["page", "crawl"])
def test_cloud_source_stub_fails_explicitly_without_http_requests(monkeypatch, operation):
    def unexpected_http_client(*args, **kwargs):
        raise AssertionError("The unwired Cloud stub must not create an HTTP client.")

    monkeypatch.setattr(httpx, "Client", unexpected_http_client)
    source = ConfluenceCloudSource(
        ConfluenceCloudConfig(
            site_url="https://example.atlassian.net",
            account_email="fixture@example.invalid",
            api_token="synthetic-api-token",
            space_id="101",
            root_page_id="202",
        )
    )
    with pytest.raises(NotImplementedError, match="TODO: Implement"):
        if operation == "page":
            source.page("202")
        else:
            source.crawl(lambda: None)


def test_cleaning_preserves_headings_tables_and_code():
    html = (
        "<nav>menu</nav><h2>Retry policy</h2><table><tr><th>Error</th><th>Action</th></tr>"
        "<tr><td>503</td><td>Retry</td></tr></table><pre>retry()</pre><script>bad()</script>"
    )
    text = clean_html(html)
    assert "## Retry policy" in text
    assert "503 | Retry" in text
    assert "```\nretry()\n```" in text
    assert "bad()" not in text and "menu" not in text
