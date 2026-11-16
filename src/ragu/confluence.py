import ssl
from dataclasses import dataclass

import httpx

from ragu.config import CONFLUENCE_URL, ROOT_PAGE_ID
from ragu.errors import AppError
from ragu.html import clean_html


@dataclass(frozen=True)
class Page:
    id: str
    title: str
    version: int
    text: str
    updated_at: str

    @property
    def url(self) -> str:
        return f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={self.id}"


class ConfluenceSource:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.client = httpx.Client(
            base_url=CONFLUENCE_URL,
            verify=ssl.create_default_context(
                cafile=str(settings.confluence_ca_bundle) if settings.confluence_ca_bundle else None
            ),
            follow_redirects=False,
            timeout=30,
            transport=transport,
        )

    def _get(self, path: str, token: str, params=None) -> dict:
        try:
            with self.client.stream(
                "GET",
                path,
                params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            ) as response:
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 5_000_000:
                        raise AppError(503, "Confluence response exceeds the safe size limit.")
        except httpx.HTTPError as exc:
            raise AppError(
                503, "Confluence is unavailable; content access cannot be verified."
            ) from exc
        if response.status_code in {401, 403, 404}:
            raise AppError(404, "Confluence content is unavailable to this credential.")
        if response.status_code != 200:
            raise AppError(503, "Confluence returned an unsuccessful response.")
        try:
            import json

            value = json.loads(body)
        except ValueError as exc:
            raise AppError(503, "Confluence returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise AppError(503, "Confluence returned an invalid response shape.")
        return value

    def _page(self, data: dict) -> Page:
        try:
            page_id = str(data["id"])
            ancestors = data["ancestors"]
            if (
                not page_id.isdigit()
                or data["space"]["key"] != "GDPID"
                or (
                    page_id != ROOT_PAGE_ID
                    and not any(str(parent["id"]) == ROOT_PAGE_ID for parent in ancestors)
                )
            ):
                raise AppError(404, "Page is outside the configured project.")
            version = data["version"]["number"]
            if type(version) is not int or version < 1:
                raise ValueError("Invalid version")
            return Page(
                page_id,
                data["title"],
                version,
                clean_html(data["body"]["storage"]["value"]),
                data["version"].get("when", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(
                503, "Confluence page is missing required content or access metadata."
            ) from exc

    def page(self, page_id: str, token: str) -> Page:
        if not page_id.isdigit():
            raise AppError(404, "Document not found.")
        return self._page(
            self._get(
                f"/rest/api/content/{page_id}",
                token,
                {"expand": "body.storage,version,space,ancestors"},
            )
        )

    def crawl(self, token: str, heartbeat) -> list[Page]:
        root = self.page(ROOT_PAGE_ID, token)
        pages = {root.id: root}
        pending = [root.id]
        for parent_id in pending:
            start = 0
            while True:
                heartbeat()
                data = self._get(
                    f"/rest/api/content/{parent_id}/child/page",
                    token,
                    {"expand": "body.storage,version,space,ancestors", "start": start, "limit": 50},
                )
                results = data.get("results")
                if not isinstance(results, list) or not isinstance(data.get("_links"), dict):
                    raise AppError(
                        503, "Incomplete Confluence listing; the old index is preserved."
                    )
                for item in results:
                    page = self._page(item)
                    if page.id in pages:
                        raise AppError(
                            503, "Confluence tree changed during traversal; retry ingestion."
                        )
                    pages[page.id] = page
                    pending.append(page.id)
                    if len(pages) > self.settings.max_pages:
                        raise AppError(503, "Project exceeds the configured page limit.")
                if not data["_links"].get("next"):
                    break
                if not results:
                    raise AppError(503, "Confluence pagination did not advance.")
                start += len(results)
        return list(pages.values())
