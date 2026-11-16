from collections.abc import Callable
from typing import NoReturn

from ragu.config import ConfluenceCloudConfig


class ConfluenceCloudSource:
    """TODO: Implement and tenant-test Cloud reads before wiring any controller or worker."""

    def __init__(self, connection: ConfluenceCloudConfig):
        self.connection = connection

    def page(self, page_id: str) -> NoReturn:
        raise NotImplementedError(
            "TODO: Implement Confluence Cloud page retrieval with live permission checks."
        )

    def crawl(self, heartbeat: Callable[[], None]) -> NoReturn:
        raise NotImplementedError(
            "TODO: Implement root-scoped Confluence Cloud traversal and validated pagination."
        )
