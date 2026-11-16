import os
import re
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath
from typing import Protocol, TypeVar

from ragu.errors import AppError


class FileVersion(Protocol):
    @property
    def relative_path(self) -> str: ...

    @property
    def modified_ns(self) -> int: ...

    @property
    def version(self) -> str: ...


T = TypeVar("T", bound=FileVersion)


class FileUtils:
    _version_suffix = re.compile(
        r"(?:\s*\((?:\d+|copy(?:[ _-]*\d+)?|v(?:ersion)?[ _-]*\d+(?:\.\d+)*)\)"
        r"|[ _-]+(?:copy(?:[ _-]*\d+)?|v(?:ersion)?[ _-]*\d+(?:\.\d+)*))$",
        re.IGNORECASE,
    )

    @classmethod
    def version_key(cls, relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        stem = path.stem
        while match := cls._version_suffix.search(stem):
            base = stem[: match.start()].rstrip(" _-")
            if not base:
                break
            stem = base
        return os.path.normcase(str(path.parent / stem))

    @classmethod
    def group_versions(cls, documents: Iterable[T]) -> dict[str, list[T]]:
        groups: dict[str, list[T]] = {}
        for document in documents:
            groups.setdefault(cls.version_key(document.relative_path), []).append(document)
        return groups

    @staticmethod
    def newest_version(documents: Sequence[T]) -> T:
        newest = max(document.modified_ns for document in documents)
        tied = [document for document in documents if document.modified_ns == newest]
        if len({document.version for document in tied}) != 1:
            raise AppError(
                409,
                "Local file versions have different content but the same latest modification "
                "time. Resolve the timestamp conflict before retrying.",
            )
        return min(tied, key=lambda document: document.relative_path)
