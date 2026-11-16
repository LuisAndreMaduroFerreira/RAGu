import hashlib
import json
import logging
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid5

from ragu.errors import AppError
from ragu.file_utils import FileUtils
from ragu.html import read_html_document

logger = logging.getLogger("ragu.local_consumption")


@dataclass(frozen=True)
class LocalDocument:
    id: str
    title: str
    version: str
    text: str
    updated_at: str
    relative_path: str
    url: str
    size_bytes: int
    aliases: tuple[str, ...]
    modified_ns: int
    candidates: tuple[str, ...]
    validated_size_bytes: int


class LocalConsumptionSource:
    extensions = frozenset({".md", ".html", ".htm", ".txt"})

    def __init__(self, settings):
        self.settings = settings

    @property
    def scope(self) -> str:
        if self.settings.local_consumption_root is None:
            raise AppError(503, "Configure the absolute RAGU_LOCAL_CONSUMPTION_ROOT first.")
        normalized = os.path.normcase(os.path.abspath(self.settings.local_consumption_root))
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def _no_links(path: Path) -> None:
        info = path.lstat()
        if path.is_symlink() or path.is_junction():
            raise AppError(409, "Local sources cannot contain symbolic links or junctions.")
        if getattr(info, "st_reparse_tag", 0) in {0xA000000C, 0xA0000003}:
            raise AppError(409, "Local sources cannot contain redirected filesystem links.")

    def _root(self) -> Path:
        if self.settings.local_consumption_root is None:
            raise AppError(503, "Configure the absolute RAGU_LOCAL_CONSUMPTION_ROOT first.")
        root = self.settings.local_consumption_root
        self._no_links(root)
        if not root.is_dir():
            raise OSError("Local source root is not a directory")
        return root.resolve(strict=True)

    def _path(self, root: Path, relative: str) -> Path:
        logical = PurePosixPath(relative)
        if (
            logical.is_absolute()
            or not logical.parts
            or any(part in {"..", "."} or "\\" in part or ":" in part for part in logical.parts)
        ):
            raise AppError(404, "Local document not found.")
        path = root
        for part in logical.parts:
            path = path / part
            self._no_links(path)
        if path.suffix.casefold() not in self.extensions or not path.resolve(
            strict=True
        ).is_relative_to(root):
            raise AppError(404, "Local document not found.")
        return path

    @staticmethod
    def _verify_handle(handle, root: Path) -> None:
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            function = ctypes.windll.kernel32.GetFinalPathNameByHandleW
            function.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
            function.restype = wintypes.DWORD
            buffer = ctypes.create_unicode_buffer(32768)
            length = function(msvcrt.get_osfhandle(handle.fileno()), buffer, len(buffer), 0)
            if not length or length >= len(buffer):
                raise OSError("Cannot verify opened local document")
            value = buffer.value
            if value.startswith("\\\\?\\UNC\\"):
                value = "\\\\" + value[8:]
            elif value.startswith("\\\\?\\"):
                value = value[4:]
            if not Path(value).is_relative_to(root):
                raise AppError(409, "Opened document escaped the configured local root.")
        elif Path("/proc/self/fd").is_dir():
            actual = Path(os.readlink(Path("/proc/self/fd") / str(handle.fileno())))
            if not actual.is_relative_to(root):
                raise AppError(409, "Opened document escaped the configured local root.")

    def _read(self, root: Path, relative: str) -> LocalDocument:
        path = self._path(root, relative)
        if not path.is_file():
            raise AppError(422, "Local sources must be regular files.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as handle:
            self._verify_handle(handle, root)
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
                raise AppError(409, "Local sources cannot contain special files or hard links.")
            limit = self.settings.local_consumption_max_file_bytes
            if before.st_size > limit:
                raise AppError(413, "A local document exceeds the configured size limit.")
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        if len(data) > limit:
            raise AppError(413, "A local document exceeds the configured size limit.")
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AppError(409, "Local files changed during scanning; retry ingestion.")
        text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if "\x00" in text:
            raise AppError(422, "A local document contains invalid binary content.")
        fallback_title = path.stem
        if path.suffix.casefold() in {".html", ".htm"}:
            html_title, text = read_html_document(text)
            fallback_title = html_title or fallback_title
        title = next(
            (
                line[2:].strip()
                for line in text.splitlines()
                if line.startswith("# ") and line[2:].strip()
            ),
            fallback_title,
        )
        digest = hashlib.sha256(text.encode()).hexdigest()
        # Keep the original namespace so existing indexes and document IDs remain valid.
        identifier = str(uuid5(NAMESPACE_URL, f"markdown:{self.scope}:{digest}"))
        return LocalDocument(
            identifier,
            title,
            digest,
            text,
            datetime.fromtimestamp(after.st_mtime, UTC).isoformat(),
            relative,
            f"{self.settings.public_url}/documents/{identifier}?source=local_consumption",
            len(data),
            (relative,),
            after.st_mtime_ns,
            (relative,),
            len(data),
        )

    def _current_version(self, root: Path, document: LocalDocument) -> LocalDocument:
        candidates = [document]
        total = document.size_bytes
        if total > self.settings.local_consumption_max_total_bytes:
            raise AppError(413, "Local source exceeds the configured scan limits.")
        logical = PurePosixPath(document.relative_path)
        key = FileUtils.version_key(document.relative_path)
        parent = root.joinpath(*logical.parts[:-1])
        for sibling in parent.iterdir():
            relative = sibling.relative_to(root).as_posix()
            if (
                os.path.normcase(relative) == os.path.normcase(document.relative_path)
                or sibling.suffix.casefold() not in self.extensions
                or FileUtils.version_key(relative) != key
            ):
                continue
            self._no_links(sibling)
            if sibling.is_dir():
                continue
            candidate = self._read(root, relative)
            candidates.append(candidate)
            total += candidate.size_bytes
            if (
                len(candidates) > self.settings.max_pages
                or total > self.settings.local_consumption_max_total_bytes
            ):
                raise AppError(413, "Local source exceeds the configured scan limits.")
        return FileUtils.newest_version(candidates)

    def document(self, relative: str) -> LocalDocument:
        root = None
        try:
            root = self._root()
            return self._current_version(root, self._read(root, relative))
        except FileNotFoundError as exc:
            if root is None or not root.is_dir():
                logger.warning("local_consumption_root_unavailable")
                raise AppError(503, "Local source is unavailable or unreadable.") from None
            raise AppError(404, "Local document is no longer available.") from exc
        except (OSError, UnicodeError, RuntimeError) as exc:
            logger.warning("local_consumption_read_failed reason=%s", type(exc).__name__)
            raise AppError(503, "Local source is unavailable or unreadable.") from None

    def scan(self, heartbeat: Callable[[], None]) -> list[LocalDocument]:
        try:
            root = self._root()
            documents = []
            total = 0

            def unavailable(error):
                raise error

            for directory, children, names in os.walk(root, followlinks=False, onerror=unavailable):
                heartbeat()
                for child in children:
                    self._no_links(Path(directory) / child)
                for name in sorted(names):
                    if Path(name).suffix.casefold() not in self.extensions:
                        continue
                    relative = (Path(directory) / name).relative_to(root).as_posix()
                    document = self._read(root, relative)
                    documents.append(document)
                    total += document.size_bytes
                    if (
                        len(documents) > self.settings.max_pages
                        or total > self.settings.local_consumption_max_total_bytes
                    ):
                        raise AppError(413, "Local source exceeds the configured scan limits.")
                    heartbeat()
            selected = []
            for versions in FileUtils.group_versions(documents).values():
                newest = FileUtils.newest_version(versions)
                # Preserve all candidates for counts/reuse, but only selected paths are read aliases.
                selected.append(
                    replace(
                        newest,
                        candidates=tuple(sorted(item.relative_path for item in versions)),
                        validated_size_bytes=sum(item.validated_size_bytes for item in versions),
                    )
                )
            grouped = {}
            for document in sorted(selected, key=lambda item: item.relative_path):
                grouped.setdefault(document.version, []).append(document)
            return sorted(
                (
                    replace(
                        copies[0],
                        aliases=tuple(sorted(path for item in copies for path in item.aliases)),
                        candidates=tuple(
                            sorted(path for item in copies for path in item.candidates)
                        ),
                        validated_size_bytes=sum(item.validated_size_bytes for item in copies),
                    )
                    for copies in grouped.values()
                ),
                key=lambda document: document.id,
            )
        except (OSError, UnicodeError, RuntimeError) as exc:
            logger.warning("local_consumption_scan_failed reason=%s", type(exc).__name__)
            raise AppError(503, "Local scan failed; the published index was preserved.") from None

    @staticmethod
    def fingerprint(documents: list[LocalDocument]) -> str:
        values = {document.id: document.version for document in documents}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
