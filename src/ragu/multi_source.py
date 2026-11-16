import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ragu.auth import Principal
from ragu.confluence import Page
from ragu.database import MultiSourceIndexState, Role, User, retire_index, utcnow
from ragu.errors import AppError
from ragu.import_cache import ImportCache
from ragu.local_consumption import LocalConsumptionSource
from ragu.schemas import FolderInput, IndexMode, MultiSourceRequest
from ragu.strategies import IndexScope

logger = logging.getLogger("ragu.multi_source")


class ConfluencePageProvider(Protocol):
    def page(self, url: str, principal: Principal) -> Page: ...


class MockConfluenceProvider:
    """Explicit test/demo injection only. Never performs network requests."""

    def __init__(self, pages: dict[str, Page]):
        self.pages = pages

    def page(self, url: str, principal: Principal) -> Page:
        if url not in self.pages:
            raise AppError(404, "Mock Confluence page is unavailable.")
        return self.pages[url]


@dataclass(frozen=True)
class ImportDocument:
    identifier: str
    text: str
    metadata: dict


class StaleImportDocument(AppError):
    pass


class MultiSourceStrategy(IndexScope):
    source = storage_source = collection_prefix = "multi_source"
    index_model = MultiSourceIndexState

    def __init__(self, db, auth, settings, vectors, *, confluence=None):
        self.db, self.auth, self.settings, self.vectors = db, auth, settings, vectors
        self.confluence: ConfluencePageProvider | None = confluence
        self.cache_factory = lambda job_id: ImportCache(settings, job_id)

    def bind(self, session, owner_id):
        raise AppError(422, "Use POST /ingestions/multi-source with an explicit sources list.")

    def schedule_valid(self, session, schedule):
        return False

    def _folder(self, path: Path) -> LocalConsumptionSource:
        roots = self.settings.multi_source_roots or (
            [self.settings.local_consumption_root] if self.settings.local_consumption_root else []
        )
        try:
            for allowed in roots:
                if not path.is_relative_to(allowed):
                    continue
                source = LocalConsumptionSource(
                    self.settings.model_copy(update={"local_consumption_root": allowed})
                )
                resolved_allowed = source._root()
                current = allowed
                for part in path.relative_to(allowed).parts:
                    current /= part
                    source._no_links(current)
                source = LocalConsumptionSource(
                    self.settings.model_copy(update={"local_consumption_root": path})
                )
                if not source._root().is_relative_to(resolved_allowed):
                    raise AppError(403, "The folder is outside the approved import roots.")
                return source
        except (OSError, RuntimeError):
            logger.warning("multi_source_folder_unavailable")
            raise AppError(503, "An approved import folder is unavailable.") from None
        raise AppError(403, "The folder is outside the approved import roots.")

    def _confluence_url(self, url: str) -> None:
        try:
            parsed = urlsplit(url)
        except ValueError:
            raise AppError(422, "Invalid mock Confluence page URL.") from None
        origin = urlsplit(self.settings.mock_confluence_origin)
        if (
            parsed.scheme != "https"
            or parsed.netloc.casefold() != origin.netloc.casefold()
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"/wiki/spaces/[^/]+/pages/[0-9]+(?:/[^/]+)?", parsed.path)
        ):
            raise AppError(
                422, "Only page links on the configured mock Confluence origin are allowed."
            )
        if self.confluence is None:
            raise AppError(
                503, "Confluence is disabled; only an explicitly injected mock is supported."
            )

    def _binding(self, request: MultiSourceRequest, mode=IndexMode.DENSE, *, legacy=False) -> str:
        seen = set()
        for item in request.sources:
            if isinstance(item, FolderInput):
                source = self._folder(item.path)
                identity = ("folder", source.scope)
            else:
                self._confluence_url(item.url)
                identity = ("confluence", item.url)
            if identity in seen:
                raise AppError(422, "The sources list contains the same source more than once.")
            seen.add(identity)
        return hashlib.sha256(
            json.dumps(
                {
                    "sources": request.model_dump(mode="json"),
                    "model": self.settings.embedding_model
                    if legacy or mode != IndexMode.SPARSE
                    else None,
                    "roots": [
                        os.path.normcase(str(root)) for root in self.settings.multi_source_roots
                    ],
                    "default_root": str(self.settings.local_consumption_root),
                    "mock_origin": self.settings.mock_confluence_origin,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def bind_request(self, session, owner_id, request: MultiSourceRequest) -> str:
        state = self._owner(session, owner_id, write=True)
        if self.settings.import_redis_url is None:
            raise AppError(503, "Configure RAGU_IMPORT_REDIS_URL for multi-source imports.")
        mode = IndexMode(request.index_mode or (state.index_mode if state else "dense"))
        binding = self._binding(request, mode)
        if self.corpus_id is not None:
            sources = request.model_dump(mode="json")["sources"]
            approved = state.configuration.get("sources")
            if approved is None:
                # Source approval is administrative; delegated writers cannot widen a corpus.
                user = session.get(User, owner_id)
                role = session.get(Role, user.role_id)
                if role is None or role.code != "admin":
                    raise AppError(403, "An administrator must first bind this corpus's sources.")
                state.configuration = {**state.configuration, "sources": sources}
                logger.info(
                    "corpus_sources_bound corpus_id=%s actor_id=%s", self.corpus_id, owner_id
                )
            elif approved != sources:
                raise AppError(422, "sources must match the corpus's approved sources list.")
        return binding

    def _binding_matches(self, request, job):
        mode = IndexMode(job.index_mode)
        if self._binding(request, mode) == job.source_root:
            return True
        # Pre-fix sparse jobs retain their authorized fingerprint when its inputs still match.
        return mode == IndexMode.SPARSE and (
            self._binding(request, mode, legacy=True) == job.source_root
        )

    def _owner(self, session, owner_id, *, write=False):
        return self._owner_access(session, owner_id, write=write)

    def _collect(self, request, principal, heartbeat):
        documents = []
        files = total = 0
        for item in request.sources:
            heartbeat()
            if isinstance(item, FolderInput):
                source = self._folder(item.path)
                selected = source.scan(heartbeat)
                files += sum(len(document.candidates) for document in selected)
                entries = [
                    (
                        f"folder:{source.scope}:{document.id}",
                        document.text,
                        {
                            "kind": "folder",
                            "path": str(item.path),
                            "paths": list(document.aliases),
                            "title": document.title,
                            "relative_path": document.relative_path,
                            "digest": document.version,
                        },
                        document.validated_size_bytes,
                    )
                    for document in selected
                ]
            else:
                self._confluence_url(item.url)
                page = self.confluence.page(item.url, principal)
                if not isinstance(page.text, str) or not isinstance(page.title, str):
                    raise AppError(503, "Mock Confluence returned invalid content.")
                files += 1
                entries = [
                    (
                        f"confluence:{item.url}",
                        page.text,
                        {
                            "kind": "confluence",
                            "url": item.url,
                            "title": page.title,
                            "digest": hashlib.sha256(page.text.encode()).hexdigest(),
                        },
                        len(page.text.encode()),
                    )
                ]
            for identity, text, metadata, size_bytes in entries:
                total += size_bytes
                if (
                    total > self.settings.local_consumption_max_total_bytes
                    or files > self.settings.max_pages
                    or len(documents) >= self.settings.max_pages
                ):
                    raise AppError(413, "Multi-source import exceeds the configured scan limits.")
                if len(text.encode()) > self.settings.local_consumption_max_file_bytes:
                    raise AppError(413, "An import document exceeds the configured size limit.")
                documents.append(
                    ImportDocument(hashlib.sha256(identity.encode()).hexdigest(), text, metadata)
                )
        return documents, files

    def _chunk_profile(self, mode):
        return (
            {"model": self.settings.sparse_model, "language": self.settings.sparse_language}
            if mode == IndexMode.SPARSE
            else {"model": self.settings.embedding_model}
        )

    def _chunk_id(self, text, mode=IndexMode.DENSE):
        digest = hashlib.sha256(
            json.dumps(
                {**self._chunk_profile(mode), "role": "document", "input": text},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return str(uuid5(NAMESPACE_URL, f"{self.source}:{digest}"))

    def _write(self, collection, passages, cache, dimension, mode=IndexMode.DENSE):
        unique = {self._chunk_id(text, mode): text for text in passages}
        profile = self._chunk_profile(mode)
        known = cache.known(list(unique))
        missing_ids = [identifier for identifier in unique if identifier not in known]
        # Redis can evict markers; deterministic IDs and Qdrant remain authoritative.
        existing = (
            self.vectors.client.retrieve(
                collection_name=collection, ids=missing_ids, with_payload=True, with_vectors=False
            )
            if missing_ids
            else []
        )
        for point in existing:
            if point.payload.get("text") != unique[str(point.id)] or any(
                point.payload.get(key) != value for key, value in profile.items()
            ):
                raise AppError(503, "An existing canonical chunk has inconsistent content.")
        present = {str(point.id) for point in existing}
        missing = [identifier for identifier in missing_ids if identifier not in present]
        vectors = self.vectors.encode([unique[identifier] for identifier in missing], mode)
        if len(vectors) != len(missing):
            raise AppError(503, "Embedding model returned an incomplete batch.")
        points = []
        for identifier, vector in zip(missing, vectors, strict=True):
            values = self.vectors.dense_values(vector)
            if values is not None and (
                len(values) != dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                )
            ):
                raise AppError(503, "Embedding model returned an invalid vector.")
            points.append(
                models.PointStruct(
                    id=identifier,
                    vector=vector,
                    payload={
                        "root_id": self.source,
                        **profile,
                        "text": unique[identifier],
                    },
                )
            )
        if points:
            self.vectors.client.upsert(collection_name=collection, points=points, wait=True)
        cache.remember(missing_ids)
        return [self._chunk_id(text, mode) for text in passages], len(points)

    def ingest(self, job, heartbeat, authorize):
        request = MultiSourceRequest.model_validate(job.parameters)
        if not self._binding_matches(request, job):
            raise AppError(409, "Import sources or configuration changed after authorization.")
        with self.db.session() as session:
            authorize(session, job.id)
            self._owner(session, job.requested_by, write=True)
            user = session.get(User, job.requested_by)
            role = session.get(Role, user.role_id)
            principal = Principal(user.id, user.username, role.code, user.auth_version, frozenset())
        documents, files = self._collect(request, principal, heartbeat)
        with self.db.session() as session:
            authorize(session, job.id)
        if not self._binding_matches(request, job):
            raise AppError(409, "Import configuration changed during validation.")
        collection = f"ragu_{self.collection_prefix}_{job.id.hex}"
        manifest = {}
        references = unique_count = 0
        try:
            with self.cache_factory(job.id) as cache:
                mode = IndexMode(job.index_mode)
                self.vectors.create(collection, mode=mode)
                params = self.vectors.client.get_collection(collection).config.params.vectors
                dimension = (
                    params.size
                    if isinstance(params, models.VectorParams)
                    else params["dense"].size
                    if "dense" in params
                    else None
                )
                for document in documents:
                    heartbeat()
                    passages = self.vectors.splitter.split_text(document.text)
                    references += len(passages)
                    if references > self.settings.max_chunks:
                        raise AppError(413, "Import exceeds the configured chunk-reference limit.")
                    identifiers = []
                    for offset in range(0, len(passages), 100):
                        heartbeat()
                        chunk_ids, created = self._write(
                            collection, passages[offset : offset + 100], cache, dimension, mode
                        )
                        identifiers.extend(chunk_ids)
                        unique_count += created
                    manifest[document.identifier] = {**document.metadata, "chunks": identifiers}
        except (ResponseHandlingException, UnexpectedResponse):
            logger.warning("multi_source_index_write_failed")
            raise AppError(503, "The document index is unavailable.") from None
        if not self._binding_matches(request, job):
            raise AppError(409, "Import configuration changed during indexing.")
        with self.db.session() as session:
            current = authorize(session, job.id)
            state = self._owner(session, job.requested_by, write=True)
            if state is None:
                state = self.index_model(id=1, owner_id=job.requested_by)
                session.add(state)
            retire_index(session, state.collection)
            state.collection = collection
            state.model = self.settings.embedding_model
            state.manifest, state.published_at = manifest, utcnow()
            self._publication(state, job)
            current.status, current.finished_at = "succeeded", utcnow()
            current.page_count, current.file_count = len(documents), files
            current.duplicate_count = files - len(documents)
            current.chunk_count = unique_count
            current.chunk_reference_count = references
            current.reused_chunk_count = references - unique_count
        logger.info("multi_source_import_published job_id=%s chunks=%s", job.id, unique_count)

    def _context(self, principal):
        self.auth.throttle("search", str(principal.id), 60, 60)
        with self.db.session() as session:
            self._access(session, principal)
            state = self._owner(session, principal.id)
            if state is None or state.collection is None:
                raise AppError(409, "No published multi-source import.")
            self.validate_profile(state)
            return state.collection, dict(state.manifest)

    def _page(self, identifier, entry, principal):
        if entry["kind"] == "folder":
            source = self._folder(Path(entry["path"]))
            for path in entry["paths"]:
                try:
                    page = source.document(path)
                except AppError as exc:
                    if exc.status == 404:
                        continue
                    raise
                if page.version == entry["digest"]:
                    return {
                        "page_id": identifier,
                        "title": page.title,
                        "text": page.text,
                        "relative_path": page.relative_path,
                        "updated_at": page.updated_at,
                        "version": page.version,
                        "source_url": self._document_url(
                            f"{self.settings.public_url}/documents/{identifier}?source={self.source}"
                        ),
                        "source": self.source,
                        "source_kind": "folder",
                    }
        else:
            self._confluence_url(entry["url"])
            try:
                page = self.confluence.page(entry["url"], principal)
            except AppError as exc:
                if exc.status == 404:
                    raise StaleImportDocument(
                        404, "Source document is no longer available."
                    ) from exc
                raise
            if hashlib.sha256(page.text.encode()).hexdigest() == entry["digest"]:
                return {
                    "page_id": identifier,
                    "title": page.title,
                    "text": page.text,
                    "version": entry["digest"],
                    "updated_at": page.updated_at,
                    "source_url": entry["url"],
                    "source": self.source,
                    "source_kind": "confluence",
                }
        raise StaleImportDocument(409, "Source document changed or disappeared; rerun the import.")

    def document(self, page_id, principal):
        _, manifest = self._context(principal)
        if page_id not in manifest:
            raise AppError(404, "Imported document not found.")
        return self._page(page_id, manifest[page_id], principal)

    def search(self, data, principal, *, candidate_ids=None, generation_id=None):
        collection, manifest = self._context(principal)
        bindings = {}
        for identifier, entry in manifest.items():
            for chunk in set(entry["chunks"]):
                bindings.setdefault(chunk, []).append(identifier)
        candidates = self._points(collection, data, self.source, candidate_ids, generation_id)
        checked = {}
        results = []
        for point in candidates:
            citations = []
            for identifier in bindings.get(str(point.id), []):
                if identifier not in checked:
                    try:
                        checked[identifier] = self._page(
                            identifier, manifest[identifier], principal
                        )
                    except StaleImportDocument:
                        checked[identifier] = None
                page = checked[identifier]
                if page is not None:
                    citations.append({key: value for key, value in page.items() if key != "text"})
            if citations:
                results.append(
                    {
                        **citations[0],
                        "passage": point.payload["text"],
                        "score": point.score,
                        "citations": citations,
                        "candidate_id": str(point.id),
                        "score_type": data.candidate_method.value,
                    }
                )
            if len(results) == data.limit:
                break
        return self._batch(results, collection)
