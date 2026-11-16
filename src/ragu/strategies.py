import logging
from collections.abc import Callable, Iterator
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from ragu.auth import Principal
from ragu.database import (
    Corpus,
    IndexState,
    Job,
    LocalConsumptionIndexState,
    MultiSourceIndexState,
    Schedule,
    retire_index,
    utcnow,
)
from ragu.errors import AppError
from ragu.file_utils import FileUtils
from ragu.schemas import DEFAULT_SOURCE, CandidateMethod, IndexMode, SearchQuery

logger = logging.getLogger("ragu.strategies")
AuthorizeJob = Callable[[Session, UUID], Job]


class RetrievalStrategy(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def storage_source(self) -> str: ...

    @property
    def index_model(
        self,
    ) -> (
        type[IndexState]
        | type[LocalConsumptionIndexState]
        | type[MultiSourceIndexState]
        | type[Corpus]
    ): ...

    @property
    def collection_prefix(self) -> str: ...

    def bind(self, session: Session, owner_id: UUID) -> str | None: ...

    def schedule_valid(self, session: Session, schedule: Schedule) -> bool: ...

    def ingest(self, job: Job, heartbeat: Callable[[], None], authorize: AuthorizeJob) -> None: ...

    def search(
        self,
        data: SearchQuery,
        principal: Principal,
        *,
        candidate_ids: list[str] | None = None,
        generation_id: UUID | None = None,
    ) -> dict: ...

    def authorize(self, principal: Principal) -> None: ...

    def document(self, page_id: str, principal: Principal) -> dict: ...


class StrategyRegistry:
    def __init__(self, *strategies: RetrievalStrategy):
        self._strategies = strategies
        self._by_source: dict[str, RetrievalStrategy] = {}
        for strategy in strategies:
            for name in {strategy.source, strategy.storage_source}:
                if name in self._by_source:
                    raise ValueError(f"Duplicate retrieval strategy: {name}")
                self._by_source[name] = strategy

    def __iter__(self) -> Iterator[RetrievalStrategy]:
        return iter(self._strategies)

    def get(self, source: str = DEFAULT_SOURCE) -> RetrievalStrategy:
        try:
            return self._by_source[source]
        except KeyError:
            raise AppError(422, "Unsupported retrieval source.") from None

    def collection_name(self, job_id: UUID, source: str) -> str:
        return f"ragu_{self.get(source).collection_prefix}_{job_id.hex}"


class _StaleLocalDocument(AppError):
    pass


class IndexScope:
    corpus_id = None
    index_key = 1
    read_requires_admin = True

    def configure_scope(self, corpus_id, projects):
        self.corpus_id, self.projects = corpus_id, projects
        self.index_model, self.index_key = Corpus, corpus_id
        self.collection_prefix = f"corpus_{corpus_id.hex}"

    def _state(self, session):
        return session.get(self.index_model, self.index_key)

    def _access(self, session, principal, *, admin=True):
        self.auth.current(session, principal, admin=admin and self.corpus_id is None)
        if self.corpus_id is not None:
            self.projects.corpus_access(session, self.corpus_id, principal.id)

    def _owner_access(self, session, user_id, *, write=False):
        state = self._state(session)
        if self.corpus_id is not None:
            self.projects.corpus_access(session, self.corpus_id, user_id, write=write)
        elif state is not None and state.owner_id != user_id:
            raise AppError(403, "The corpus belongs to another administrator.")
        return state

    def _publication(self, state, job):
        state.index_mode = job.index_mode
        state.index_profile = job.index_profile

    def profile_matches(self, state):
        mode = IndexMode(state.index_mode)
        return (mode == IndexMode.SPARSE or state.model == self.settings.embedding_model) and (
            state.index_profile is None or state.index_profile == self.vectors.profile(mode)
        )

    def validate_profile(self, state):
        if state.index_mode != IndexMode.SPARSE and state.model != self.settings.embedding_model:
            raise AppError(409, "Embedding model changed. Reindex before searching.")
        if not self.profile_matches(state):
            raise AppError(409, "Index profile changed. Reindex before searching.")

    def authorize(self, principal):
        with self.db.session() as session:
            self._access(session, principal, admin=self.read_requires_admin)
            if self.read_requires_admin:
                self._owner_access(session, principal.id)

    def _points(self, collection, data, root, candidate_ids, generation_id):
        if generation_id is not None and self._generation(collection) != str(generation_id):
            raise AppError(409, "Candidate generation was replaced; retrieve candidates again.")
        if candidate_ids is not None:
            return self.vectors.select(collection, candidate_ids, root)
        args = (collection, data.query, min(data.limit * 5, 100), root)
        if data.candidate_method == CandidateMethod.DENSE:
            return self.vectors.search(*args)
        return self.vectors.search(*args, method=data.candidate_method)

    @staticmethod
    def _generation(collection):
        return str(UUID(collection.rsplit("_", 1)[-1]))

    def _batch(self, results, collection):
        result = {
            "results": results,
            "collection": collection,
            "generation_id": self._generation(collection),
            "source": self.source,
        }
        if self.corpus_id is not None:
            result["corpus_id"] = str(self.corpus_id)
        return result

    def _document_url(self, url):
        return f"{url}&corpus_id={self.corpus_id}" if self.corpus_id is not None else url


class LocalConsumptionStrategy(IndexScope):
    source = "local_consumption"
    # Existing PostgreSQL constraints, Qdrant collections, and manifests stay compatible.
    storage_source = "markdown"
    collection_prefix = "markdown"
    index_model = LocalConsumptionIndexState

    def __init__(self, db, auth, settings, source, vectors):
        self.db, self.auth, self.settings = db, auth, settings
        self.files, self.vectors = source, vectors

    def bind(self, session: Session, owner_id: UUID) -> str:
        scope = self.files.scope
        self._owner_access(session, owner_id, write=True)
        return scope

    def schedule_valid(self, session: Session, schedule: Schedule) -> bool:
        state = self._state(session)
        return (
            self.files.settings.local_consumption_root is not None
            and schedule.source_root == self.files.scope
            and (self.corpus_id is not None or state is None or state.owner_id == schedule.owner_id)
        )

    @staticmethod
    def _manifest_entry(document, chunks):
        return {
            "path": document.relative_path,
            "paths": list(document.aliases),
            "candidates": list(document.candidates),
            "digest": document.version,
            "chunks": chunks,
        }

    def ingest(self, job: Job, heartbeat: Callable[[], None], authorize: AuthorizeJob) -> None:
        job_id, user_id, root = job.id, job.requested_by, job.source_root
        if root is None or root != self.files.scope:
            raise AppError(409, "Local consumption root changed after this job was authorized.")
        documents = self.files.scan(heartbeat)
        fingerprint = self.files.fingerprint(documents)
        file_count = sum(len(document.candidates) for document in documents)
        with self.db.session() as session:
            current = authorize(session, job_id)
            if self.files.scope != root:
                raise AppError(409, "Local consumption root changed during validation.")
            state = self._owner_access(session, user_id, write=True)
            if (
                state
                and state.source_root == root
                and state.fingerprint == fingerprint
                and state.index_mode == job.index_mode
                and self.profile_matches(state)
                and self.vectors.exists(state.collection)
            ):
                current.status, current.finished_at, current.unchanged = "succeeded", utcnow(), True
                current.page_count = len(documents)
                current.chunk_count = sum(entry["chunks"] for entry in state.manifest.values())
                current.file_count = file_count
                current.duplicate_count = file_count - len(documents)
                state.manifest = {
                    document.id: self._manifest_entry(
                        document, state.manifest[document.id]["chunks"]
                    )
                    for document in documents
                }
                logger.info(
                    "local_consumption_unchanged job_id=%s documents=%s", job_id, len(documents)
                )
                return
            previous = (
                state.collection
                if state and state.source_root == root and self.vectors.exists(state.collection)
                else None
            )
            previous_model = state.model if previous else None
            previous_manifest = dict(state.manifest) if previous else {}
        previous_groups = {
            FileUtils.version_key(path): identifier
            for identifier, entry in previous_manifest.items()
            for path in entry.get("candidates", entry["paths"])
        }
        collection = f"ragu_{self.collection_prefix}_{job_id.hex}"
        self.vectors.create(
            collection,
            previous if previous_model == self.settings.embedding_model else None,
            mode=IndexMode(job.index_mode),
        )
        manifest = {}
        chunks = 0
        for document in documents:
            heartbeat()
            previous_id = (
                document.id
                if document.id in previous_manifest
                else next(
                    (
                        previous_groups[FileUtils.version_key(path)]
                        for path in document.aliases
                        if FileUtils.version_key(path) in previous_groups
                    ),
                    None,
                )
            )
            count = self.vectors.index_page(
                collection,
                document,
                previous,
                previous_model,
                root,
                previous_id,
                heartbeat=heartbeat,
                mode=IndexMode(job.index_mode),
            )
            chunks += count
            if chunks > self.settings.max_chunks:
                raise AppError(503, "Index exceeds the configured chunk limit.")
            manifest[document.id] = self._manifest_entry(document, count)
        with self.db.session() as session:
            current = authorize(session, job_id)
            if self.files.scope != root:
                raise AppError(409, "Local consumption root changed during ingestion.")
            state = self._owner_access(session, user_id, write=True)
            if state is None:
                state = self.index_model(id=1, owner_id=user_id)
                session.add(state)
            retire_index(session, state.collection)
            state.source_root, state.collection, state.model = (
                root,
                collection,
                self.settings.embedding_model,
            )
            state.fingerprint, state.manifest, state.published_at = fingerprint, manifest, utcnow()
            self._publication(state, job)
            current.status, current.finished_at = "succeeded", utcnow()
            current.page_count, current.chunk_count = len(documents), chunks
            current.file_count, current.duplicate_count = file_count, file_count - len(documents)
        logger.info(
            "local_consumption_ingestion_succeeded job_id=%s documents=%s chunks=%s",
            job_id,
            len(documents),
            chunks,
        )

    def _context(self, principal: Principal):
        self.auth.throttle("search", str(principal.id), 60, 60)
        with self.db.session() as session:
            self._access(session, principal)
            state = self._owner_access(session, principal.id)
            if state is None or state.collection is None:
                raise AppError(
                    409, "No published local consumption index. Run local consumption first."
                )
            if state.source_root != self.files.scope:
                raise AppError(409, "Local consumption root changed. Reindex before searching.")
            self.validate_profile(state)
            return state.collection, state.source_root, dict(state.manifest)

    def _page(self, identifier: str, manifest: dict):
        entry = manifest.get(identifier)
        if entry is None:
            raise _StaleLocalDocument(404, "Indexed local consumption document not found.")
        changed = False
        checked_groups = set()
        for path in entry["paths"]:
            group = FileUtils.version_key(path)
            if group in checked_groups:
                continue
            try:
                page = self.files.document(path)
            except AppError as exc:
                if exc.status != 404:
                    raise
                continue
            checked_groups.add(group)
            if page.id == identifier and page.version == entry["digest"]:
                return page
            changed = True
        if changed:
            raise _StaleLocalDocument(
                409, "Local document changed since indexing. Rerun local consumption."
            )
        raise _StaleLocalDocument(404, "Local consumption document is no longer available.")

    def search(
        self, data: SearchQuery, principal: Principal, *, candidate_ids=None, generation_id=None
    ) -> dict:
        collection, root, manifest = self._context(principal)
        candidates = self._points(collection, data, root, candidate_ids, generation_id)
        checked = {}
        results = []
        for point in candidates:
            identifier = point.payload["page_id"]
            if identifier not in checked:
                try:
                    checked[identifier] = self._page(identifier, manifest)
                except _StaleLocalDocument:
                    checked[identifier] = None
            page = checked[identifier]
            if page is None or point.payload["version"] != page.version:
                continue
            results.append(
                {
                    "page_id": page.id,
                    "title": page.title,
                    "passage": point.payload["text"],
                    "source_url": self._document_url(page.url),
                    "relative_path": page.relative_path,
                    "updated_at": page.updated_at,
                    "version": page.version,
                    "score": point.score,
                    "source": self.source,
                    "candidate_id": str(point.id),
                    "score_type": data.candidate_method.value,
                }
            )
            if len(results) == data.limit:
                break
        return self._batch(results, collection)

    def document(self, page_id: str, principal: Principal) -> dict:
        _, _, manifest = self._context(principal)
        page = self._page(page_id, manifest)
        return {
            "page_id": page.id,
            "title": page.title,
            "text": page.text,
            "source_url": self._document_url(page.url),
            "relative_path": page.relative_path,
            "updated_at": page.updated_at,
            "version": page.version,
            "source": self.source,
        }


class ConfluenceRetrievalStrategy(IndexScope):
    read_requires_admin = False
    source = storage_source = "confluence"
    collection_prefix = "41276872"
    index_model = IndexState

    def __init__(self, db, auth, settings, source, vectors):
        self.db, self.auth, self.settings = db, auth, settings
        self.source_client, self.vectors = source, vectors

    def bind(self, session: Session, owner_id: UUID) -> None:
        if self.corpus_id is not None:
            self.projects.corpus_access(session, self.corpus_id, owner_id, write=True)
        if not self.settings.confluence_ingestion_token:
            raise AppError(503, "Configure the Confluence ingestion credential first.")

    def schedule_valid(self, session: Session, schedule: Schedule) -> bool:
        return True

    def ingest(self, job: Job, heartbeat: Callable[[], None], authorize: AuthorizeJob) -> None:
        if not self.settings.confluence_ingestion_token:
            raise AppError(503, "Confluence ingestion credential is missing.")
        with self.db.session() as session:
            previous = self._state(session)
            previous_collection = (
                previous.collection
                if previous and self.vectors.exists(previous.collection)
                else None
            )
            previous_model = previous.model if previous else None
        pages = self.source_client.crawl(
            self.settings.confluence_ingestion_token.get_secret_value(), heartbeat
        )
        collection = f"ragu_{self.collection_prefix}_{job.id.hex}"
        self.vectors.create(collection, mode=IndexMode(job.index_mode))
        chunks = 0
        for page in pages:
            heartbeat()
            chunks += self.vectors.index_page(
                collection,
                page,
                previous_collection,
                previous_model,
                heartbeat=heartbeat,
                mode=IndexMode(job.index_mode),
            )
            if chunks > self.settings.max_chunks:
                raise AppError(503, "Index exceeds the configured chunk limit.")
        with self.db.session() as session:
            current = authorize(session, job.id)
            state = self._state(session)
            if state is None:
                state = self.index_model(id=1)
                session.add(state)
            retire_index(session, state.collection)
            state.collection, state.model = collection, self.settings.embedding_model
            state.published_at = utcnow()
            self._publication(state, job)
            current.status, current.finished_at = "succeeded", utcnow()
            current.page_count, current.chunk_count = len(pages), chunks
        logger.info(
            "confluence_ingestion_succeeded job_id=%s pages=%s chunks=%s",
            job.id,
            len(pages),
            chunks,
        )

    def _context(self, principal: Principal):
        self.auth.throttle("search", str(principal.id), 60, 60)
        token = self.auth.confluence_token(principal)
        with self.db.session() as session:
            self._access(session, principal, admin=False)
            state = self._state(session)
            if state is None or state.collection is None:
                raise AppError(409, "No published index. An admin must run ingestion first.")
            self.validate_profile(state)
            return state.collection, token

    def search(
        self, data: SearchQuery, principal: Principal, *, candidate_ids=None, generation_id=None
    ) -> dict:
        collection, token = self._context(principal)
        from ragu.config import ROOT_PAGE_ID

        candidates = self._points(collection, data, ROOT_PAGE_ID, candidate_ids, generation_id)
        checked = {}
        results = []
        for point in candidates:
            payload = point.payload
            page_id = payload["page_id"]
            if page_id not in checked:
                try:
                    checked[page_id] = self.source_client.page(page_id, token)
                except AppError as exc:
                    if exc.status != 404:
                        raise
                    checked[page_id] = None
            page = checked[page_id]
            if page is None or page.version != payload["version"]:
                continue
            results.append(
                {
                    "page_id": page.id,
                    "title": page.title,
                    "passage": payload["text"],
                    "source_url": page.url,
                    "updated_at": page.updated_at,
                    "version": page.version,
                    "score": point.score,
                    "candidate_id": str(point.id),
                    "score_type": data.candidate_method.value,
                }
            )
            if len(results) == data.limit:
                break
        return self._batch(results, collection)

    def document(self, page_id: str, principal: Principal) -> dict:
        collection, token = self._context(principal)
        page = self.source_client.page(page_id, token)
        points = self.vectors.page_points(collection, page_id)
        if not points:
            raise AppError(404, "Indexed document not found.")
        if any(point.payload["version"] != page.version for point in points):
            raise AppError(409, "Document changed since indexing. Ask an admin to rerun ingestion.")
        return {
            "page_id": page.id,
            "title": page.title,
            "text": page.text,
            "source_url": page.url,
            "version": page.version,
            "updated_at": page.updated_at,
        }
