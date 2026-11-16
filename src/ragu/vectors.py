import logging
import math
from collections.abc import Callable
from importlib.metadata import version
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ragu.config import ROOT_PAGE_ID
from ragu.errors import AppError
from ragu.schemas import CandidateMethod, IndexMode

logger = logging.getLogger("ragu.vectors")


class Vectors:
    def __init__(self, settings, client=None, embedder=None, sparse_embedder=None):
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=30,
            check_compatibility=False,
        )
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._loading = Lock()
        self._implementation = {
            "fastembed": version("fastembed"),
            "splitter": version("langchain-text-splitters"),
        }
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    @property
    def embedder(self):
        with self._loading:
            if self._embedder is None:
                from fastembed import TextEmbedding

                try:
                    self._embedder = TextEmbedding(
                        model_name=self.settings.embedding_model,
                        cache_dir=str(self.settings.embedding_cache),
                        local_files_only=self.settings.embedding_offline,
                        threads=self.settings.model_threads,
                        providers=["CPUExecutionProvider"],
                    )
                except (ValueError, OSError, RuntimeError) as exc:
                    raise AppError(
                        503, "Local embedding model unavailable. Run 'ragu download-model'."
                    ) from exc
        return self._embedder

    @property
    def sparse_embedder(self):
        with self._loading:
            if self._sparse_embedder is None:
                from fastembed import SparseTextEmbedding

                try:
                    self._sparse_embedder = SparseTextEmbedding(
                        model_name=self.settings.sparse_model,
                        cache_dir=str(self.settings.embedding_cache),
                        local_files_only=self.settings.embedding_offline,
                        language=self.settings.sparse_language,
                    )
                except (ValueError, OSError, RuntimeError) as exc:
                    raise AppError(
                        503, "Local sparse model unavailable. Run 'ragu download-sparse-model'."
                    ) from exc
        return self._sparse_embedder

    def profile(self, mode: IndexMode | str) -> dict:
        mode = IndexMode(mode)
        return {
            "mode": mode.value,
            "dense_model": self.settings.embedding_model if mode != IndexMode.SPARSE else None,
            "sparse_model": self.settings.sparse_model if mode != IndexMode.DENSE else None,
            "sparse_language": self.settings.sparse_language if mode != IndexMode.DENSE else None,
            "chunk_size": 800,
            "chunk_overlap": 120,
            "schema": 1,
            "implementation": self._implementation,
        }

    def exists(self, collection: str | None) -> bool:
        if collection is None:
            return False
        try:
            return self.client.collection_exists(collection)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise AppError(503, "The document index is unavailable.") from exc

    def create(
        self, collection: str, previous: str | None = None, *, mode: IndexMode = IndexMode.DENSE
    ) -> None:
        dimension = None
        if previous and self.exists(previous):
            parameters = self.client.get_collection(previous).config.params.vectors
            dense = (
                parameters
                if isinstance(parameters, models.VectorParams)
                else parameters.get("dense")
            )
            dimension = dense.size if dense else None
        if mode != IndexMode.SPARSE and dimension is None:
            dimension = len(next(iter(self.embedder.embed(["dimension probe"]))))
        dense_config = (
            models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            if dimension is not None
            else None
        )
        self.client.create_collection(
            collection_name=collection,
            vectors_config=dense_config
            if mode == IndexMode.DENSE
            else ({"dense": dense_config} if mode == IndexMode.HYBRID else {}),
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}
            if mode != IndexMode.DENSE
            else None,
            metadata={"ragu_profile": self.profile(mode)},
        )

    @staticmethod
    def dense_values(vector):
        return vector.get("dense") if isinstance(vector, dict) else vector

    def encode(self, texts: list[str], mode: IndexMode, reused: dict | None = None) -> list:
        dense = dict(reused or {})
        if mode != IndexMode.SPARSE:
            missing = list(dict.fromkeys(text for text in texts if text not in dense))
            generated = list(self.embedder.embed(missing)) if missing else []
            if len(generated) != len(missing):
                raise AppError(503, "Embedding model returned an incomplete batch.")
            for text, vector in zip(missing, generated, strict=True):
                values = vector.tolist()
                if not values or any(not math.isfinite(value) for value in values):
                    raise AppError(503, "Embedding model returned an invalid vector.")
                dense[text] = values
        sparse = []
        if mode != IndexMode.DENSE:
            generated = list(self.sparse_embedder.embed(texts)) if texts else []
            if len(generated) != len(texts):
                raise AppError(503, "Sparse model returned an incomplete batch.")
            for vector in generated:
                indices, values = vector.indices.tolist(), vector.values.tolist()
                if len(indices) != len(set(indices)) or any(
                    not math.isfinite(value) for value in values
                ):
                    raise AppError(503, "Sparse model returned an invalid vector.")
                sparse.append(models.SparseVector(indices=indices, values=values))
        if mode == IndexMode.DENSE:
            return [dense[text] for text in texts]
        return [
            {"bm25": sparse[index], **({"dense": dense[text]} if mode == IndexMode.HYBRID else {})}
            for index, text in enumerate(texts)
        ]

    def page_points(
        self,
        collection: str,
        page_id: str,
        root_id: str = ROOT_PAGE_ID,
        *,
        heartbeat: Callable[[], None] | None = None,
    ) -> list:
        points = []
        offset = None
        while True:
            if heartbeat is not None:
                heartbeat()
            try:
                batch, offset = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="page_id", match=models.MatchValue(value=page_id)
                            ),
                            models.FieldCondition(
                                key="root_id", match=models.MatchValue(value=root_id)
                            ),
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
            except (ResponseHandlingException, UnexpectedResponse) as exc:
                raise AppError(503, "The document index is unavailable.") from exc
            points.extend(batch)
            if offset is None:
                return points

    def index_page(
        self,
        collection: str,
        page,
        previous: str | None,
        previous_model: str | None,
        root_id: str = ROOT_PAGE_ID,
        previous_page_id: str | None = None,
        *,
        heartbeat: Callable[[], None],
        mode: IndexMode = IndexMode.DENSE,
    ) -> int:
        old = (
            self.page_points(previous, previous_page_id or page.id, root_id, heartbeat=heartbeat)
            if previous and self.exists(previous)
            else []
        )
        if (
            old
            and previous_model == self.settings.embedding_model
            and all(point.payload["version"] == page.version for point in old)
            and self.client.get_collection(previous).config.metadata
            == {"ragu_profile": self.profile(mode)}
        ):
            points = [
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in old
            ]
        else:
            passages = self.splitter.split_text(page.text)
            cached = {}
            previous_profile = (
                (self.info(previous).config.metadata or {}).get("ragu_profile") if old else None
            )
            if previous_model == self.settings.embedding_model and (
                previous_profile is None
                or previous_profile.get("implementation") == self._implementation
            ):
                for point in old:
                    dense = self.dense_values(point.vector)
                    if dense is not None:
                        cached[point.payload["text"]] = dense
            encoded = {}
            # Renew the lease between bounded embedding and write batches, not only between pages.
            for offset in range(0, len(passages), 100):
                heartbeat()
                batch = list(dict.fromkeys(passages[offset : offset + 100]))
                vectors = self.encode(batch, mode, cached)
                encoded.update(zip(batch, vectors, strict=True))
                cached.update(
                    (text, self.dense_values(vector))
                    for text, vector in zip(batch, vectors, strict=True)
                    if self.dense_values(vector) is not None
                )
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{root_id}:{page.id}:{index}")),
                    vector=encoded[text],
                    payload={
                        "page_id": page.id,
                        "root_id": root_id,
                        "title": page.title,
                        "version": page.version,
                        "text": text,
                        "source_url": page.url,
                        "updated_at": page.updated_at,
                        "chunk": index,
                    },
                )
                for index, text in enumerate(passages)
            ]
        for offset in range(0, len(points), 100):
            heartbeat()
            self.client.upsert(
                collection_name=collection, points=points[offset : offset + 100], wait=True
            )
        return len(points)

    def methods(self, collection: str) -> list[CandidateMethod]:
        params = self.info(collection).config.params
        dense = bool(params.vectors)
        sparse = bool(params.sparse_vectors and "bm25" in params.sparse_vectors)
        return (
            ([CandidateMethod.DENSE] if dense else [])
            + ([CandidateMethod.BM25] if sparse else [])
            + ([CandidateMethod.HYBRID_RRF] if dense and sparse else [])
        )

    def info(self, collection):
        try:
            return self.client.get_collection(collection)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise AppError(503, "The document index is unavailable.") from exc

    def select(self, collection: str, identifiers: list[str], root_id: str):
        try:
            records = self.client.retrieve(
                collection_name=collection, ids=identifiers, with_payload=True, with_vectors=False
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise AppError(503, "The document index is unavailable.") from exc
        by_id = {
            str(point.id): point for point in records if point.payload.get("root_id") == root_id
        }
        if set(by_id) != set(identifiers):
            raise AppError(404, "One or more candidates are unavailable.")
        return [
            models.ScoredPoint(
                id=identifier, version=0, score=0.0, payload=by_id[identifier].payload
            )
            for identifier in identifiers
        ]

    def search(
        self,
        collection: str,
        query: str,
        limit: int,
        root_id: str = ROOT_PAGE_ID,
        *,
        method: CandidateMethod = CandidateMethod.DENSE,
    ):
        try:
            methods = self.methods(collection)
            if method not in methods:
                raise AppError(409, "Requested candidate method requires a different index mode.")
            restriction = models.Filter(
                must=[models.FieldCondition(key="root_id", match=models.MatchValue(value=root_id))]
            )
            dense = sparse = None
            params = self.client.get_collection(collection).config.params
            dense_name = "dense" if isinstance(params.vectors, dict) else None
            if method != CandidateMethod.BM25:
                dense = next(iter(self.embedder.query_embed(query))).tolist()
            if method != CandidateMethod.DENSE:
                embedded = next(iter(self.sparse_embedder.query_embed(query)))
                sparse = models.SparseVector(
                    indices=embedded.indices.tolist(), values=embedded.values.tolist()
                )
            if method == CandidateMethod.HYBRID_RRF:
                return self.client.query_points(
                    collection_name=collection,
                    prefetch=[
                        models.Prefetch(
                            query=dense, using=dense_name, filter=restriction, limit=limit
                        ),
                        models.Prefetch(
                            query=sparse, using="bm25", filter=restriction, limit=limit
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    query_filter=restriction,
                    with_payload=True,
                ).points
            return self.client.query_points(
                collection_name=collection,
                query=dense if method == CandidateMethod.DENSE else sparse,
                using=dense_name if method == CandidateMethod.DENSE else "bm25",
                limit=limit,
                query_filter=restriction,
                with_payload=True,
            ).points
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise AppError(503, "The document index is unavailable.") from exc

    def close(self):
        self.client.close()
