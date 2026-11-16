import logging
import math
from threading import BoundedSemaphore, Lock

from ragu.errors import AppError
from ragu.schemas import RerankMethod

logger = logging.getLogger("ragu.reranking")


class Reranker:
    def __init__(self, settings, model=None):
        self.settings = settings
        self._model = model
        self._loading = Lock()
        self._slots = BoundedSemaphore(settings.reranking_concurrency)

    @property
    def model(self):
        with self._loading:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                try:
                    self._model = TextCrossEncoder(
                        model_name=self.settings.reranking_model,
                        cache_dir=str(self.settings.embedding_cache),
                        local_files_only=self.settings.embedding_offline,
                        threads=self.settings.model_threads,
                        providers=["CPUExecutionProvider"],
                    )
                except (ValueError, OSError, RuntimeError) as exc:
                    logger.warning("reranker_unavailable reason=%s", type(exc).__name__)
                    raise AppError(
                        503, "Local reranker unavailable. Run 'ragu download-reranker'."
                    ) from exc
            return self._model

    def rank(self, query: str, candidates: list[dict], method: RerankMethod) -> list[dict]:
        if method == RerankMethod.NONE or not candidates:
            return candidates
        if method != RerankMethod.CROSS_ENCODER:
            raise AppError(422, "Unsupported reranking method.")
        if len(candidates) > 100 or (
            len(query) + sum(len(item["passage"]) for item in candidates)
            > self.settings.reranking_max_characters
        ):
            raise AppError(413, "Reranking input exceeds the configured budget.")
        if not self._slots.acquire(blocking=False):
            raise AppError(429, "Reranker capacity reached; try again later.")
        try:
            try:
                scores = list(
                    self.model.rerank(
                        query, [item["passage"] for item in candidates], batch_size=16
                    )
                )
            except (ValueError, OSError, RuntimeError) as exc:
                logger.warning("reranking_failed reason=%s", type(exc).__name__)
                raise AppError(503, "Local reranking failed.") from exc
            if len(scores) != len(candidates) or any(
                not math.isfinite(float(score)) for score in scores
            ):
                raise AppError(503, "Reranker returned invalid scores.")
            ranked = [
                {**item, "rerank_score": float(score)}
                for item, score in zip(candidates, scores, strict=True)
            ]
            return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)
        finally:
            self._slots.release()
