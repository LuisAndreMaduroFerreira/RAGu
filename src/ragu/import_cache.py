import logging
from collections.abc import Sequence
from uuid import UUID

from redis import Redis
from redis.exceptions import RedisError

from ragu.errors import AppError

logger = logging.getLogger("ragu.import_cache")


class ImportCache:
    """Temporary write-completion markers; vectors remain exclusively in Qdrant."""

    def __init__(self, settings, job_id: UUID):
        if settings.import_redis_url is None:
            raise AppError(503, "Configure RAGU_IMPORT_REDIS_URL for multi-source imports.")
        self.client = Redis.from_url(
            settings.import_redis_url.get_secret_value(),
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True,
        )
        self.prefix = f"ragu:import:{job_id.hex}:"
        self.ttl = settings.import_cache_ttl_seconds

    def __enter__(self):
        try:
            self.client.ping()
        except RedisError:
            self.client.close()
            logger.warning("import_redis_unavailable")
            raise AppError(503, "The import Redis service is unavailable.") from None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.clear()
        except AppError:
            if exc_type is None:
                raise
            logger.exception("import_cache_cleanup_failed; keys will expire")
        finally:
            self.client.close()

    def known(self, identifiers: Sequence[str]) -> set[str]:
        if not identifiers:
            return set()
        try:
            values = self.client.mget([self.prefix + identifier for identifier in identifiers])
        except RedisError:
            logger.warning("import_cache_read_failed")
            raise AppError(503, "The import Redis service is unavailable.") from None
        if any(value not in {None, "written"} for value in values):
            raise AppError(503, "The import cache contains invalid markers.")
        return {identifier for identifier, value in zip(identifiers, values, strict=True) if value}

    def remember(self, identifiers: Sequence[str]) -> None:
        try:
            with self.client.pipeline(transaction=False) as pipeline:
                for identifier in identifiers:
                    pipeline.set(self.prefix + identifier, "written", ex=self.ttl)
                pipeline.execute()
        except RedisError:
            logger.warning("import_cache_write_failed")
            raise AppError(503, "The import Redis service is unavailable.") from None

    def clear(self) -> None:
        try:
            batch = []
            for key in self.client.scan_iter(match=self.prefix + "*", count=500):
                batch.append(key)
                if len(batch) == 500:
                    self.client.unlink(*batch)
                    batch.clear()
            if batch:
                self.client.unlink(*batch)
        except RedisError:
            logger.warning("import_cache_cleanup_failed")
            raise AppError(503, "Temporary import cache cleanup failed.") from None
