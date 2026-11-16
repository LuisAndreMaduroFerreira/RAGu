import hmac
import logging
from time import perf_counter

from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from ragu.database import Job, Schedule, aware
from ragu.errors import AppError
from ragu.strategies import StrategyRegistry

logger = logging.getLogger("ragu.management")
JOB_STATUSES = ("pending", "running", "succeeded", "failed")


def _timestamp(value) -> float:
    return aware(value).timestamp() if value is not None else 0


class IngestionCollector:
    def __init__(self, db, strategies: StrategyRegistry):
        self.db, self.strategies = db, strategies

    def collect(self):
        jobs = GaugeMetricFamily(
            "ragu_ingestion_jobs",
            "Retained ingestion jobs by current state.",
            labels=["source", "status"],
        )
        pending = GaugeMetricFamily(
            "ragu_ingestion_oldest_pending_timestamp_seconds",
            "Oldest queued ingestion time; zero when no job is pending.",
            labels=["source"],
        )
        heartbeat = GaugeMetricFamily(
            "ragu_ingestion_running_heartbeat_timestamp_seconds",
            "Most recent running-job heartbeat; zero when no job is running.",
            labels=["source"],
        )
        duration = GaugeMetricFamily(
            "ragu_ingestion_last_duration_seconds",
            "Wall time of the latest completed ingestion; zero before any execution.",
            labels=["source"],
        )
        success = GaugeMetricFamily(
            "ragu_ingestion_last_success_timestamp_seconds",
            "Last successful folder check or crawl, including unchanged runs.",
            labels=["source"],
        )
        documents = GaugeMetricFamily(
            "ragu_ingestion_last_success_documents",
            "Unique documents in the latest successful ingestion.",
            labels=["source"],
        )
        chunks = GaugeMetricFamily(
            "ragu_ingestion_last_success_chunks",
            "Chunks in the latest successful ingestion.",
            labels=["source"],
        )
        duplicates = GaugeMetricFamily(
            "ragu_ingestion_last_success_duplicate_files",
            "Exact duplicates and superseded local file versions ignored by the latest ingestion.",
            labels=["source"],
        )
        unchanged = GaugeMetricFamily(
            "ragu_ingestion_last_success_unchanged",
            "Whether the latest successful ingestion skipped vector publication.",
            labels=["source"],
        )
        published = GaugeMetricFamily(
            "ragu_index_published_timestamp_seconds",
            "Active index publication time; zero when no index is published.",
            labels=["source"],
        )
        scheduled = GaugeMetricFamily(
            "ragu_schedule_enabled",
            "Whether this source owns the enabled daily schedule.",
            labels=["source"],
        )
        next_run = GaugeMetricFamily(
            "ragu_schedule_next_run_timestamp_seconds",
            "Next scheduled run; zero for a disabled or absent schedule.",
            labels=["source"],
        )
        with self.db.session() as session:
            counts = {
                (source, status): count
                for source, status, count in session.execute(
                    select(Job.source, Job.status, func.count()).group_by(Job.source, Job.status)
                )
            }
            schedule = session.get(Schedule, 1)
            for strategy in self.strategies:
                source, model = strategy.source, strategy.index_model
                stored_source = strategy.storage_source
                for status in JOB_STATUSES:
                    jobs.add_metric([source, status], counts.get((stored_source, status), 0))
                queued_at = session.scalar(
                    select(func.min(Job.created_at)).where(
                        Job.source == stored_source,
                        Job.status == "pending",
                    )
                )
                running_at = session.scalar(
                    select(func.max(Job.heartbeat_at)).where(
                        Job.source == stored_source,
                        Job.status == "running",
                    )
                )
                pending.add_metric([source], _timestamp(queued_at))
                heartbeat.add_metric([source], _timestamp(running_at))
                latest = session.execute(
                    select(Job.started_at, Job.finished_at)
                    .where(
                        Job.source == stored_source,
                        Job.status.in_(("succeeded", "failed")),
                        Job.finished_at.is_not(None),
                    )
                    .order_by(Job.finished_at.desc(), Job.id.desc())
                    .limit(1)
                ).first()
                elapsed = (
                    (aware(latest.finished_at) - aware(latest.started_at)).total_seconds()
                    if latest and latest.started_at
                    else 0
                )
                duration.add_metric([source], max(0, elapsed))
                latest_success = session.execute(
                    select(
                        Job.finished_at,
                        Job.page_count,
                        Job.chunk_count,
                        Job.duplicate_count,
                        Job.unchanged,
                    )
                    .where(
                        Job.source == stored_source,
                        Job.status == "succeeded",
                    )
                    .order_by(Job.finished_at.desc(), Job.id.desc())
                    .limit(1)
                ).first()
                success.add_metric(
                    [source], _timestamp(latest_success.finished_at) if latest_success else 0
                )
                documents.add_metric([source], latest_success.page_count if latest_success else 0)
                chunks.add_metric([source], latest_success.chunk_count if latest_success else 0)
                duplicates.add_metric(
                    [source], latest_success.duplicate_count if latest_success else 0
                )
                unchanged.add_metric(
                    [source], int(latest_success.unchanged) if latest_success else 0
                )
                published_at = session.scalar(select(model.published_at).where(model.id == 1))
                published.add_metric([source], _timestamp(published_at))
                enabled = bool(schedule and schedule.enabled and schedule.source == stored_source)
                scheduled.add_metric([source], int(enabled))
                next_run.add_metric([source], _timestamp(schedule.next_run_at) if enabled else 0)
        yield from (
            jobs,
            pending,
            heartbeat,
            duration,
            success,
            documents,
            chunks,
            duplicates,
            unchanged,
            published,
            scheduled,
            next_run,
        )


class ManagementService:
    def __init__(self, db, auth, settings, vectors, strategies: StrategyRegistry):
        self.db, self.auth, self.settings, self.vectors = db, auth, settings, vectors
        self.registry = CollectorRegistry()
        GCCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        ProcessCollector(registry=self.registry)
        self.registry.register(IngestionCollector(db, strategies))
        self.requests = Counter(
            "ragu_http_requests_total",
            "Completed HTTP requests.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "ragu_http_request_duration_seconds",
            "HTTP request wall time, including streaming.",
            ("method", "route"),
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.inflight = Gauge(
            "ragu_http_requests_in_progress",
            "HTTP requests currently in progress.",
            registry=self.registry,
        )

    def _authorize(self, token: str | None) -> None:
        if not token:
            raise AppError(401, "Monitoring authentication required.")
        expected = self.settings.monitoring_token
        if expected and hmac.compare_digest(token.encode(), expected.get_secret_value().encode()):
            return
        principal = self.auth.authenticate(token, self.settings.api_audience, "api")
        with self.db.session() as session:
            self.auth.current(session, principal, admin=True)

    def prometheus(self, token: str | None) -> Response:
        self._authorize(token)
        return Response(
            generate_latest(self.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @staticmethod
    def liveness() -> dict:
        return {"status": "UP"}

    def readiness(self, token: str | None) -> JSONResponse:
        self._authorize(token)
        components = {}
        try:
            with self.db.session() as session:
                session.execute(select(1))
            components["database"] = {"status": "UP"}
        except SQLAlchemyError as exc:
            logger.warning(
                "readiness_check_failed component=database reason=%s", type(exc).__name__
            )
            components["database"] = {"status": "DOWN"}
        try:
            self.vectors.client.get_collections()
            components["qdrant"] = {"status": "UP"}
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            logger.warning("readiness_check_failed component=qdrant reason=%s", type(exc).__name__)
            components["qdrant"] = {"status": "DOWN"}
        healthy = all(component["status"] == "UP" for component in components.values())
        return JSONResponse(
            {"status": "UP" if healthy else "DOWN", "components": components},
            status_code=200 if healthy else 503,
        )


class HttpMetricsMiddleware:
    def __init__(self, app, management: ManagementService):
        self.app, self.management = app, management

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith("/actuator/"):
            return await self.app(scope, receive, send)
        method = scope["method"]
        if method not in {
            "GET",
            "HEAD",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "OPTIONS",
            "TRACE",
            "CONNECT",
        }:
            method = "OTHER"
        started = perf_counter()
        status = 500
        self.management.inflight.inc()

        async def observed(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observed)
        finally:
            route = getattr(scope.get("route"), "path", None)
            if not route:
                route = "/mcp" if scope["path"].rstrip("/") == "/mcp" else "unmatched"
            self.management.requests.labels(method, route, str(status)).inc()
            self.management.duration.labels(method, route).observe(perf_counter() - started)
            self.management.inflight.dec()
