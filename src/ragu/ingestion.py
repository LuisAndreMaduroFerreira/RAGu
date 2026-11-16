import logging
from datetime import timedelta
from uuid import UUID, uuid4

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import delete, select, update

from ragu.database import (
    Corpus,
    CorpusSchedule,
    Database,
    Job,
    OAuthCode,
    OAuthRequest,
    RetiredIndex,
    Role,
    Schedule,
    Throttle,
    User,
    aware,
    retire_index,
    utcnow,
)
from ragu.errors import AppError
from ragu.schemas import DEFAULT_SOURCE, DailySchedule, IndexMode, JobView, MultiSourceRequest
from ragu.strategies import StrategyRegistry

logger = logging.getLogger("ragu.ingestion")


class IngestionService:
    def __init__(
        self,
        db,
        auth,
        settings,
        vectors,
        strategies: StrategyRegistry,
        multi_source=None,
        projects=None,
    ):
        self.db, self.auth, self.settings = db, auth, settings
        self.vectors, self.strategies = vectors, strategies
        self.multi_source = multi_source
        self.projects = projects

    def _authorize_write(self, session, principal, corpus_id=None) -> None:
        # Queue, user, then schedule/job rows is the shared lock order for ingestion writers.
        Database.lock(session, 1002)
        self.auth.current(session, principal, admin=corpus_id is None)
        if corpus_id is not None:
            self.projects.corpus_access(session, corpus_id, principal.id, write=True)

    def _strategy(self, session, source, corpus_id=None):
        if corpus_id is not None:
            return self.projects.for_corpus(session.get(Corpus, corpus_id))
        return self.strategies.get(source)

    def _enqueue(
        self,
        session,
        user_id: UUID,
        source: str,
        source_root: str | None = None,
        parameters: dict | None = None,
        corpus_id=None,
        index_mode=None,
    ) -> Job:
        if session.scalar(select(Job.id).where(Job.status.in_(["pending", "running"]))):
            raise AppError(409, "An ingestion job is already pending or running.")
        strategy = self._strategy(session, source, corpus_id)
        state = session.get(strategy.index_model, getattr(strategy, "index_key", 1))
        mode = IndexMode(index_mode or (state.index_mode if state else "dense"))
        job = Job(
            id=uuid4(),
            requested_by=user_id,
            source=strategy.storage_source,
            source_root=source_root,
            parameters=parameters,
            corpus_id=corpus_id,
            index_mode=mode,
            index_profile=self.vectors.profile(mode),
        )
        session.add(job)
        session.flush()
        logger.info("ingestion_queued job_id=%s user_id=%s", job.id, user_id)
        return job

    def start(
        self,
        principal,
        source: str = DEFAULT_SOURCE,
        *,
        index_mode=None,
        corpus_id=None,
        explicit_source=False,
    ) -> JobView:
        with self.db.session() as session:
            self._authorize_write(session, principal, corpus_id)
            strategy = self._strategy(session, source, corpus_id)
            if explicit_source and corpus_id is not None and source != strategy.source:
                raise AppError(422, "source does not match the selected corpus.")
            root = strategy.bind(session, principal.id)
            return JobView.model_validate(
                self._enqueue(
                    session,
                    principal.id,
                    strategy.source,
                    root,
                    corpus_id=corpus_id,
                    index_mode=index_mode,
                )
            )

    def start_request(self, data, principal):
        return self.start(
            principal,
            data.source,
            index_mode=data.index_mode,
            corpus_id=data.corpus_id,
            explicit_source="source" in data.model_fields_set,
        )

    def status(self, job_id: UUID, principal) -> JobView:
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise AppError(404, "Job not found.")
            self.auth.current(session, principal, admin=job.corpus_id is None)
            if job.corpus_id is not None:
                self.projects.corpus_access(session, job.corpus_id, principal.id)
            return JobView.model_validate(job)

    def start_multi_source(self, principal, data: MultiSourceRequest) -> JobView:
        with self.db.session() as session:
            self._authorize_write(session, principal, data.corpus_id)
            if self.multi_source is None:
                raise AppError(503, "Multi-source ingestion is unavailable.")
            strategy = self._strategy(session, "multi_source", data.corpus_id)
            if strategy.source != "multi_source":
                raise AppError(422, "Use a multi-source corpus for this endpoint.")
            root = strategy.bind_request(session, principal.id, data)
            job = self._enqueue(
                session,
                principal.id,
                "multi_source",
                root,
                data.model_dump(mode="json"),
                corpus_id=data.corpus_id,
                index_mode=data.index_mode,
            )
            return JobView.model_validate(job)

    def list_jobs(self, principal, corpus_id=None) -> list[JobView]:
        with self.db.session() as session:
            self.auth.current(session, principal, admin=corpus_id is None)
            if corpus_id is not None:
                self.projects.corpus_access(session, corpus_id, principal.id)
            return [
                JobView.model_validate(job)
                for job in session.scalars(
                    select(Job)
                    .where(Job.corpus_id == corpus_id)
                    .order_by(Job.created_at.desc())
                    .limit(50)
                )
            ]

    @staticmethod
    def next_run(hour: int, minute: int):
        now = utcnow()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target if target > now else target + timedelta(days=1)

    def configure_schedule(self, data: DailySchedule, principal) -> dict:
        with self.db.session() as session:
            self._authorize_write(session, principal, data.corpus_id)
            strategy = self._strategy(session, data.source, data.corpus_id)
            if (
                data.corpus_id is not None
                and "source" in data.model_fields_set
                and data.source != strategy.source
            ):
                raise AppError(422, "source does not match the selected corpus.")
            if strategy.source == "multi_source":
                raise AppError(422, "Daily multi-source scheduling is not supported.")
            root = strategy.bind(session, principal.id) if data.enabled else None
            model, key = (CorpusSchedule, data.corpus_id) if data.corpus_id else (Schedule, 1)
            schedule = session.get(model, key)
            if schedule is None:
                schedule = (
                    CorpusSchedule(corpus_id=data.corpus_id) if data.corpus_id else Schedule(id=1)
                )
                session.add(schedule)
            schedule.owner_id = principal.id
            schedule.source = strategy.storage_source
            schedule.source_root = root
            schedule.enabled = data.enabled
            schedule.hour_utc = data.hour_utc
            schedule.minute_utc = data.minute_utc
            schedule.next_run_at = self.next_run(data.hour_utc, data.minute_utc)
            schedule.index_mode = data.index_mode
            logger.info("schedule_changed user_id=%s enabled=%s", principal.id, data.enabled)
            return {
                "enabled": schedule.enabled,
                "source": strategy.source,
                "hour_utc": schedule.hour_utc,
                "minute_utc": schedule.minute_utc,
                "next_run_at": schedule.next_run_at,
                "corpus_id": data.corpus_id,
                "index_mode": schedule.index_mode,
            }

    def schedule(self, principal, corpus_id=None) -> dict:
        with self.db.session() as session:
            self.auth.current(session, principal, admin=corpus_id is None)
            if corpus_id is not None:
                self.projects.corpus_access(session, corpus_id, principal.id)
            item = session.get(CorpusSchedule, corpus_id) if corpus_id else session.get(Schedule, 1)
            return (
                {"enabled": False}
                if item is None
                else {
                    "enabled": item.enabled,
                    "source": self.strategies.get(item.source).source,
                    "hour_utc": item.hour_utc,
                    "minute_utc": item.minute_utc,
                    "next_run_at": item.next_run_at,
                    "corpus_id": corpus_id,
                    "index_mode": item.index_mode,
                }
            )

    def heartbeat(self, job_id: UUID) -> None:
        with self.db.session() as session:
            job = session.get(Job, job_id, with_for_update=True)
            if job is None or job.status != "running":
                raise AppError(409, "Ingestion lease was lost.")
            job.heartbeat_at = utcnow()

    def _maintenance(self):
        now = utcnow()
        with self.db.session() as session:
            Database.lock(session, 1002)
            for job in session.scalars(
                select(Job).where(Job.status == "running").with_for_update()
            ):
                if aware(job.heartbeat_at or job.started_at) < (
                    now - timedelta(seconds=self.settings.stale_job_seconds)
                ):
                    job.status, job.finished_at = "failed", now
                    job.error = "Worker lease expired; rerun ingestion."
                    retire_index(session, self.collection_name(job.id, job.source, job.corpus_id))
                    logger.error("ingestion_lease_expired job_id=%s", job.id)
            session.execute(
                delete(Throttle).where(Throttle.window_start < now - timedelta(hours=1))
            )
            session.execute(delete(OAuthCode).where(OAuthCode.expires_at < now))
            session.execute(delete(OAuthRequest).where(OAuthRequest.expires_at < now))
            schedules = [
                (Schedule, key)
                for key in session.scalars(
                    select(Schedule.id).where(Schedule.enabled, Schedule.next_run_at <= now)
                )
            ]
            schedules.extend(
                (CorpusSchedule, key)
                for key in session.scalars(
                    select(CorpusSchedule.corpus_id).where(
                        CorpusSchedule.enabled, CorpusSchedule.next_run_at <= now
                    )
                )
            )
        # Commit each schedule before acquiring another owner's account or project locks.
        for model, key in schedules:
            self._maintain_schedule(model, key)

    def _maintain_schedule(self, model: type[Schedule] | type[CorpusSchedule], key: int | UUID):
        with self.db.session() as session:
            Database.lock(session, 1002)
            schedule = session.get(model, key)
            if not schedule or not schedule.enabled or aware(schedule.next_run_at) > utcnow():
                return
            corpus_id = getattr(schedule, "corpus_id", None)
            if not self._owner_authorized(session, schedule.owner_id, corpus_id):
                schedule.enabled = False
                logger.warning("schedule_disabled_owner_not_authorized corpus_id=%s", corpus_id)
                return
            try:
                if not self._strategy(session, schedule.source, corpus_id).schedule_valid(
                    session, schedule
                ):
                    schedule.enabled = False
                    logger.warning(
                        "schedule_disabled_source_binding_changed corpus_id=%s", corpus_id
                    )
                elif not session.scalar(
                    select(Job.id).where(Job.status.in_(["pending", "running"]))
                ):
                    self._enqueue(
                        session,
                        schedule.owner_id,
                        schedule.source,
                        schedule.source_root,
                        corpus_id=corpus_id,
                        index_mode=schedule.index_mode,
                    )
                    schedule.next_run_at = self.next_run(schedule.hour_utc, schedule.minute_utc)
            except AppError as exc:
                if exc.status == 503:
                    logger.warning(
                        "schedule_source_unavailable corpus_id=%s source=%s",
                        corpus_id,
                        schedule.source,
                    )
                elif exc.status in (403, 404, 409, 422):
                    schedule.enabled = False
                    logger.warning(
                        "schedule_disabled_invalid_source corpus_id=%s status=%s",
                        corpus_id,
                        exc.status,
                    )
                else:
                    raise

    def cleanup_indexes(self):
        with self.db.session() as session:
            expired = session.scalars(
                select(RetiredIndex)
                .where(RetiredIndex.retired_at < utcnow() - timedelta(hours=1))
                .with_for_update(skip_locked=True)
            )
            for retired in expired:
                try:
                    if self.vectors.client.collection_exists(retired.collection):
                        self.vectors.client.delete_collection(retired.collection)
                except (ResponseHandlingException, UnexpectedResponse) as exc:
                    # Cleanup outages must remain retryable without terminating the worker.
                    logger.warning(
                        "index_cleanup_failed collection=%s reason=%s",
                        retired.collection,
                        type(exc).__name__,
                    )
                    continue
                session.delete(retired)

    def collection_name(self, job_id: UUID, source: str, corpus_id=None) -> str:
        if corpus_id is not None:
            return f"ragu_corpus_{corpus_id.hex}_{job_id.hex}"
        return self.strategies.collection_name(job_id, source)

    def _owner_authorized(self, session, owner_id, corpus_id):
        owner = session.get(User, owner_id, with_for_update={"key_share": True})
        if owner is None or not owner.is_active:
            return False
        if corpus_id is not None:
            try:
                self.projects.corpus_access(session, corpus_id, owner_id, write=True)
            except AppError as exc:
                if exc.status != 404:
                    raise
                return False
            return True
        role = session.get(Role, owner.role_id)
        return role is not None and role.code == "admin"

    def _running_job(self, session, job_id: UUID) -> Job:
        Database.lock(session, 1002)
        job = session.get(Job, job_id, with_for_update=True)
        if (
            not job
            or job.status != "running"
            or not self._owner_authorized(session, job.requested_by, job.corpus_id)
        ):
            raise AppError(409, "Ingestion authorization or lease was revoked.")
        if job.index_profile is not None and job.index_profile != self.vectors.profile(
            job.index_mode
        ):
            raise AppError(409, "Index profile changed during ingestion.")
        return job

    def tick(self) -> bool:
        self._maintenance()
        self.cleanup_indexes()
        with self.db.session() as session:
            Database.lock(session, 1002)
            job = session.scalar(select(Job).where(Job.status == "pending").with_for_update())
            if job is None:
                return False
            if not self._owner_authorized(session, job.requested_by, job.corpus_id):
                job.status, job.finished_at, job.error = (
                    "failed",
                    utcnow(),
                    "Owner is not authorized.",
                )
                return True
            job.status = "running"
            job.started_at = job.heartbeat_at = utcnow()
            job_id = job.id
        try:
            if job.index_profile is not None and job.index_profile != self.vectors.profile(
                job.index_mode
            ):
                raise AppError(409, "Index profile changed after this job was authorized.")
            with self.db.session() as session:
                strategy = self._strategy(session, job.source, job.corpus_id)
            strategy.ingest(job, lambda: self.heartbeat(job_id), self._running_job)
        except Exception as exc:
            logger.exception("ingestion_interrupted job_id=%s", job_id)
            with self.db.session() as session:
                error = (
                    exc.message
                    if isinstance(exc, AppError)
                    else ("Ingestion failed. Consult the private server logs.")
                )
                # A lost commit acknowledgement must not overwrite an already committed outcome.
                changed = session.execute(
                    update(Job)
                    .where(Job.id == job_id, Job.status == "running")
                    .values(status="failed", finished_at=utcnow(), error=error)
                ).rowcount
                if changed:
                    retire_index(session, self.collection_name(job.id, job.source, job.corpus_id))
        return True
