from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class Base(DeclarativeBase):
    pass


class Role(Base):
    __tablename__ = "app_roles"
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), Identity(), primary_key=True
    )
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(100))


class User(Base):
    __tablename__ = "app_users"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(Text)
    role_id: Mapped[int] = mapped_column(ForeignKey("app_roles.id"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    auth_version: Mapped[int] = mapped_column(BigInteger, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    password_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confluence_token: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        Index("app_users_username_ci_unique", func.lower(username), unique=True),
        CheckConstraint("auth_version > 0"),
        CheckConstraint("username <> '' AND username = trim(username)"),
        CheckConstraint("password_hash <> ''"),
    )


class Throttle(Base):
    __tablename__ = "auth_throttles"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Job(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    requested_by: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    source: Mapped[str] = mapped_column(String(16), default="confluence")
    source_root: Mapped[str | None] = mapped_column(String(64))
    corpus_id: Mapped[UUID | None] = mapped_column(ForeignKey("corpora.id"))
    index_mode: Mapped[str] = mapped_column(String(16), default="dense")
    index_profile: Mapped[dict | None] = mapped_column(JSON)
    unchanged: Mapped[bool] = mapped_column(Boolean, default=False)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_reference_count: Mapped[int] = mapped_column(Integer, default=0)
    reused_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    parameters: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'succeeded', 'failed')"),
        CheckConstraint("source IN ('confluence', 'markdown', 'multi_source')"),
        Index(
            "one_active_ingestion",
            text("(1)"),
            unique=True,
            postgresql_where=text("status IN ('pending', 'running')"),
            sqlite_where=text("status IN ('pending', 'running')"),
        ),
    )


class Schedule(Base):
    __tablename__ = "daily_schedule"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    source: Mapped[str] = mapped_column(String(16), default="confluence")
    source_root: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean)
    hour_utc: Mapped[int] = mapped_column(Integer)
    minute_utc: Mapped[int] = mapped_column(Integer)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    index_mode: Mapped[str | None] = mapped_column(String(16))
    __table_args__ = (
        CheckConstraint("id = 1 AND hour_utc BETWEEN 0 AND 23 AND minute_utc BETWEEN 0 AND 59"),
        CheckConstraint("source IN ('confluence', 'markdown')"),
    )


class IndexState(Base):
    __tablename__ = "index_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    collection: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    index_mode: Mapped[str] = mapped_column(String(16), default="dense")
    index_profile: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (CheckConstraint("id = 1"),)


class LocalConsumptionIndexState(Base):
    __tablename__ = "markdown_index_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    source_root: Mapped[str] = mapped_column(String(64))
    collection: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    fingerprint: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSON)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    index_mode: Mapped[str] = mapped_column(String(16), default="dense")
    index_profile: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (CheckConstraint("id = 1"),)


class MultiSourceIndexState(Base):
    __tablename__ = "multi_source_index_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    collection: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    manifest: Mapped[dict] = mapped_column(JSON)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    index_mode: Mapped[str] = mapped_column(String(16), default="dense")
    index_profile: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (CheckConstraint("id = 1"),)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProjectMember(Base):
    __tablename__ = "project_members"
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))
    __table_args__ = (CheckConstraint("role IN ('reader', 'maintainer')"),)


class Corpus(Base):
    __tablename__ = "corpora"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(32))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    source_root: Mapped[str | None] = mapped_column(String(64))
    collection: Mapped[str | None] = mapped_column(String(100))
    model: Mapped[str | None] = mapped_column(String(200))
    index_mode: Mapped[str] = mapped_column(String(16), default="dense")
    index_profile: Mapped[dict | None] = mapped_column(JSON)
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("source IN ('confluence', 'local_consumption', 'multi_source')"),
        CheckConstraint("index_mode IN ('dense', 'sparse', 'hybrid')"),
    )


class CorpusGrant(Base):
    __tablename__ = "corpus_grants"
    corpus_id: Mapped[UUID] = mapped_column(ForeignKey("corpora.id"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"), primary_key=True)
    permission: Mapped[str] = mapped_column(String(16))
    __table_args__ = (CheckConstraint("permission IN ('read', 'ingest')"),)


class CorpusSchedule(Base):
    __tablename__ = "corpus_schedules"
    corpus_id: Mapped[UUID] = mapped_column(ForeignKey("corpora.id"), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    source: Mapped[str] = mapped_column(String(16))
    source_root: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean)
    hour_utc: Mapped[int] = mapped_column(Integer)
    minute_utc: Mapped[int] = mapped_column(Integer)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    index_mode: Mapped[str | None] = mapped_column(String(16))
    __table_args__ = (
        CheckConstraint("hour_utc BETWEEN 0 AND 23 AND minute_utc BETWEEN 0 AND 59"),
        CheckConstraint("source IN ('confluence', 'markdown')"),
    )


class RetiredIndex(Base):
    __tablename__ = "retired_indexes"
    collection: Mapped[str] = mapped_column(String(100), primary_key=True)
    retired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


def retire_index(session, collection: str | None) -> None:
    # Retirement commits with publication, preserving the grace period for in-flight readers.
    if collection is not None and session.get(RetiredIndex, collection) is None:
        session.add(RetiredIndex(collection=collection))


class OAuthRequest(Base):
    __tablename__ = "oauth_requests"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    data: Mapped[dict] = mapped_column(JSON)
    csrf_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OAuthCode(Base):
    __tablename__ = "oauth_codes"
    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_users.id"))
    auth_version: Mapped[int] = mapped_column(BigInteger)
    data: Mapped[dict] = mapped_column(JSON)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Database:
    def __init__(self, url, **engine_options):
        self.engine = create_engine(url, pool_pre_ping=True, hide_parameters=True, **engine_options)
        self.factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self):
        with self.factory.begin() as session:
            yield session

    @staticmethod
    def lock(session, key: int) -> None:
        if session.bind.dialect.name == "postgresql":
            session.execute(func.pg_advisory_xact_lock(key).select())
