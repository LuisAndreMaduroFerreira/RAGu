import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from qdrant_client import QdrantClient
from sqlalchemy import URL, MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.exc import ProgrammingError

from ragu.cli import protect_config
from ragu.config import Settings
from ragu.database import (
    Base,
    CorpusSchedule,
    Database,
    IndexState,
    Job,
    LocalConsumptionIndexState,
    MultiSourceIndexState,
    RetiredIndex,
    Role,
    Schedule,
    User,
    retire_index,
    utcnow,
)
from ragu.errors import AppError
from ragu.runtime import Runtime
from ragu.schemas import (
    CorpusCreate,
    DailySchedule,
    GrantUpdate,
    MemberUpdate,
    MultiSourceRequest,
    ProjectCreate,
)
from ragu.vectors import Vectors
from tests.conftest import Embeddings, Source

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def postgres_server(tmp_path_factory):
    configured = os.environ.get("RAGU_TEST_POSTGRES_BIN")
    executable = shutil.which("initdb")
    if not configured and executable is None:
        pytest.skip("Set RAGU_TEST_POSTGRES_BIN or put PostgreSQL server binaries on PATH.")
    binaries = Path(configured) if configured else Path(executable).parent
    suffix = ".exe" if os.name == "nt" else ""
    initdb, pg_ctl = (binaries / (name + suffix) for name in ("initdb", "pg_ctl"))
    assert initdb.is_file() and pg_ctl.is_file(), "PostgreSQL server binaries are incomplete."
    # A private temporary cluster exercises real migrations and locks without touching the app DB.
    root = tmp_path_factory.mktemp("ragu-postgres")
    data, password_file = root / "data", root / "password"
    password_file.touch(mode=0o600)
    protect_config(password_file)
    password = secrets.token_urlsafe(32)
    password_file.write_text(password, encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    admin = None
    initialized = False
    try:
        subprocess.run(
            [
                str(initdb),
                "-D",
                str(data),
                "-U",
                "ragu_test_admin",
                "--auth=scram-sha-256",
                "--pwfile",
                str(password_file),
                "--no-locale",
                "--encoding=UTF8",
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        initialized = True
        password_file.unlink()
        with (root / "pg_ctl.log").open("w", encoding="utf-8") as output:
            subprocess.run(
                [
                    str(pg_ctl),
                    "-D",
                    str(data),
                    "-l",
                    str(root / "server.log"),
                    "-o",
                    f"-h 127.0.0.1 -p {port}",
                    "-w",
                    "-t",
                    "30",
                    "start",
                ],
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=45,
            )
        url = URL.create(
            "postgresql+psycopg",
            username="ragu_test_admin",
            password=password,
            host="127.0.0.1",
            port=port,
            database="postgres",
        )
        admin = create_engine(url, isolation_level="AUTOCOMMIT", hide_parameters=True)
        with admin.connect() as connection:
            assert connection.scalar(text("SELECT version()")).startswith("PostgreSQL")
            connection.execute(text("CREATE ROLE ragu_app NOLOGIN"))
        yield admin, url
    finally:
        if admin is not None:
            admin.dispose()
        if initialized:
            status = subprocess.run(
                [str(pg_ctl), "-D", str(data), "status"],
                check=False,
                capture_output=True,
                timeout=10,
            )
            if status.returncode not in (0, 3):
                status.check_returncode()
            if status.returncode == 0:
                subprocess.run(
                    [str(pg_ctl), "-D", str(data), "-m", "fast", "-w", "-t", "30", "stop"],
                    check=True,
                    capture_output=True,
                    timeout=45,
                )
        shutil.rmtree(root)


@pytest.fixture
def postgres_database(postgres_server):
    admin, url = postgres_server
    name = "ragu_test_" + uuid4().hex
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    db = Database(
        url.set(database=name),
        connect_args={"options": "-c lock_timeout=5000 -c statement_timeout=15000"},
    )
    try:
        yield db
    finally:
        db.engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


def migrate(db, revision="head"):
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    with db.engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


@pytest.fixture
def postgres_runtime(postgres_database, tmp_path):
    migrate(postgres_database)
    settings = Settings(
        _env_file=None,
        database_password="unused-test-password",
        jwt_secret=secrets.token_urlsafe(48),
        credential_key=Fernet.generate_key().decode(),
        confluence_ingestion_token="ingestion-test-token",
        multi_source_roots=[tmp_path],
        import_redis_url="redis://127.0.0.1:6379/0",
    )
    runtime = Runtime(
        settings,
        db=postgres_database,
        source=Source(),
        vectors=Vectors(settings, QdrantClient(":memory:"), Embeddings()),
    )
    password = secrets.token_urlsafe(24)
    with postgres_database.session() as session:
        role = session.scalar(select(Role).where(Role.code == "admin"))
        session.add(
            User(
                username="admin",
                password_hash=runtime.auth.hasher.hash(password),
                role_id=role.id,
            )
        )
    principal = runtime.auth.credentials("admin", password, "postgres-test")
    try:
        yield runtime, principal, tmp_path
    finally:
        runtime.close()


def test_real_migrations_preserve_accounts_backfill_retirement_and_enforce_grants(
    postgres_database,
):
    db = postgres_database
    migrate(db, "0002")
    with db.session() as session:
        role = session.scalar(select(Role).where(Role.code == "admin"))
        user = User(username="migration-fixture", password_hash="synthetic-hash", role_id=role.id)
        session.add(user)
        session.flush()
        identifier = user.id
    migrate(db, "0003")
    inactive = set()
    # Seed historical tables with their historical schema, not the current ORM's added columns.
    tables = {
        model: Table(model.__tablename__, MetaData(), autoload_with=db.engine)
        for model in (Job, IndexState, LocalConsumptionIndexState, MultiSourceIndexState)
    }
    with db.session() as session:
        for source, prefix in (
            ("confluence", "41276872"),
            ("markdown", "markdown"),
            ("multi_source", "multi_source"),
        ):
            for active in (True, False):
                values = {
                    "id": uuid4(),
                    "requested_by": identifier,
                    "source": source,
                    "status": "succeeded" if active else "failed",
                    "created_at": utcnow(),
                    "page_count": 0,
                    "chunk_count": 0,
                    "finished_at": utcnow() - timedelta(days=2),
                }
                session.execute(tables[Job].insert().values(**values))
                collection = f"ragu_{prefix}_{values['id'].hex}"
                if not active:
                    inactive.add(collection)
                elif source == "confluence":
                    session.execute(
                        tables[IndexState]
                        .insert()
                        .values(id=1, collection=collection, model="test", published_at=utcnow())
                    )
                elif source == "markdown":
                    session.execute(
                        tables[LocalConsumptionIndexState]
                        .insert()
                        .values(
                            id=1,
                            owner_id=identifier,
                            source_root="0" * 64,
                            collection=collection,
                            model="test",
                            fingerprint="0" * 64,
                            manifest={},
                            published_at=utcnow(),
                        )
                    )
                else:
                    session.execute(
                        tables[MultiSourceIndexState]
                        .insert()
                        .values(
                            id=1,
                            owner_id=identifier,
                            collection=collection,
                            model="test",
                            manifest={},
                            published_at=utcnow(),
                        )
                    )
        session.execute(
            tables[Job]
            .insert()
            .values(
                id=uuid4(),
                requested_by=identifier,
                source="confluence",
                status="succeeded",
                unchanged=True,
                created_at=utcnow(),
                page_count=0,
                chunk_count=0,
            )
        )
    before = utcnow()
    migrate(db)
    with db.session() as session:
        rows = list(session.scalars(select(RetiredIndex)))
        assert {row.collection for row in rows} == inactive
        assert all(row.retired_at >= before for row in rows)
        assert session.get(User, identifier).password_hash == "synthetic-hash"
        assert session.scalar(text("SELECT version_num FROM alembic_version")) == "0005"
    with db.engine.begin() as connection:
        connection.execute(text("SET LOCAL ROLE ragu_app"))
        assert set(connection.scalars(select(RetiredIndex.collection))) == inactive
        connection.execute(
            text("INSERT INTO retired_indexes VALUES ('runtime-fixture', CURRENT_TIMESTAMP)")
        )
        connection.execute(
            text(
                "UPDATE retired_indexes SET retired_at = CURRENT_TIMESTAMP "
                "WHERE collection = 'runtime-fixture'"
            )
        )
        connection.execute(text("DELETE FROM retired_indexes WHERE collection = 'runtime-fixture'"))
    with pytest.raises(ProgrammingError) as denied, db.engine.begin() as connection:
        connection.execute(text("SET LOCAL ROLE ragu_app"))
        connection.execute(text("SELECT version_num FROM alembic_version"))
    assert denied.value.orig.sqlstate == "42501"


@pytest.mark.parametrize(
    "filename, previous, version",
    [
        ("fresh-schema.sql", None, "0005"),
        ("0004-index-retirement.sql", "0003", "0003"),
        ("0005-retrieval-projects.sql", "0004", "0004"),
    ],
)
def test_sql_references_execute_on_real_postgres(postgres_database, filename, previous, version):
    db = postgres_database
    if previous:
        migrate(db, previous)
    path = Path(__file__).resolve().parents[1] / "documentation" / "sql" / filename
    with db.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql(path.read_text(encoding="utf-8"))
        expected = set(Base.metadata.tables)
        if filename == "0004-index-retirement.sql":
            expected -= {
                "projects",
                "project_members",
                "corpora",
                "corpus_grants",
                "corpus_schedules",
            }
        assert set(inspect(connection).get_table_names()) == expected | {"alembic_version"}
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == version


@pytest.mark.parametrize("operation", ["start", "multi_source", "schedule"])
def test_real_manual_and_scheduled_writers_do_not_deadlock(
    postgres_runtime, monkeypatch, operation
):
    runtime, principal, root = postgres_runtime
    runtime.ingestion.configure_schedule(
        DailySchedule(enabled=True, hour_utc=3, minute_utc=0, source="confluence"), principal
    )
    with runtime.db.session() as session:
        session.get(Schedule, 1).next_run_at = utcnow() - timedelta(minutes=1)
    locked, waiting, release = threading.Event(), threading.Event(), threading.Event()
    original = Database.lock
    scheduler_locks = 0

    def observe(session, key):
        nonlocal scheduler_locks
        thread = threading.current_thread().name
        if thread == "ragu-test-api" and key == 1002:
            waiting.set()
        original(session, key)
        if thread == "ragu-test-scheduler" and key == 1002:
            scheduler_locks += 1
            if scheduler_locks == 2:
                locked.set()
                assert release.wait(5)

    def schedule():
        threading.current_thread().name = "ragu-test-scheduler"
        runtime.ingestion._maintenance()

    def request():
        threading.current_thread().name = "ragu-test-api"
        try:
            if operation == "start":
                runtime.ingestion.start(principal, "confluence")
            elif operation == "multi_source":
                runtime.ingestion.start_multi_source(
                    principal,
                    MultiSourceRequest.model_validate(
                        {"sources": [{"kind": "folder", "path": str(root)}]}
                    ),
                )
            else:
                runtime.ingestion.configure_schedule(
                    DailySchedule(enabled=False, hour_utc=4, minute_utc=0, source="confluence"),
                    principal,
                )
        except AppError as exc:
            return exc.status
        return 200

    monkeypatch.setattr(Database, "lock", staticmethod(observe))
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(schedule)
        try:
            assert locked.wait(5)
            manual = pool.submit(request)
            assert waiting.wait(5)
        finally:
            release.set()
        worker.result(timeout=10)
        assert manual.result(timeout=10) == (200 if operation == "schedule" else 409)
    with runtime.db.session() as session:
        assert len(list(session.scalars(select(Job).where(Job.status == "pending")))) == 1


def test_real_maintenance_does_not_expire_a_concurrently_renewed_lease(
    postgres_runtime, monkeypatch
):
    runtime, principal, _ = postgres_runtime
    job = runtime.ingestion.start(principal, "confluence")
    with runtime.db.session() as session:
        current = session.get(Job, job.id)
        current.status = "running"
        current.started_at = current.heartbeat_at = utcnow() - timedelta(hours=1)
    original = runtime.db.session
    started = threading.Event()
    backend = []

    @contextmanager
    def observed_session():
        with original() as session:
            if threading.current_thread().name == "ragu-test-maintenance":
                backend.append(session.scalar(text("SELECT pg_backend_pid()")))
                started.set()
            yield session

    def maintenance():
        threading.current_thread().name = "ragu-test-maintenance"
        runtime.ingestion._maintenance()

    monkeypatch.setattr(runtime.db, "session", observed_session)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with original() as session:
            current = session.get(Job, job.id, with_for_update=True)
            current.heartbeat_at = utcnow()
            worker = pool.submit(maintenance)
            assert started.wait(5)
            deadline = time.monotonic() + 5
            with runtime.db.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as observer:
                while True:
                    waiting = observer.scalar(
                        text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                        {"pid": backend[0]},
                    )
                    if waiting == "Lock":
                        break
                    assert time.monotonic() < deadline, "Maintenance never waited for the row lock."
                    time.sleep(0.01)
        worker.result(timeout=10)
    with original() as session:
        assert session.get(Job, job.id).status == "running"


def test_real_retirement_rolls_back_with_publication(postgres_runtime):
    runtime, _, _ = postgres_runtime
    with runtime.db.session() as session:
        session.add(IndexState(id=1, collection="previous", model="test", published_at=utcnow()))
    with pytest.raises(RuntimeError, match="Synthetic"), runtime.db.session() as session:
        state = session.get(IndexState, 1)
        retire_index(session, state.collection)
        state.collection = "replacement"
        raise RuntimeError("Synthetic publication failure")
    with runtime.db.session() as session:
        assert session.get(IndexState, 1).collection == "previous"
        assert session.get(RetiredIndex, "previous") is None


def test_real_project_grants_support_readers_and_runtime_role_has_only_required_dml(
    postgres_runtime,
):
    from uuid import UUID

    from ragu.database import Corpus, Project
    from ragu.schemas import CorpusCreate, GrantUpdate, MemberUpdate, ProjectCreate, SearchQuery
    from tests.test_retrieval import SparseEmbeddings

    runtime, principal, root = postgres_runtime
    (root / "policy.txt").write_text("Synthetic BLUE-482 retry policy.", encoding="utf-8")
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    project = runtime.projects.create(ProjectCreate(name="Integration fixture"), principal)
    corpus = runtime.projects.create_corpus(
        project["id"], CorpusCreate(name="Docs", folder=root), principal
    )
    job = runtime.ingestion.start(principal, corpus_id=corpus["id"], index_mode="hybrid")
    assert runtime.ingestion.tick()
    assert runtime.ingestion.status(job.id, principal).status == "succeeded"
    password = secrets.token_urlsafe(24)
    reader = runtime.auth.create_reader("project-reader", password, principal)
    reader_id = UUID(reader["id"])
    runtime.projects.member(project["id"], reader_id, MemberUpdate(role="reader"), principal)
    identity = runtime.auth.credentials("project-reader", password, "integration-fixture")
    with ThreadPoolExecutor(max_workers=1) as pool, runtime.db.session() as reading:
        runtime.auth.current(reading, identity)
        granted = pool.submit(
            runtime.projects.grant,
            corpus["id"],
            reader_id,
            GrantUpdate(permission="read"),
            principal,
        )
        granted.result(timeout=5)
    query = SearchQuery(query="BLUE-482", corpus_id=corpus["id"], candidate_method="hybrid_rrf")
    assert runtime.search.search(query, identity)["results"]
    with runtime.db.engine.begin() as connection:
        connection.execute(text("SET LOCAL ROLE ragu_app"))
        assert connection.scalar(select(Project.id)) == project["id"]
        assert connection.scalar(select(Corpus.index_mode)) == "hybrid"
    runtime.projects.grant(corpus["id"], reader_id, None, principal)
    with pytest.raises(AppError) as revoked:
        runtime.search.search(query, identity)
    assert revoked.value.status == 404


def test_real_multi_owner_schedules_release_account_and_project_locks_between_owners(
    postgres_runtime,
    monkeypatch,
):
    runtime, owner, root = postgres_runtime
    project = runtime.projects.create(ProjectCreate(name="Scheduling fixture"), owner)
    first = runtime.projects.create_corpus(
        project["id"],
        CorpusCreate(name="First", folder=root),
        owner,
    )
    second = runtime.projects.create_corpus(
        project["id"],
        CorpusCreate(name="Second", folder=root),
        owner,
    )
    password = secrets.token_urlsafe(24)
    account = runtime.auth.create_reader("scheduled-reader", password, owner)
    reader_id = UUID(account["id"])
    runtime.projects.member(project["id"], reader_id, MemberUpdate(role="reader"), owner)
    runtime.projects.grant(second["id"], reader_id, GrantUpdate(permission="ingest"), owner)
    reader = runtime.auth.credentials("scheduled-reader", password, "scheduling-fixture")
    for corpus, principal in ((first, owner), (second, reader)):
        runtime.ingestion.configure_schedule(
            DailySchedule(
                corpus_id=corpus["id"],
                enabled=True,
                hour_utc=3,
                minute_utc=0,
            ),
            principal,
        )
    with runtime.db.session() as session:
        for schedule in session.scalars(select(CorpusSchedule)):
            schedule.next_run_at = utcnow() - timedelta(minutes=1)
    project_locked, reader_locked = threading.Event(), threading.Event()
    first_owner = []
    access, current = runtime.projects.project_access, runtime.auth.current

    def observed_access(session, project_id, user_id, **kwargs):
        result = access(session, project_id, user_id, **kwargs)
        if threading.current_thread().name == "ragu-multi-owner-maintenance" and not first_owner:
            first_owner.append(user_id)
            project_locked.set()
            assert reader_locked.wait(5)
        return result

    def observed_current(session, principal, **kwargs):
        result = current(session, principal, **kwargs)
        if threading.current_thread().name == "ragu-concurrent-project-reader":
            reader_locked.set()
        return result

    def maintenance():
        threading.current_thread().name = "ragu-multi-owner-maintenance"
        runtime.ingestion._maintenance()

    def read():
        threading.current_thread().name = "ragu-concurrent-project-reader"
        principal = reader if first_owner[0] == owner.id else owner
        return runtime.projects.corpora(project["id"], principal)

    monkeypatch.setattr(runtime.projects, "project_access", observed_access)
    monkeypatch.setattr(runtime.auth, "current", observed_current)
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(maintenance)
        try:
            assert project_locked.wait(5)
            request = pool.submit(read)
            assert reader_locked.wait(5)
        finally:
            reader_locked.set()
        worker.result(timeout=10)
        assert request.result(timeout=10)
    with runtime.db.session() as session:
        assert len(list(session.scalars(select(Job).where(Job.status == "pending")))) == 1
        assert all(schedule.enabled for schedule in session.scalars(select(CorpusSchedule)))


def test_real_login_rehash_cannot_restore_an_old_password_after_concurrent_change(
    postgres_runtime, monkeypatch
):
    from pwdlib import PasswordHash
    from pwdlib.hashers.argon2 import Argon2Hasher

    from ragu.schemas import PasswordChange

    runtime, principal, _ = postgres_runtime
    old_password, new_password = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    weak = PasswordHash((Argon2Hasher(time_cost=1, memory_cost=8192, parallelism=1),))
    with runtime.db.session() as session:
        session.get(User, principal.id).password_hash = weak.hash(old_password)
    verifying, release, changing = threading.Event(), threading.Event(), threading.Event()
    verify = runtime.auth.hasher.verify_and_update
    original_session = runtime.db.session
    backends = []

    def held_verification(password, stored):
        verifying.set()
        assert release.wait(10)
        return verify(password, stored)

    @contextmanager
    def observed_session():
        with original_session() as session:
            if threading.current_thread().name == "ragu-password-change":
                backends.append(session.scalar(text("SELECT pg_backend_pid()")))
                if len(backends) == 2:
                    changing.set()
            yield session

    def change():
        threading.current_thread().name = "ragu-password-change"
        return runtime.auth.change_password(
            PasswordChange(current_password=old_password, new_password=new_password), principal
        )

    monkeypatch.setattr(runtime.auth.hasher, "verify_and_update", held_verification)
    monkeypatch.setattr(runtime.db, "session", observed_session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        login = pool.submit(
            runtime.auth.credentials, principal.username, old_password, "rehash-test"
        )
        try:
            assert verifying.wait(5)
            changed = pool.submit(change)
            assert changing.wait(5)
            deadline = time.monotonic() + 5
            with runtime.db.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as conn:
                while (
                    conn.scalar(
                        text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                        {"pid": backends[-1]},
                    )
                    != "Lock"
                ):
                    assert time.monotonic() < deadline, "Password change did not wait for login."
                    time.sleep(0.01)
        finally:
            release.set()
        login.result(timeout=10)
        changed.result(timeout=10)
    with original_session() as session:
        user = session.get(User, principal.id)
        assert runtime.auth.hasher.verify(new_password, user.password_hash)
        assert not runtime.auth.hasher.verify(old_password, user.password_hash)
        assert user.auth_version == principal.auth_version + 1
