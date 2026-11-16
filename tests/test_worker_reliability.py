from contextlib import contextmanager
from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import SecretStr
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ragu.config import ROOT_PAGE_ID
from ragu.confluence import Page
from ragu.database import Database, IndexState, Job, RetiredIndex, utcnow
from ragu.schemas import DailySchedule, MultiSourceRequest
from tests.conftest import ingest


@pytest.mark.parametrize(
    "outage",
    [UnexpectedResponse(503, "Unavailable", b"", {}), ResponseHandlingException(ConnectionError())],
)
@pytest.mark.parametrize("operation", ["collection_exists", "delete_collection"])
def test_cleanup_outage_is_retried_without_stopping_ingestion(
    authorized, monkeypatch, caplog, outage, operation
):
    runtime, client, auth, _ = authorized
    collection = "ragu_confluence_cleanup_fixture"
    runtime.vectors.create(collection)
    with runtime.db.session() as session:
        session.add(RetiredIndex(collection=collection, retired_at=utcnow() - timedelta(hours=2)))
    original = getattr(runtime.vectors.client, operation)

    def unavailable(name):
        if name == collection:
            raise outage
        return original(name)

    monkeypatch.setattr(runtime.vectors.client, operation, unavailable)
    ingest(runtime, client, auth)
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, collection) is not None
    assert "index_cleanup_failed" in caplog.text
    monkeypatch.setattr(runtime.vectors.client, operation, original)
    runtime.ingestion.cleanup_indexes()
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, collection) is None
    assert not runtime.vectors.client.collection_exists(collection)


def test_retention_starts_at_replacement_and_completed_cleanup_is_not_repeated(
    authorized, monkeypatch
):
    runtime, client, auth, _ = authorized
    first = UUID(ingest(runtime, client, auth))
    with runtime.db.session() as session:
        old = session.get(IndexState, 1).collection
        session.get(Job, first).finished_at = utcnow() - timedelta(days=1)
    ingest(runtime, client, auth)
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, old) is not None
        current = session.get(IndexState, 1).collection
    runtime.ingestion.cleanup_indexes()
    assert runtime.vectors.client.collection_exists(old)
    with runtime.db.session() as session:
        session.get(RetiredIndex, old).retired_at = utcnow() - timedelta(hours=2)
    checked = []
    exists = runtime.vectors.client.collection_exists

    def record(name):
        checked.append(name)
        return exists(name)

    monkeypatch.setattr(runtime.vectors.client, "collection_exists", record)
    runtime.ingestion.cleanup_indexes()
    runtime.ingestion.cleanup_indexes()
    assert checked == [old]
    assert not exists(old)
    assert exists(current)
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, old) is None


def test_failed_staging_is_retired_without_retiring_the_published_generation(authorized):
    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    with runtime.db.session() as session:
        published = session.get(IndexState, 1).collection
    runtime.source.failure = True
    response = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    identifier = UUID(response.json()["id"])
    with runtime.db.session() as session:
        assert session.get(Job, identifier).status == "failed"
        assert session.get(IndexState, 1).collection == published
        assert session.get(RetiredIndex, published) is None
        assert (
            session.get(RetiredIndex, runtime.ingestion.collection_name(identifier, "confluence"))
            is not None
        )


def test_lost_commit_acknowledgement_preserves_committed_success(authorized, monkeypatch):
    runtime, client, auth, _ = authorized
    response = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert response.status_code == 202
    identifier = UUID(response.json()["id"])
    original = runtime.db.session
    injected = False

    @contextmanager
    def lost_acknowledgement():
        nonlocal injected
        with original() as session:
            yield session
            published = any(
                isinstance(row, Job) and row.status == "succeeded" for row in session.dirty
            )
        if published and not injected:
            injected = True
            raise OperationalError("COMMIT", None, RuntimeError("Synthetic lost acknowledgement"))

    monkeypatch.setattr(runtime.db, "session", lost_acknowledgement)
    assert runtime.ingestion.tick()
    with original() as session:
        job = session.get(Job, identifier)
        state = session.get(IndexState, 1)
        assert injected
        assert job.status == "succeeded"
        assert job.error is None
        assert state.collection == runtime.ingestion.collection_name(identifier, "confluence")
        assert session.get(RetiredIndex, state.collection) is None


def test_expired_lease_outcome_is_not_overwritten_by_the_old_worker(authorized, monkeypatch):
    runtime, client, auth, _ = authorized
    response = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    identifier = UUID(response.json()["id"])

    def expired(job, heartbeat, authorize):
        with runtime.db.session() as session:
            current = session.get(Job, job.id)
            current.status = "failed"
            current.error = "Worker lease expired; rerun ingestion."
        heartbeat()

    monkeypatch.setattr(runtime.strategies.get("confluence"), "ingest", expired)
    assert runtime.ingestion.tick()
    with runtime.db.session() as session:
        assert session.get(Job, identifier).error == "Worker lease expired; rerun ingestion."


@pytest.mark.parametrize("operation", ["start", "multi_source", "schedule"])
def test_ingestion_writers_lock_queue_before_user_rows(env, tmp_path, monkeypatch, operation):
    runtime, _, password, _ = env
    principal = runtime.auth.credentials("admin", password, "test")
    runtime.settings.multi_source_roots = [tmp_path]
    runtime.settings.import_redis_url = SecretStr("redis://127.0.0.1:6379/0")
    order = []
    lock, current = Database.lock, runtime.auth.current

    def observed_lock(session, key):
        order.append(("lock", key))
        return lock(session, key)

    def observed_user(session, principal, **kwargs):
        order.append(("user", principal.id))
        return current(session, principal, **kwargs)

    monkeypatch.setattr(Database, "lock", staticmethod(observed_lock))
    monkeypatch.setattr(runtime.auth, "current", observed_user)
    if operation == "start":
        runtime.ingestion.start(principal, "confluence")
    elif operation == "multi_source":
        runtime.ingestion.start_multi_source(
            principal,
            MultiSourceRequest.model_validate(
                {"sources": [{"kind": "folder", "path": str(tmp_path)}]}
            ),
        )
    else:
        runtime.ingestion.configure_schedule(
            DailySchedule(enabled=False, hour_utc=3, minute_utc=0, source="confluence"), principal
        )
    assert order[:2] == [("lock", 1002), ("user", principal.id)]


@pytest.mark.parametrize("source", ["confluence", "local_consumption"])
def test_long_pages_renew_leases_between_embedding_and_write_batches(
    authorized, tmp_path, monkeypatch, source
):
    runtime, client, auth, _ = authorized
    text = "\n\n".join(f"Policy {number}: " + "synthetic policy. " * 40 for number in range(2400))
    runtime.source.pages[ROOT_PAGE_ID] = Page(ROOT_PAGE_ID, "Large page", 1, text, "today")
    (tmp_path / "large.txt").write_text(text, encoding="utf-8")
    runtime.settings.local_consumption_root = tmp_path
    start, elapsed = utcnow(), 0
    batches = []
    embed, upsert = runtime.vectors.embedder.embed, runtime.vectors.client.upsert

    def advance():
        nonlocal elapsed
        elapsed += 29
        runtime.ingestion._maintenance()

    def slow_embed(texts):
        batch = list(texts)
        batches.append(len(batch))
        result = list(embed(batch))
        advance()
        return result

    def slow_upsert(*args, **kwargs):
        result = upsert(*args, **kwargs)
        advance()
        return result

    monkeypatch.setattr("ragu.ingestion.utcnow", lambda: start + timedelta(seconds=elapsed))
    monkeypatch.setattr(runtime.vectors.embedder, "embed", slow_embed)
    monkeypatch.setattr(runtime.vectors.client, "upsert", slow_upsert)
    response = client.post("/ingestions", headers=auth, json={"source": source})
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    identifier = UUID(response.json()["id"])
    with runtime.db.session() as session:
        assert session.get(Job, identifier).status == "succeeded"
        assert session.scalar(select(RetiredIndex.collection)) is None
    assert elapsed > runtime.settings.stale_job_seconds
    assert len(batches) > 2
    assert max(batches) <= 100
    if source == "local_consumption":
        (tmp_path / "additional.txt").write_text("An additional policy.", encoding="utf-8")
    scroll = runtime.vectors.client.scroll
    scrolls = 0

    def slow_scroll(*args, **kwargs):
        nonlocal scrolls
        result = scroll(*args, **kwargs)
        scrolls += 1
        advance()
        return result

    monkeypatch.setattr(runtime.vectors.client, "scroll", slow_scroll)
    response = client.post("/ingestions", headers=auth, json={"source": source})
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    with runtime.db.session() as session:
        assert session.get(Job, UUID(response.json()["id"])).status == "succeeded"
    assert scrolls * 29 > runtime.settings.stale_job_seconds
