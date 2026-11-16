from datetime import timedelta

from ragu.config import ROOT_PAGE_ID
from ragu.confluence import Page
from ragu.database import IndexState, Job, LocalConsumptionIndexState, Role, Schedule, User, utcnow
from tests.conftest import ingest


def test_complete_ingestion_and_authorized_search(authorized):
    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    response = client.post(
        "/documents/search", headers=auth, json={"query": "retry failures", "source": "confluence"}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == 2
    assert all(
        item["source_url"].startswith("https://confluence.weareplanet.com/")
        for item in response.json()["results"]
    )
    assert client.get("/documents/123?source=confluence", headers=auth).json()["title"] == "Retries"
    assert client.get("/documents/999?source=confluence", headers=auth).status_code == 404


def test_failed_rerun_preserves_published_index(authorized):
    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    with runtime.db.session() as session:
        old = session.get(IndexState, 1).collection
    runtime.source.failure = True
    job = client.post("/ingestions", headers=auth, json={"source": "confluence"}).json()
    runtime.ingestion.tick()
    assert client.get(f"/ingestions/{job['id']}", headers=auth).json()["status"] == "failed"
    with runtime.db.session() as session:
        assert session.get(IndexState, 1).collection == old


def test_unchanged_pages_reuse_vectors_and_deleted_pages_disappear(authorized):
    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    count = runtime.vectors.embedder.embedded
    ingest(runtime, client, auth)
    assert runtime.vectors.embedder.embedded == count + 1
    del runtime.source.pages["123"]
    ingest(runtime, client, auth)
    response = client.post(
        "/documents/search", headers=auth, json={"query": "retry", "source": "confluence"}
    )
    assert [item["page_id"] for item in response.json()["results"]] == [ROOT_PAGE_ID]


def test_revoked_access_and_stale_passages_are_not_released(authorized):
    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    runtime.source.denied.add("123")
    runtime.source.pages[ROOT_PAGE_ID] = Page(ROOT_PAGE_ID, "Gateway", 2, "Changed", "today")
    response = client.post(
        "/documents/search", headers=auth, json={"query": "retry", "source": "confluence"}
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert client.get("/documents/123?source=confluence", headers=auth).status_code == 404
    assert (
        client.get(f"/documents/{ROOT_PAGE_ID}?source=confluence", headers=auth).status_code == 409
    )


def test_only_one_job_and_schedule_owner_authorization(authorized):
    runtime, client, auth, user_id = authorized
    first = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert first.status_code == 202
    assert (
        client.post("/ingestions", headers=auth, json={"source": "confluence"}).status_code == 409
    )
    runtime.ingestion.tick()
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={"enabled": True, "hour_utc": 3, "minute_utc": 0, "source": "confluence"},
    )
    assert response.status_code == 200
    with runtime.db.session() as session:
        session.get(Schedule, 1).next_run_at = utcnow() - timedelta(minutes=1)
        reader = session.query(Role).filter_by(code="reader").one()
        session.get(User, user_id).role_id = reader.id
    runtime.ingestion.tick()
    with runtime.db.session() as session:
        assert session.get(Schedule, 1).enabled is False


def test_due_schedule_runs_once_and_stale_worker_is_recovered(authorized):
    runtime, client, auth, user_id = authorized
    client.put(
        "/schedules/daily",
        headers=auth,
        json={"enabled": True, "hour_utc": 3, "minute_utc": 0, "source": "confluence"},
    )
    with runtime.db.session() as session:
        session.get(Schedule, 1).next_run_at = utcnow() - timedelta(minutes=1)
    assert runtime.ingestion.tick()
    assert not runtime.ingestion.tick()
    with runtime.db.session() as session:
        job = Job(
            requested_by=user_id,
            status="running",
            started_at=utcnow() - timedelta(hours=1),
            heartbeat_at=utcnow() - timedelta(hours=1),
        )
        session.add(job)
        session.flush()
        identifier = job.id
    runtime.ingestion.tick()
    with runtime.db.session() as session:
        assert session.get(Job, identifier).status == "failed"


def test_document_index_outage_is_a_controlled_error(authorized, monkeypatch):
    from qdrant_client.http.exceptions import UnexpectedResponse

    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)

    def unavailable(**kwargs):
        raise UnexpectedResponse(503, "Unavailable", b"", {})

    monkeypatch.setattr(runtime.vectors.client, "scroll", unavailable)
    response = client.get("/documents/123?source=confluence", headers=auth)
    assert response.status_code == 503
    assert response.json() == {"detail": "The document index is unavailable."}


def test_same_content_in_two_sources_has_independent_embeddings(authorized, tmp_path, monkeypatch):
    runtime, client, auth, _ = authorized
    text = "A synthetic shared retrieval policy."
    runtime.source.pages["123"] = Page("123", "Shared policy", 1, text, "2026-11-12")
    root = tmp_path / "source"
    root.mkdir()
    (root / "shared.md").write_text(text, encoding="utf-8")
    runtime.settings.local_consumption_root = root
    embedded = []
    embed = runtime.vectors.embedder.embed

    def record(texts):
        values = list(texts)
        embedded.extend(values)
        return embed(values)

    monkeypatch.setattr(runtime.vectors.embedder, "embed", record)
    ingest(runtime, client, auth)
    response = client.post("/ingestions", headers=auth)
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    assert (
        client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()["status"]
        == "succeeded"
    )
    assert embedded.count(text) == 2
    with runtime.db.session() as session:
        remote = session.get(IndexState, 1)
        local = session.get(LocalConsumptionIndexState, 1)
        assert remote.collection != local.collection
        identifier = next(iter(local.manifest))
        assert runtime.vectors.page_points(remote.collection, "123")
        assert runtime.vectors.page_points(local.collection, identifier, local.source_root)
