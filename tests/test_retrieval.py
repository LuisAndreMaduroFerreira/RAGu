import hashlib
import re
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import select

from ragu.database import (
    Corpus,
    CorpusSchedule,
    Job,
    LocalConsumptionIndexState,
    User,
    aware,
    utcnow,
)
from ragu.errors import AppError
from ragu.schemas import IndexMode
from tests.conftest import headers


class SparseEmbeddings:
    def embed(self, texts):
        for text in texts:
            terms = sorted(set(re.findall(r"[\w-]+", text.lower())))
            yield SimpleNamespace(
                indices=np.array(
                    [int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) for term in terms],
                    dtype=np.uint32,
                ),
                values=np.ones(len(terms)),
            )

    def query_embed(self, query):
        return self.embed([query])


class CrossEncoder:
    def rerank(self, query, documents, batch_size=16):
        for text in documents:
            yield 10.0 if "BLUE-482" in text else -2.0


@pytest.fixture
def retrieval(authorized, tmp_path):
    runtime, client, auth, user_id = authorized
    runtime.settings.local_consumption_root = tmp_path
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    runtime.reranker._model = CrossEncoder()
    (tmp_path / "alpha.txt").write_text(
        "The retry validation code is BLUE-482. Review within 72 hours.", encoding="utf-8"
    )
    (tmp_path / "beta.txt").write_text(
        "Deployment documentation requires a reviewer and 48 hours notice.", encoding="utf-8"
    )
    return runtime, client, auth, user_id, tmp_path


def publish(context, mode=None, *, corpus_id=None, auth=None, source=None):
    runtime, client, default_auth, *_ = context
    body = {}
    if mode is not None:
        body["index_mode"] = mode
    if corpus_id is not None:
        body["corpus_id"] = corpus_id
    if source is not None:
        body["source"] = source
    response = client.post("/ingestions", json=body, headers=auth or default_auth)
    assert response.status_code == 202, response.text
    assert runtime.ingestion.tick()
    result = client.get(f"/ingestions/{response.json()['id']}", headers=auth or default_auth)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "succeeded", result.text
    return result.json()


def search(context, **options):
    _, client, auth, *_ = context
    return client.post("/documents/search", json={"query": "BLUE-482", **options}, headers=auth)


@pytest.mark.parametrize(
    "mode, methods",
    [
        ("dense", ["dense"]),
        ("sparse", ["bm25"]),
        ("hybrid", ["dense", "bm25", "hybrid_rrf"]),
    ],
)
def test_index_modes_expose_only_supported_methods_and_sparse_never_embeds_dense(
    retrieval, mode, methods
):
    runtime, client, auth, *_ = retrieval
    before = runtime.vectors.embedder.embedded
    job = publish(retrieval, mode)
    assert job["index_mode"] == mode
    capabilities = client.get("/documents/capabilities", headers=auth)
    assert capabilities.status_code == 200, capabilities.text
    assert capabilities.json()["candidate_methods"] == methods
    for method in ("dense", "bm25", "hybrid_rrf"):
        response = search(retrieval, candidate_method=method)
        if method in methods:
            assert response.status_code == 200, response.text
            assert response.json()["results"]
            assert response.json()["results"][0]["score_type"] == method
        else:
            assert response.status_code == 409, response.text
    if mode == "sparse":
        assert runtime.vectors.embedder.embedded == before


def test_hybrid_upgrade_reuses_dense_vectors_and_inherits_mode_on_noop(retrieval):
    runtime, client, auth, *_ = retrieval
    first = publish(retrieval)
    count = runtime.vectors.embedder.embedded
    upgraded = publish(retrieval, "hybrid")
    assert not upgraded["unchanged"]
    assert runtime.vectors.embedder.embedded == count
    assert upgraded["id"] != first["id"]
    repeated = publish(retrieval)
    assert repeated["unchanged"]
    assert repeated["index_mode"] == "hybrid"
    capability = client.get("/documents/capabilities", headers=auth).json()
    assert capability["generation_id"] == upgraded["id"]
    assert capability["index_profile"]["mode"] == "hybrid"


@pytest.mark.parametrize("source", ["local_consumption", "confluence"])
@pytest.mark.parametrize("scoped", [False, True])
def test_sparse_evidence_survives_unrelated_dense_model_changes(retrieval, source, scoped):
    runtime, client, auth, *_ = retrieval
    corpus_id = create_project_corpus(retrieval, source=source)[1] if scoped else None
    selector = {"corpus_id": corpus_id} if scoped else {"source": source}
    before = runtime.vectors.embedder.embedded
    published = publish(retrieval, "sparse", **selector)
    body = {"query": "retry", "candidate_method": "bm25", "limit": 1, **selector}
    initial = client.post("/documents/candidates", headers=auth, json=body)
    assert initial.status_code == 200, initial.text
    candidate = initial.json()["results"][0]
    runtime.settings.embedding_model = "unrelated-dense-model"
    response = client.post("/documents/search", headers=auth, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["generation_id"] == published["id"]
    params = {"corpus_id": corpus_id} if scoped else {"source": source}
    assert client.get("/documents/capabilities", headers=auth, params=params).status_code == 200
    document = client.get(f"/documents/{candidate['page_id']}", headers=auth, params=params)
    assert document.status_code == 200, document.text
    reranked = client.post(
        "/documents/rerank",
        headers=auth,
        json={
            "query": "retry",
            "generation_id": published["id"],
            "candidate_ids": [candidate["candidate_id"]],
            "limit": 1,
            **selector,
        },
    )
    assert reranked.status_code == 200, reranked.text
    repeated = publish(retrieval, **selector)
    if source == "local_consumption":
        assert repeated["unchanged"]
    assert runtime.vectors.embedder.embedded == before
    runtime.settings.sparse_language = "portuguese"
    assert client.post("/documents/search", headers=auth, json=body).status_code == 409


@pytest.mark.parametrize("source", ["local_consumption", "confluence"])
def test_pending_sparse_jobs_ignore_dense_model_changes(retrieval, source):
    runtime, client, auth, *_ = retrieval
    before = runtime.vectors.embedder.embedded
    queued = client.post(
        "/ingestions",
        headers=auth,
        json={"source": source, "index_mode": "sparse"},
    )
    assert queued.status_code == 202, queued.text
    runtime.settings.embedding_model = "unrelated-dense-model"
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    assert runtime.vectors.embedder.embedded == before


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_indexes_without_profile_metadata_still_validate_the_dense_model(retrieval, mode):
    runtime, *_ = retrieval
    publish(retrieval, mode)
    with runtime.db.session() as session:
        session.get(LocalConsumptionIndexState, 1).index_profile = None
    runtime.settings.embedding_model = "changed-dense-model"
    assert search(retrieval).status_code == 409


@pytest.mark.parametrize("failure", ["unavailable", "unapproved"])
def test_failed_scheduled_source_does_not_stop_unrelated_ingestion(retrieval, failure, caplog):
    runtime, client, auth, _, root = retrieval
    folder = root / "scheduled"
    folder.mkdir()
    document = folder / "policy.txt"
    document.write_text("Synthetic scheduled policy.", encoding="utf-8")
    _, corpus_id = create_project_corpus(retrieval, folder=folder)
    published = publish(retrieval, corpus_id=corpus_id)
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={"corpus_id": corpus_id, "enabled": True, "hour_utc": 3, "minute_utc": 0},
    )
    assert response.status_code == 200, response.text
    due = utcnow()
    with runtime.db.session() as session:
        session.get(CorpusSchedule, UUID(corpus_id)).next_run_at = due
        collection = session.get(Corpus, UUID(corpus_id)).collection
    if failure == "unavailable":
        document.unlink()
        folder.rmdir()
    else:
        other = root / "other-approved"
        other.mkdir()
        runtime.settings.multi_source_roots = [other]
    queued = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert queued.status_code == 202, queued.text
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    with runtime.db.session() as session:
        schedule = session.get(CorpusSchedule, UUID(corpus_id))
        assert schedule.enabled == (failure == "unavailable")
        assert aware(schedule.next_run_at) == due
        assert session.get(Corpus, UUID(corpus_id)).collection == collection
    event = (
        "schedule_source_unavailable"
        if failure == "unavailable"
        else ("schedule_disabled_invalid_source")
    )
    assert event in caplog.text
    assert corpus_id in caplog.text
    assert str(folder) not in caplog.text
    assert published["id"] != queued.json()["id"]
    if failure == "unavailable":
        folder.mkdir()
        document.write_text("Synthetic recovered policy.", encoding="utf-8")
        assert runtime.ingestion.tick()
        jobs = client.get("/ingestions", headers=auth, params={"corpus_id": corpus_id}).json()
        assert jobs[0]["status"] == "succeeded"
        assert jobs[0]["id"] != published["id"]


@pytest.mark.parametrize(
    "error", [AppError(500, "Synthetic source failure."), RuntimeError("Synthetic source failure.")]
)
def test_unexpected_scheduled_source_errors_are_not_silenced(retrieval, monkeypatch, error):
    runtime, client, auth, *_ = retrieval
    _, corpus_id = create_project_corpus(retrieval)
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={"corpus_id": corpus_id, "enabled": True, "hour_utc": 3, "minute_utc": 0},
    )
    assert response.status_code == 200, response.text
    with runtime.db.session() as session:
        session.get(CorpusSchedule, UUID(corpus_id)).next_run_at = utcnow()

    def broken_source(corpus):
        raise error

    monkeypatch.setattr(runtime.projects, "for_corpus", broken_source)
    with pytest.raises(type(error), match="Synthetic source failure"):
        runtime.ingestion._maintenance()
    with runtime.db.session() as session:
        assert session.get(CorpusSchedule, UUID(corpus_id)).enabled
        assert not list(session.scalars(select(Job.id)))


def test_combined_reranking_returns_model_scores_without_reinterpreting_retrieval_score(retrieval):
    publish(retrieval, "hybrid")
    response = search(
        retrieval,
        candidate_method="hybrid_rrf",
        candidate_limit=20,
        rerank_method="cross_encoder",
        limit=2,
    )
    assert response.status_code == 200, response.text
    batch = response.json()
    assert batch["candidate_count"] == 2
    assert not batch["partial"]
    assert "BLUE-482" in batch["results"][0]["passage"]
    assert [item["rerank_score"] for item in batch["results"]] == [10, -2]
    assert all(item["score_type"] == "hybrid_rrf" for item in batch["results"])


def test_separate_candidate_and_rerank_endpoints_are_generation_bound(retrieval):
    _, client, auth, *_ = retrieval
    job = publish(retrieval, "hybrid")
    response = client.post(
        "/documents/candidates",
        headers=auth,
        json={"query": "retry", "candidate_method": "hybrid_rrf", "candidate_limit": 10},
    )
    assert response.status_code == 200, response.text
    batch = response.json()
    assert batch["generation_id"] == job["id"]
    assert all("rerank_score" not in item for item in batch["results"])
    body = {
        "query": "retry",
        "generation_id": batch["generation_id"],
        "candidate_ids": [item["candidate_id"] for item in batch["results"]],
        "limit": 1,
    }
    reranked = client.post("/documents/rerank", headers=auth, json=body)
    assert reranked.status_code == 200, reranked.text
    assert "BLUE-482" in reranked.json()["results"][0]["passage"]
    body["generation_id"] = str(uuid4())
    assert client.post("/documents/rerank", headers=auth, json=body).status_code == 409
    body["generation_id"] = batch["generation_id"]
    body["candidate_ids"] = [str(uuid4())]
    assert client.post("/documents/rerank", headers=auth, json=body).status_code == 404


def test_source_change_after_candidates_and_during_reranking_fails_closed(retrieval):
    runtime, client, auth, _, root = retrieval
    publish(retrieval)
    batch = search(retrieval).json()
    body = {
        "query": "retry",
        "generation_id": batch["generation_id"],
        "candidate_ids": [item["candidate_id"] for item in batch["results"]],
        "limit": 1,
    }
    (root / "alpha.txt").write_text("Changed before reranking.", encoding="utf-8")
    assert client.post("/documents/rerank", headers=auth, json=body).status_code == 409
    publish(retrieval)

    class MutatingEncoder:
        def rerank(self, query, documents, batch_size=16):
            (root / "alpha.txt").write_text("Changed during inference.", encoding="utf-8")
            return [1.0] * len(documents)

    runtime.reranker._model = MutatingEncoder()
    assert search(retrieval, rerank_method="cross_encoder").status_code == 409


@pytest.mark.parametrize(
    "bad",
    [
        {"candidate_method": "unknown"},
        {"candidate_limit": 101},
        {"candidate_limit": 1, "limit": 2},
        {"rerank_method": "remote-provider"},
        {"model_url": "https://example.invalid/model"},
        {"query": " "},
    ],
)
def test_search_rejects_unsupported_or_unbounded_options(retrieval, bad):
    response = search(retrieval, **bad)
    assert response.status_code == 422, response.text


def test_requested_reranker_failure_has_no_success_shaped_fallback(retrieval):
    runtime, *_ = retrieval
    publish(retrieval)

    class Broken:
        def rerank(self, *args, **kwargs):
            raise RuntimeError("synthetic model failure")

    runtime.reranker._model = Broken()
    assert search(retrieval, rerank_method="cross_encoder").status_code == 503
    assert search(retrieval).status_code == 200
    runtime.reranker._model = SimpleNamespace(rerank=lambda *args, **kwargs: [float("nan"), 1.0])
    assert search(retrieval, rerank_method="cross_encoder").status_code == 503


def test_reranker_budget_and_concurrency_are_bounded(retrieval):
    runtime, *_ = retrieval
    publish(retrieval)
    runtime.settings.reranking_max_characters = 1
    assert search(retrieval, rerank_method="cross_encoder").status_code == 413
    runtime.settings.reranking_max_characters = 100000
    slots = runtime.reranker._slots
    for _ in range(runtime.settings.reranking_concurrency):
        assert slots.acquire(blocking=False)
    try:
        assert search(retrieval, rerank_method="cross_encoder").status_code == 429
    finally:
        for _ in range(runtime.settings.reranking_concurrency):
            slots.release()


def test_missing_generation_is_rebuilt_instead_of_reporting_unchanged(retrieval):
    runtime, *_ = retrieval
    publish(retrieval)
    with runtime.db.session() as session:
        collection = session.get(LocalConsumptionIndexState, 1).collection
    runtime.vectors.client.delete_collection(collection)
    rebuilt = publish(retrieval)
    assert not rebuilt["unchanged"]
    assert search(retrieval).json()["results"]


def test_backend_outage_is_not_treated_as_missing_index(retrieval, monkeypatch):
    runtime, client, auth, *_ = retrieval
    publish(retrieval)
    monkeypatch.setattr(
        runtime.vectors.client,
        "collection_exists",
        lambda *args: (_ for _ in ()).throw(ResponseHandlingException(OSError("offline"))),
    )
    queued = client.post("/ingestions", json={}, headers=auth)
    runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth).json()
    assert status["status"] == "failed"
    assert "unavailable" in status["error"]


def create_project_corpus(context, source="local_consumption", folder=None):
    _, client, auth, *_ = context
    project = client.post("/projects", json={"name": "Synthetic project"}, headers=auth)
    assert project.status_code == 201, project.text
    project_id = project.json()["id"]
    data = {"name": "Documentation", "source": source}
    if folder is not None:
        data["folder"] = str(folder)
    corpus = client.post(f"/projects/{project_id}/corpora", json=data, headers=auth)
    assert corpus.status_code == 201, corpus.text
    return project_id, corpus.json()["id"]


def reader_grant(context, project_id, corpus_id, permission="read"):
    runtime, client, auth, *_ = context
    with runtime.db.session() as session:
        reader_id = session.scalar(select(User.id).where(User.username == "reader"))
    assert (
        client.put(
            f"/projects/{project_id}/members/{reader_id}",
            json={"role": "reader"},
            headers=auth,
        ).status_code
        == 200
    )
    assert (
        client.put(
            f"/corpora/{corpus_id}/grants/{reader_id}",
            json={"permission": permission},
            headers=auth,
        ).status_code
        == 200
    )
    return reader_id


def test_project_corpora_require_both_membership_and_explicit_grant(retrieval, env):
    _, client, _, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    publish(retrieval, "hybrid", corpus_id=corpus_id)
    password = env[2]
    reader = headers(client, password, "reader")
    body = {"query": "retry", "corpus_id": corpus_id, "candidate_method": "hybrid_rrf"}
    assert client.post("/documents/search", json=body, headers=reader).status_code == 404
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    response = client.post("/documents/search", json=body, headers=reader)
    assert response.status_code == 200, response.text
    passage = response.json()["results"][0]
    assert f"corpus_id={corpus_id}" in passage["source_url"]
    assert (
        client.get(
            f"/documents/{passage['page_id']}?corpus_id={corpus_id}", headers=reader
        ).status_code
        == 200
    )
    assert (
        client.post("/documents/search", json={"query": "retry"}, headers=reader).status_code == 403
    )
    assert (
        client.post("/ingestions", json={"corpus_id": corpus_id}, headers=reader).status_code == 404
    )
    assert (
        client.get(f"/projects/{project_id}/corpora", headers=reader).json()[0]["id"] == corpus_id
    )
    _, _, auth, *_ = retrieval
    assert (
        client.delete(f"/projects/{project_id}/members/{reader_id}", headers=auth).status_code
        == 200
    )
    assert client.post("/documents/search", json=body, headers=reader).status_code == 404
    assert client.get("/projects", headers=reader).json() == []


def test_project_writer_can_ingest_without_global_admin_and_revocation_stops_queued_job(
    retrieval, env
):
    runtime, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    reader_id = reader_grant(retrieval, project_id, corpus_id, "ingest")
    reader = headers(client, env[2], "reader")
    publish(retrieval, "sparse", corpus_id=corpus_id, auth=reader)
    queued = client.post("/ingestions", json={"corpus_id": corpus_id}, headers=reader).json()
    client.delete(f"/corpora/{corpus_id}/grants/{reader_id}", headers=auth)
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued['id']}", headers=auth)
    assert status.json()["status"] == "failed"
    assert client.get("/ingestions", headers=reader).status_code == 403


def test_identical_chunk_ids_cannot_cross_corpus_generation_boundaries(retrieval):
    _, client, auth, *_ = retrieval
    _, first = create_project_corpus(retrieval)
    _, second = create_project_corpus(retrieval)
    publish(retrieval, corpus_id=first)
    publish(retrieval, corpus_id=second)
    batch = search(retrieval, corpus_id=first).json()
    body = {
        "query": "retry",
        "corpus_id": second,
        "generation_id": batch["generation_id"],
        "candidate_ids": [batch["results"][0]["candidate_id"]],
        "limit": 1,
    }
    assert client.post("/documents/rerank", json=body, headers=auth).status_code == 409


def test_project_schedules_are_independent_and_inherit_the_published_mode(retrieval):
    runtime, client, auth, *_ = retrieval
    _, corpus_id = create_project_corpus(retrieval)
    publish(retrieval, "hybrid", corpus_id=corpus_id)
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={"enabled": True, "hour_utc": 3, "minute_utc": 0, "corpus_id": corpus_id},
    )
    assert response.status_code == 200, response.text
    with runtime.db.session() as session:
        from uuid import UUID

        session.get(CorpusSchedule, UUID(corpus_id)).next_run_at = utcnow()
    runtime.ingestion.tick()
    with runtime.db.session() as session:
        jobs = list(session.scalars(select(Job).where(Job.corpus_id.is_not(None))))
        assert len(jobs) == 2
        assert all(job.index_mode == IndexMode.HYBRID for job in jobs)
        assert all(job.status == "succeeded" for job in jobs)
    assert client.get("/schedules/daily", headers=auth).json() == {"enabled": False}


def test_index_configuration_is_immutable_for_an_authorized_pending_job(retrieval):
    runtime, client, auth, *_ = retrieval
    response = client.post("/ingestions", json={"index_mode": "hybrid"}, headers=auth)
    runtime.settings.sparse_language = "portuguese"
    runtime.ingestion.tick()
    status = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert status["status"] == "failed"
    assert "profile changed" in status["error"]


def test_project_grant_revocation_during_inference_prevents_release(retrieval, env):
    runtime, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    publish(retrieval, corpus_id=corpus_id)
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    reader = headers(client, env[2], "reader")

    class RevokingEncoder:
        def rerank(self, query, documents, batch_size=16):
            response = client.delete(f"/corpora/{corpus_id}/grants/{reader_id}", headers=auth)
            assert response.status_code == 200
            return [1.0] * len(documents)

    runtime.reranker._model = RevokingEncoder()
    response = client.post(
        "/documents/search",
        headers=reader,
        json={"query": "retry", "corpus_id": corpus_id, "rerank_method": "cross_encoder"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "mode,method", [("dense", "dense"), ("sparse", "bm25"), ("hybrid", "hybrid_rrf")]
)
def test_confluence_modes_keep_live_personal_permission_checks(retrieval, mode, method):
    runtime, client, auth, *_ = retrieval
    queued = client.post(
        "/ingestions", json={"source": "confluence", "index_mode": mode}, headers=auth
    )
    assert queued.status_code == 202
    runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth).json()
    assert status["status"] == "succeeded", status
    body = {
        "query": "Retry",
        "source": "confluence",
        "candidate_method": method,
        "rerank_method": "cross_encoder",
    }
    response = client.post("/documents/search", json=body, headers=auth)
    assert response.status_code == 200, response.text
    assert any(item["page_id"] == "123" for item in response.json()["results"])
    runtime.source.denied.add("123")
    response = client.post("/documents/search", json=body, headers=auth)
    assert response.status_code == 200
    assert all(item["page_id"] != "123" for item in response.json()["results"])


def test_project_mcp_exposes_typed_read_only_retrieval_and_rechecks_grants(retrieval, env):
    import json

    runtime, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    publish(retrieval, "hybrid", corpus_id=corpus_id)
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    principal = runtime.auth.credentials("reader", env[2], "synthetic-mcp-test")
    token = runtime.auth.issue(
        principal,
        runtime.settings.mcp_resource,
        ["documents:read"],
        client_id=runtime.settings.oauth_client_id,
    )
    mcp_headers = {
        "Authorization": f"Bearer {token.access_token}",
        "Accept": "application/json, text/event-stream",
    }

    def invoke(method, params):
        response = client.post(
            "/mcp",
            headers=mcp_headers,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        assert response.status_code == 200, response.text
        return response.json()["result"]

    def call(name, arguments):
        result = invoke("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            return result
        return json.loads(result["content"][0]["text"])

    invoke(
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "project-retrieval-test", "version": "1"},
        },
    )
    tools = invoke("tools/list", {})["tools"]
    assert len(tools) == 5
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
    caps = call("get_retrieval_capabilities", {"corpus_id": corpus_id})
    assert caps["candidate_methods"] == ["dense", "bm25", "hybrid_rrf"]
    batch = call(
        "generate_candidates",
        {
            "query": "retry",
            "corpus_id": corpus_id,
            "candidate_method": "hybrid_rrf",
        },
    )
    result = call(
        "rerank_documents",
        {
            "query": "retry",
            "corpus_id": corpus_id,
            "generation_id": batch["generation_id"],
            "candidate_ids": [item["candidate_id"] for item in batch["results"]],
            "limit": 1,
        },
    )
    assert result["results"][0]["rerank_score"] == 10
    assert client.post("/ingestions", headers=mcp_headers, json={}).status_code == 401
    client.delete(f"/corpora/{corpus_id}/grants/{reader_id}", headers=auth)
    result = call("search_documents", {"query": "retry", "corpus_id": corpus_id})
    assert result["isError"] is True


def test_reader_provisioning_is_admin_only_and_grants_no_implicit_corpus_access(retrieval, env):
    runtime, _, _, *_ = retrieval
    principal = runtime.auth.credentials("admin", env[2], "synthetic-provisioning")
    password = "synthetic-reader-password-482"
    result = runtime.auth.create_reader("project-agent", password, principal)
    assert result["role"] == "reader"
    reader = runtime.auth.credentials("project-agent", password, "synthetic-reader")
    assert runtime.projects.list(reader) == []
    with pytest.raises(AppError) as denied:
        runtime.auth.create_reader("another-agent", password, reader)
    assert denied.value.status == 403


def test_scoped_ingestion_and_schedule_reject_explicit_contradictory_source(retrieval):
    _, client, auth, *_ = retrieval
    _, corpus_id = create_project_corpus(retrieval, source="confluence")
    response = client.post(
        "/ingestions",
        headers=auth,
        json={"corpus_id": corpus_id, "source": "local_consumption"},
    )
    assert response.status_code == 422, response.text
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={
            "corpus_id": corpus_id,
            "source": "local_consumption",
            "enabled": True,
            "hour_utc": 3,
            "minute_utc": 0,
        },
    )
    assert response.status_code == 422, response.text
    publish(retrieval, "hybrid", corpus_id=corpus_id)
    result = search(retrieval, corpus_id=corpus_id, candidate_method="hybrid_rrf")
    assert result.status_code == 200, result.text
    assert result.json()["source"] == "confluence"


def test_changed_published_profile_requires_reindexing_before_queries(retrieval):
    runtime, *_ = retrieval
    publish(retrieval, "hybrid")
    runtime.settings.sparse_language = "portuguese"
    response = search(retrieval, candidate_method="hybrid_rrf")
    assert response.status_code == 409, response.text
    publish(retrieval)
    assert search(retrieval, candidate_method="hybrid_rrf").status_code == 200


def test_grant_revocation_during_candidate_lookup_prevents_unreranked_release(
    retrieval, env, monkeypatch
):
    runtime, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    publish(retrieval, corpus_id=corpus_id)
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    reader = headers(client, env[2], "reader")
    original = runtime.vectors.search

    def revoked_lookup(*args, **kwargs):
        points = original(*args, **kwargs)
        assert (
            client.delete(f"/corpora/{corpus_id}/grants/{reader_id}", headers=auth).status_code
            == 200
        )
        return points

    monkeypatch.setattr(runtime.vectors, "search", revoked_lookup)
    response = client.post(
        "/documents/candidates",
        headers=reader,
        json={"corpus_id": corpus_id, "query": "retry"},
    )
    assert response.status_code == 404, response.text


def test_project_revocation_during_ingestion_preserves_published_generation(
    retrieval, env, monkeypatch
):
    runtime, client, auth, _, root = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    published = publish(retrieval, corpus_id=corpus_id)
    reader_id = reader_grant(retrieval, project_id, corpus_id, "ingest")
    reader = headers(client, env[2], "reader")
    (root / "alpha.txt").write_text(
        "New synthetic content that must not publish.", encoding="utf-8"
    )
    original = runtime.vectors.encode

    def revoked_encoding(*args, **kwargs):
        result = original(*args, **kwargs)
        assert (
            client.delete(f"/corpora/{corpus_id}/grants/{reader_id}", headers=auth).status_code
            == 200
        )
        return result

    monkeypatch.setattr(runtime.vectors, "encode", revoked_encoding)
    queued = client.post("/ingestions", json={"corpus_id": corpus_id}, headers=reader)
    assert queued.status_code == 202, queued.text
    runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth)
    assert status.json()["status"] == "failed", status.text
    caps = client.get(f"/documents/capabilities?corpus_id={corpus_id}", headers=auth)
    assert caps.json()["generation_id"] == published["id"]


def test_project_maintainer_cannot_select_new_server_filesystem_sources(retrieval, env):
    _, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    assert (
        client.put(
            f"/projects/{project_id}/members/{reader_id}",
            headers=auth,
            json={"role": "maintainer"},
        ).status_code
        == 200
    )
    reader = headers(client, env[2], "reader")
    response = client.post(
        f"/projects/{project_id}/corpora",
        headers=reader,
        json={"name": "Unapproved binding"},
    )
    assert response.status_code == 403, response.text


def test_membership_revocation_also_removes_disabled_accounts_and_grants(retrieval):
    from ragu.database import CorpusGrant, ProjectMember

    runtime, client, auth, *_ = retrieval
    project_id, corpus_id = create_project_corpus(retrieval)
    reader_id = reader_grant(retrieval, project_id, corpus_id)
    with runtime.db.session() as session:
        session.get(User, reader_id).is_active = False
    response = client.delete(f"/projects/{project_id}/members/{reader_id}", headers=auth)
    assert response.status_code == 200, response.text
    from uuid import UUID

    with runtime.db.session() as session:
        assert session.get(ProjectMember, (UUID(project_id), reader_id)) is None
        assert session.get(CorpusGrant, (UUID(corpus_id), reader_id)) is None
