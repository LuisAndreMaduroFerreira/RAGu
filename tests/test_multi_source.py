from uuid import UUID, uuid4

import fakeredis
import numpy as np
import pytest
from pydantic import SecretStr, ValidationError
from redis.exceptions import ConnectionError
from sqlalchemy import select

from ragu.confluence import Page
from ragu.database import Job, MultiSourceIndexState, RetiredIndex, Role, User
from ragu.errors import AppError
from ragu.import_cache import ImportCache
from ragu.multi_source import MockConfluenceProvider
from ragu.schemas import IndexMode, MultiSourceRequest
from tests.conftest import headers


@pytest.fixture
def batch(authorized, tmp_path):
    runtime, client, auth, user_id = authorized
    root = tmp_path / "approved"
    first, second = root / "first", root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    text = "Shared exact policy."
    (first / "shared.md").write_text(text, encoding="utf-8")
    (second / "shared.html").write_text(f"<p>{text}</p>", encoding="utf-8")
    runtime.settings.multi_source_roots = [root]
    runtime.settings.import_redis_url = SecretStr("redis://127.0.0.1:6379/0")
    url = "https://confluence.example.invalid/wiki/spaces/DEMO/pages/123"
    runtime.multi_source.confluence = MockConfluenceProvider(
        {url: Page("123", "Shared policy", 1, text, "2026-11-12")}
    )
    server = fakeredis.FakeServer()
    control = fakeredis.FakeRedis(server=server, decode_responses=True)
    control.set("unrelated", "preserve")
    caches = []

    def cache_factory(job_id):
        cache = ImportCache(runtime.settings, job_id)
        cache.client.close()
        cache.client = fakeredis.FakeRedis(server=server, decode_responses=True)
        caches.append(cache)
        return cache

    runtime.multi_source.cache_factory = cache_factory
    request = {
        "sources": [
            {"kind": "folder", "path": str(first)},
            {"kind": "folder", "path": str(second)},
            {"kind": "confluence", "url": url},
        ]
    }
    yield runtime, client, auth, user_id, request, control, caches
    control.close()


def published(batch):
    with batch[0].db.session() as session:
        return session.get(MultiSourceIndexState, 1)


def execute(batch):
    runtime, client, auth, _, request, *_ = batch
    response = client.post("/ingestions/multi-source", headers=auth, json=request)
    assert response.status_code == 202, response.text
    assert runtime.ingestion.tick()
    return client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_multi_source_modes_keep_unique_chunks_and_separate_citations(batch, mode):
    from tests.test_retrieval import CrossEncoder, SparseEmbeddings

    runtime, client, auth, _, request, *_ = batch
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    runtime.reranker._model = CrossEncoder()
    request["index_mode"] = mode
    status = execute(batch)
    assert status["status"] == "succeeded", status
    assert status["chunk_count"] == 1
    assert status["chunk_reference_count"] == 3
    response = client.post(
        "/documents/search",
        headers=auth,
        json={
            "source": "multi_source",
            "query": "policy",
            "candidate_method": "bm25" if mode == "sparse" else "hybrid_rrf",
            "rerank_method": "cross_encoder",
        },
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert len(results) == 1
    assert len(results[0]["citations"]) == 3
    assert results[0]["rerank_score"] == -2


@pytest.mark.parametrize("scoped", [False, True])
def test_sparse_multi_source_pending_jobs_and_evidence_ignore_dense_configuration(batch, scoped):
    from tests.test_retrieval import SparseEmbeddings, create_project_corpus

    runtime, client, auth, _, request, *_ = batch
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    request["index_mode"] = "sparse"
    corpus_id = create_project_corpus(batch, source="multi_source")[1] if scoped else None
    if scoped:
        request["corpus_id"] = corpus_id
    selector = {"corpus_id": corpus_id} if scoped else {"source": "multi_source"}
    before = runtime.vectors.embedder.embedded
    queued = client.post("/ingestions/multi-source", headers=auth, json=request)
    assert queued.status_code == 202, queued.text
    runtime.settings.embedding_model = "unrelated-dense-model"
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    response = client.post(
        "/documents/search",
        headers=auth,
        json={"query": "policy", "candidate_method": "bm25", "limit": 1, **selector},
    )
    assert response.status_code == 200, response.text
    runtime.settings.embedding_model = "another-unrelated-dense-model"
    response = client.post(
        "/documents/search",
        headers=auth,
        json={"query": "policy", "candidate_method": "bm25", "limit": 1, **selector},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"][0]["citations"]) == 3
    candidate_id = response.json()["results"][0]["candidate_id"]
    request.pop("index_mode")
    repeated = execute(batch)
    assert repeated["status"] == "succeeded", repeated
    assert repeated["index_mode"] == "sparse"
    assert repeated["chunk_count"] == 1
    assert repeated["chunk_reference_count"] == 3
    response = client.post(
        "/documents/search",
        headers=auth,
        json={"query": "policy", "candidate_method": "bm25", "limit": 1, **selector},
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["candidate_id"] == candidate_id
    assert runtime.vectors.embedder.embedded == before


@pytest.mark.parametrize("change", [None, "dense_model", "origin"])
def test_pre_fix_sparse_job_bindings_still_require_the_original_configuration(batch, change):
    from tests.test_retrieval import SparseEmbeddings

    runtime, client, auth, _, request, *_ = batch
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    request["index_mode"] = "sparse"
    old_binding = runtime.multi_source._binding(
        MultiSourceRequest.model_validate(request), IndexMode.DENSE
    )
    queued = client.post("/ingestions/multi-source", headers=auth, json=request)
    assert queued.status_code == 202, queued.text
    with runtime.db.session() as session:
        session.get(Job, UUID(queued.json()["id"])).source_root = old_binding
    if change == "dense_model":
        runtime.settings.embedding_model = "changed-dense-model"
    elif change == "origin":
        runtime.settings.mock_confluence_origin = "https://other.example.invalid"
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth).json()
    assert status["status"] == ("succeeded" if change is None else "failed"), status
    if change is not None:
        assert published(batch) is None


@pytest.mark.parametrize("field", ["sparse_model", "sparse_language"])
def test_pending_sparse_imports_reject_relevant_profile_changes(batch, field):
    runtime, client, auth, _, request, *_ = batch
    request["index_mode"] = "sparse"
    queued = client.post("/ingestions/multi-source", headers=auth, json=request)
    assert queued.status_code == 202, queued.text
    setattr(
        runtime.settings,
        field,
        "portuguese" if field == "sparse_language" else "different-sparse-model",
    )
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth).json()
    assert status["status"] == "failed", status
    assert "profile changed" in status["error"]
    assert published(batch) is None


def test_pre_fix_sparse_generations_keep_their_chunk_ids_and_payloads_readable(batch, monkeypatch):
    from tests.test_retrieval import SparseEmbeddings

    runtime, client, auth, _, request, *_ = batch
    runtime.vectors._sparse_embedder = SparseEmbeddings()
    request["index_mode"] = "sparse"
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime.multi_source,
            "_chunk_profile",
            lambda mode: {"model": runtime.settings.embedding_model},
        )
        status = execute(batch)
    assert status["status"] == "succeeded", status
    candidate_id = runtime.multi_source._chunk_id("Shared exact policy.", IndexMode.DENSE)
    runtime.settings.embedding_model = "changed-dense-model"
    response = client.post(
        "/documents/search",
        headers=auth,
        json={"source": "multi_source", "query": "policy", "candidate_method": "bm25"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["generation_id"] == status["id"]
    assert response.json()["results"][0]["candidate_id"] == candidate_id
    assert len(response.json()["results"][0]["citations"]) == 3


def test_project_multi_source_corpus_uses_the_same_import_and_permission_pipeline(batch):
    from tests.test_retrieval import create_project_corpus

    runtime, client, auth, _, request, *_ = batch
    _, corpus_id = create_project_corpus(batch, source="multi_source")
    request["corpus_id"] = corpus_id
    status = execute(batch)
    assert status["status"] == "succeeded", status
    response = client.post(
        "/documents/search", headers=auth, json={"corpus_id": corpus_id, "query": "policy"}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"][0]["citations"]) == 3
    assert response.json()["corpus_id"] == corpus_id
    assert runtime.vectors.client.count(response.json()["collection"]).count == 1


def test_delegated_importer_cannot_bind_or_expand_a_project_corpus_sources(batch, env):
    from tests.test_retrieval import create_project_corpus, reader_grant

    runtime, client, auth, _, request, *_ = batch
    project_id, corpus_id = create_project_corpus(batch, source="multi_source")
    reader_grant(batch, project_id, corpus_id, "ingest")
    reader = headers(client, env[2], "reader")
    request["corpus_id"] = corpus_id
    response = client.post("/ingestions/multi-source", headers=reader, json=request)
    assert response.status_code == 403, response.text
    assert execute(batch)["status"] == "succeeded"
    response = client.post("/ingestions/multi-source", headers=reader, json=request)
    assert response.status_code == 202, response.text
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{response.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    widened = {"corpus_id": corpus_id, "sources": request["sources"][:1]}
    response = client.post("/ingestions/multi-source", headers=reader, json=widened)
    assert response.status_code == 422, response.text
    assert published(batch) is None


def test_exact_chunks_have_one_point_and_preserve_three_live_source_citations(batch):
    runtime, client, auth, _, _, control, _ = batch
    embedded = runtime.vectors.embedder.embedded
    job = execute(batch)
    assert job["status"] == "succeeded", job
    assert job["source"] == "multi_source"
    assert job["page_count"] == 3
    assert job["chunk_count"] == 1
    assert job["chunk_reference_count"] == 3
    assert job["reused_chunk_count"] == 2
    assert runtime.vectors.embedder.embedded == embedded + 2  # Dimension probe and one input.
    state = published(batch)
    assert runtime.vectors.client.count(state.collection).count == 1
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "multi_source"}
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert len(result["citations"]) == 3
    assert result["passage"] == "Shared exact policy."
    assert {citation["source_kind"] for citation in result["citations"]} == {"folder", "confluence"}
    assert all("approved" not in citation["source_url"] for citation in result["citations"])
    for identifier in state.manifest:
        assert (
            client.get(f"/documents/{identifier}?source=multi_source", headers=auth).status_code
            == 200
        )
    assert list(control.scan_iter("ragu:import:*")) == []
    assert control.get("unrelated") == "preserve"


def test_eviction_cannot_duplicate_points_or_recompute_already_written_inputs(batch, monkeypatch):
    monkeypatch.setattr(ImportCache, "known", lambda self, identifiers: set())
    job = execute(batch)
    assert job["status"] == "succeeded", job
    assert job["chunk_count"] == 1
    assert batch[0].vectors.embedder.embedded == 2


def test_different_chunks_are_not_deduplicated_by_similar_vectors(batch):
    runtime, _, _, _, request, *_ = batch
    folder = request["sources"][0]["path"]
    from pathlib import Path

    (Path(folder) / "different.md").write_text("Shared exact policy!", encoding="utf-8")
    job = execute(batch)
    assert job["status"] == "succeeded"
    assert job["chunk_count"] == 2
    assert runtime.vectors.client.count(published(batch).collection).count == 2


def test_plain_text_participates_in_cross_source_exact_chunk_deduplication(batch):
    from pathlib import Path

    folder = Path(batch[4]["sources"][1]["path"])
    (folder / "shared.html").unlink()
    (folder / "shared.TXT").write_text("Shared exact policy.", encoding="utf-8")
    job = execute(batch)
    assert job["status"] == "succeeded", job
    assert job["chunk_count"] == 1
    assert job["chunk_reference_count"] == 3


def test_deleted_or_changed_sources_are_removed_from_citations_before_reindex(batch):
    from pathlib import Path

    runtime, client, auth, _, request, *_ = batch
    execute(batch)
    first, second, cloud = request["sources"]
    (Path(first["path"]) / "shared.md").unlink()
    runtime.multi_source.confluence.pages[cloud["url"]] = Page(
        "123", "Changed", 2, "Different policy.", "2026-11-13"
    )
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "multi_source"}
    )
    assert len(response.json()["results"][0]["citations"]) == 1
    (Path(second["path"]) / "shared.html").unlink()
    assert (
        client.post(
            "/documents/search", headers=auth, json={"query": "policy", "source": "multi_source"}
        ).json()["results"]
        == []
    )


def test_invalid_scan_never_touches_qdrant_or_redis_or_replaces_publication(batch, monkeypatch):
    from pathlib import Path

    assert execute(batch)["status"] == "succeeded"
    runtime, _, _, _, request, _, caches = batch
    old = published(batch)
    count = len(caches)
    (Path(request["sources"][1]["path"]) / "invalid.md").write_bytes(b"\xff")

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must finish before vector mutation")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    assert execute(batch)["status"] == "failed"
    assert published(batch).collection == old.collection
    assert len(caches) == count


def test_combined_raw_byte_limit_is_checked_before_cache_or_vector_mutation(batch, monkeypatch):
    runtime, _, _, _, request, _, caches = batch
    request["sources"] = request["sources"][:2]
    runtime.settings.local_consumption_max_total_bytes = 40

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw input limits must be validated before vector mutation")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    job = execute(batch)
    assert job["status"] == "failed"
    assert "scan limits" in job["error"]
    assert published(batch) is None
    assert not caches


def test_lost_lease_cleans_temporary_markers_and_preserves_publication(batch, monkeypatch):
    assert execute(batch)["status"] == "succeeded"
    old = published(batch).collection
    remember = ImportCache.remember

    def revoke(cache, identifiers):
        remember(cache, identifiers)
        with batch[0].db.session() as session:
            session.scalar(select(Job).where(Job.status == "running")).status = "failed"

    monkeypatch.setattr(ImportCache, "remember", revoke)
    assert execute(batch)["status"] == "failed"
    assert published(batch).collection == old
    assert not list(batch[5].scan_iter("ragu:import:*"))


def test_successful_empty_sources_replace_previous_corpus_and_remove_documents(batch):
    from pathlib import Path

    assert execute(batch)["status"] == "succeeded"
    old = published(batch)
    runtime, client, auth, _, request, *_ = batch
    first, second = request["sources"][:2]
    (Path(first["path"]) / "shared.md").unlink()
    (Path(second["path"]) / "shared.html").unlink()
    request["sources"] = [first, second]
    job = execute(batch)
    assert job["status"] == "succeeded"
    assert job["page_count"] == job["file_count"] == job["chunk_count"] == 0
    assert job["chunk_reference_count"] == 0
    assert published(batch).collection != old.collection
    assert runtime.vectors.client.count(published(batch).collection).count == 0
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, old.collection) is not None
        assert session.get(RetiredIndex, session.get(MultiSourceIndexState, 1).collection) is None
    identifier = next(iter(old.manifest))
    assert (
        client.get(f"/documents/{identifier}?source=multi_source", headers=auth).status_code == 404
    )


def test_oauth_mcp_can_read_the_multi_source_corpus_but_cannot_start_imports(batch, env):
    import json
    from urllib.parse import parse_qs, urlparse

    from tests.test_oauth_mcp import authorize, token_request

    assert execute(batch)["status"] == "succeeded"
    runtime, client, params, data, verifier = authorize(env)
    consent = client.post(
        "/oauth/authorize",
        data=data,
        headers={"Origin": runtime.settings.public_url},
        follow_redirects=False,
    )
    assert consent.status_code == 303
    code = parse_qs(urlparse(consent.headers["location"]).query)["code"][0]
    token = client.post("/oauth/token", data=token_request(runtime, params, verifier, code))
    assert token.status_code == 200
    auth = {
        "Authorization": f"Bearer {token.json()['access_token']}",
        "Accept": "application/json, text/event-stream",
    }
    assert client.post("/ingestions/multi-source", headers=auth, json=batch[4]).status_code == 401
    initialized = client.post(
        "/mcp",
        headers=auth,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "multi-source-test", "version": "1"},
            },
        },
    )
    assert initialized.status_code == 200
    response = client.post(
        "/mcp",
        headers=auth,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {"query": "policy", "source": "multi_source"},
            },
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result.get("isError") is not True, result
    passages = json.loads(result["content"][0]["text"])["results"]
    assert len(passages) == 1
    assert len(passages[0]["citations"]) == 3


def test_redis_failure_preserves_existing_index(batch, monkeypatch):
    assert execute(batch)["status"] == "succeeded"
    old = published(batch)

    def fail(self, identifiers):
        raise AppError(503, "The import Redis service is unavailable.")

    monkeypatch.setattr(ImportCache, "remember", fail)
    result = execute(batch)
    assert result["status"] == "failed"
    assert "Redis" in result["error"]
    assert published(batch).collection == old.collection
    assert list(batch[5].scan_iter("ragu:import:*")) == []


def test_cleanup_failure_cannot_publish_success_and_markers_still_expire(batch, monkeypatch):
    assert execute(batch)["status"] == "succeeded"
    old = published(batch)

    def fail(self):
        raise AppError(503, "Temporary import cache cleanup failed.")

    monkeypatch.setattr(ImportCache, "clear", fail)
    result = execute(batch)
    assert result["status"] == "failed"
    assert published(batch).collection == old.collection
    keys = list(batch[5].scan_iter("ragu:import:*"))
    assert keys
    assert all(batch[5].ttl(key) > 0 for key in keys)


def test_another_admin_or_reader_cannot_read_or_replace_the_batch_corpus(batch, env):
    execute(batch)
    runtime, client, _, user_id, request, *_ = batch
    password = env[2]
    with runtime.db.session() as session:
        role = session.scalar(select(Role).where(Role.code == "admin"))
        session.add(
            User(
                username="other", password_hash=runtime.auth.hasher.hash(password), role_id=role.id
            )
        )
    identifier = next(iter(published(batch).manifest))
    for name in ("other", "reader"):
        auth = headers(client, password, name)
        assert (
            client.post("/ingestions/multi-source", headers=auth, json=request).status_code == 403
        )
        assert (
            client.post(
                "/documents/search",
                headers=auth,
                json={"query": "policy", "source": "multi_source"},
            ).status_code
            == 403
        )
        assert (
            client.get(f"/documents/{identifier}?source=multi_source", headers=auth).status_code
            == 403
        )
    with runtime.db.session() as session:
        session.get(User, user_id).is_active = False
    assert (
        client.get(f"/documents/{identifier}?source=multi_source", headers=batch[2]).status_code
        == 401
    )


def test_unapproved_folders_urls_and_disabled_confluence_are_rejected(batch, tmp_path):
    runtime, client, auth, _, request, *_ = batch
    outside = tmp_path / "not-approved"
    outside.mkdir()
    bad_sources = [
        {"kind": "folder", "path": str(outside)},
        {"kind": "confluence", "url": "https://169.254.169.254/wiki/spaces/X/pages/1"},
        {"kind": "confluence", "url": "https://[broken"},
        {"kind": "confluence", "url": request["sources"][2]["url"] + "?token=hidden"},
    ]
    for source in bad_sources:
        response = client.post("/ingestions/multi-source", headers=auth, json={"sources": [source]})
        assert response.status_code in {403, 422}, response.text
    runtime.multi_source.confluence = None
    response = client.post(
        "/ingestions/multi-source", headers=auth, json={"sources": [request["sources"][2]]}
    )
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"]


def test_duplicate_sources_and_background_configuration_changes_fail_closed(batch):
    runtime, client, auth, _, request, *_ = batch
    assert (
        client.post(
            "/ingestions/multi-source",
            headers=auth,
            json={"sources": [request["sources"][0], request["sources"][0]]},
        ).status_code
        == 422
    )
    response = client.post("/ingestions/multi-source", headers=auth, json=request)
    assert response.status_code == 202
    runtime.settings.embedding_model = "changed-model"
    assert runtime.ingestion.tick()
    job = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "failed"
    assert published(batch) is None


def test_non_finite_model_output_fails_before_publication(batch, monkeypatch):
    def invalid(texts):
        return (np.array([float("nan"), 0.0, 0.0, 1.0]) for _ in texts)

    monkeypatch.setattr(batch[0].vectors.embedder, "embed", invalid)
    assert execute(batch)["status"] == "failed"
    assert published(batch) is None
    assert list(batch[5].scan_iter("ragu:import:*")) == []


def test_redis_configuration_is_tls_or_loopback_and_never_uses_another_database(env):
    settings = env[0].settings
    from ragu.config import Settings

    for url in ("redis://remote.example:6379/0", "redis://127.0.0.1:6379/1"):
        values = settings.model_dump()
        values["import_redis_url"] = url
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **values)


def test_cache_protocol_failure_is_controlled_and_keys_are_job_scoped(batch):
    settings = batch[0].settings
    cache = ImportCache(settings, uuid4())

    class Broken:
        def ping(self):
            raise ConnectionError("private connection details")

        def close(self):
            pass

    cache.client.close()
    cache.client = Broken()
    with pytest.raises(AppError) as error, cache:
        raise AssertionError("Unavailable Redis must not be used")
    assert error.value.status == 503
    assert "private" not in error.value.message
    cache = batch[0].multi_source.cache_factory(UUID(int=1))
    with cache:
        cache.remember(["one"])
        assert cache.known(["one", "two"]) == {"one"}
        assert batch[5].ttl(cache.prefix + "one") > 0
    assert batch[5].get("unrelated") == "preserve"
