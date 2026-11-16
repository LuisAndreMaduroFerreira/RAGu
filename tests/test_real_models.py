import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAGU_TEST_REAL_MODELS") != "1",
    reason="Opt in with RAGU_TEST_REAL_MODELS=1 after provisioning local model artifacts.",
)


def test_real_offline_hybrid_retrieval_and_cross_encoder(authorized, tmp_path, monkeypatch):
    runtime, client, auth, *_ = authorized
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    runtime.settings.embedding_offline = True
    runtime.settings.embedding_cache = Path(__file__).resolve().parents[1] / ".ragu" / "models"
    runtime.settings.local_consumption_root = tmp_path
    runtime.vectors._embedder = None
    (tmp_path / "retries.txt").write_text(
        "Retry transient network failures with exponential backoff and jitter. "
        "Stop after three attempts and report the error. Tracking code BLUE-482.",
        encoding="utf-8",
    )
    (tmp_path / "cafeteria.txt").write_text(
        "The cafeteria serves lunch between noon and two. The menu changes every week.",
        encoding="utf-8",
    )
    queued = client.post("/ingestions", json={"index_mode": "hybrid"}, headers=auth)
    assert queued.status_code == 202, queued.text
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{queued.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    for method in ("dense", "bm25", "hybrid_rrf"):
        response = client.post(
            "/documents/search",
            headers=auth,
            json={
                "query": "How do we retry transient network failures?",
                "candidate_method": method,
                "candidate_limit": 10,
                "rerank_method": "cross_encoder",
                "limit": 1,
            },
        )
        assert response.status_code == 200, response.text
        top = response.json()["results"][0]
        assert top["relative_path"] == "retries.txt", top
        assert top["score_type"] == method
        assert isinstance(top["rerank_score"], float)
    candidates = client.post(
        "/documents/candidates",
        headers=auth,
        json={"query": "retry network failures", "candidate_method": "hybrid_rrf", "limit": 2},
    ).json()
    reranked = client.post(
        "/documents/rerank",
        headers=auth,
        json={
            "query": "How do we retry transient network failures?",
            "generation_id": candidates["generation_id"],
            "candidate_ids": [item["candidate_id"] for item in candidates["results"]],
            "limit": 1,
        },
    )
    assert reranked.status_code == 200, reranked.text
    top = reranked.json()["results"][0]
    assert top["relative_path"] == "retries.txt"
    assert top["score"] is None
    assert top["score_type"] == "not_computed"


def test_executable_demo_with_real_offline_vectors_and_reranker(env, tmp_path, monkeypatch):
    from uuid import UUID

    from documentation.demo_e2e import run_demo
    from ragu.database import Corpus

    runtime, client, password, *_ = env
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    runtime.settings.embedding_cache = Path(__file__).resolve().parents[1] / ".ragu" / "models"
    runtime.settings.multi_source_roots = [tmp_path]
    runtime.vectors._embedder = None
    monkeypatch.setattr("documentation.demo_e2e.time.sleep", lambda _: runtime.ingestion.tick())
    result = run_demo(client, tmp_path, "admin", password, mode="hybrid", rerank=True)
    with runtime.db.session() as session:
        corpus = session.get(Corpus, UUID(result["corpus_id"]))
        points, _ = runtime.vectors.client.scroll(corpus.collection, with_vectors=True)
    assert len(points) == 2
    assert all(len(point.vector["dense"]) == 384 for point in points)
    assert points[0].vector["dense"] != points[1].vector["dense"]
    assert all(point.vector["bm25"].indices for point in points)
