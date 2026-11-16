import secrets
from contextlib import contextmanager
from datetime import timedelta

import pytest
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ragu.database import IndexState, Job, LocalConsumptionIndexState, Role, Schedule, User, utcnow
from tests.conftest import headers, ingest


def samples(response):
    text = response.text if hasattr(response, "text") else response.body.decode()
    assert response.status_code == 200, text
    assert response.headers["content-type"].startswith("text/plain;")
    return [sample for metric in text_string_to_metric_families(text) for sample in metric.samples]


def value(response, name, **labels):
    matching = [
        sample.value
        for sample in samples(response)
        if sample.name == name and sample.labels == labels
    ]
    assert len(matching) == 1, (name, labels, matching)
    return matching[0]


def monitoring(runtime):
    token = secrets.token_urlsafe(48)
    runtime.settings.monitoring_token = SecretStr(token)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "path", ["/actuator/prometheus", "/actuator/health", "/actuator/health/readiness"]
)
def test_management_requires_monitoring_or_current_admin_credentials(env, path):
    runtime, client, password, user_id = env
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Basic not-a-token"}).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
    reader = headers(client, password, "reader")
    assert client.get(path, headers=reader).status_code == 403
    admin = headers(client, password)
    assert client.get(path, headers=admin).status_code == 200
    with runtime.db.session() as session:
        session.get(User, user_id).role_id = session.scalar(
            select(Role.id).where(Role.code == "reader")
        )
    assert client.get(path, headers=admin).status_code == 403
    credential = monitoring(runtime)
    assert client.get(path, headers=credential).status_code == 200


def test_monitoring_credential_is_read_only_scoped_and_can_be_rotated(env):
    runtime, client, *_ = env
    credential = monitoring(runtime)
    assert client.get("/actuator/prometheus", headers=credential).status_code == 200
    assert client.get("/actuator/health/readiness", headers=credential).status_code == 200
    assert client.post("/ingestions", headers=credential).status_code == 401
    assert client.get("/auth/me", headers=credential).status_code == 401
    assert (
        client.post(
            "/documents/search", json={"query": "private text"}, headers=credential
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/mcp",
            headers=credential,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        ).status_code
        == 401
    )
    current = monitoring(runtime)
    assert client.get("/actuator/prometheus", headers=credential).status_code == 401
    assert client.get("/actuator/prometheus", headers=current).status_code == 200


def test_mcp_audience_token_cannot_scrape_or_probe(env):
    runtime, client, password, _ = env
    principal = runtime.auth.credentials("admin", password, "test")
    token = runtime.auth.issue(principal, runtime.settings.mcp_resource, ["documents:read"])
    auth = {"Authorization": f"Bearer {token.access_token}"}
    for path in ("/actuator/prometheus", "/actuator/health/readiness"):
        assert client.get(path, headers=auth).status_code == 401


def test_revoking_admin_session_does_not_grant_or_revoke_static_monitoring_access(env):
    runtime, client, password, _ = env
    credential = monitoring(runtime)
    auth = headers(client, password)
    assert client.post("/auth/logout-all", headers=auth).status_code == 200
    assert client.get("/actuator/prometheus", headers=auth).status_code == 401
    assert client.get("/actuator/prometheus", headers=credential).status_code == 200


def test_http_metrics_use_templates_and_bounded_methods_not_sensitive_data(authorized):
    runtime, client, auth, user_id = authorized
    ingest(runtime, client, auth)
    for identifier in ("first-private-document", "second-private-document"):
        assert (
            client.get(
                f"/documents/{identifier}?source=confluence&token=secret-query", headers=auth
            ).status_code
            == 404
        )
    for identifier in ("unknown-one", "unknown-two"):
        assert client.get(f"/{identifier}?private=query").status_code == 404
    assert client.request("ARBITRARY", "/unknown-three").status_code in {404, 405}
    assert (
        client.post(
            "/documents/search",
            headers=auth,
            json={"query": "private-search-marker", "source": "confluence"},
        ).status_code
        == 200
    )
    response = client.get("/actuator/prometheus", headers=auth)
    assert (
        value(
            response,
            "ragu_http_requests_total",
            method="GET",
            route="/documents/{page_id}",
            status="404",
        )
        == 2
    )
    assert (
        value(
            response,
            "ragu_http_request_duration_seconds_count",
            method="GET",
            route="/documents/{page_id}",
        )
        == 2
    )
    assert (
        value(response, "ragu_http_requests_total", method="OTHER", route="unmatched", status="404")
        == 1
    )
    assert value(response, "ragu_http_requests_in_progress") == 0
    for sensitive in (
        "first-private-document",
        "second-private-document",
        "unknown-one",
        "unknown-two",
        "unknown-three",
        "secret-query",
        "private-search-marker",
        str(user_id),
        auth["Authorization"].removeprefix("Bearer "),
    ):
        assert sensitive not in response.text
    repeated = client.get("/actuator/prometheus", headers=auth)
    assert not any(
        sample.labels.get("route", "").startswith("/actuator/") for sample in samples(repeated)
    )
    assert any(sample.name == "python_gc_objects_collected_total" for sample in samples(response))


def test_requests_rejected_before_routing_are_measured(env):
    runtime, client, *_ = env
    credential = monitoring(runtime)
    assert client.post("/auth/login", content=b"x" * 65537).status_code == 413
    assert client.get("/auth/me", headers={"Host": "untrusted.example"}).status_code == 400
    response = client.get("/actuator/prometheus", headers=credential)
    assert (
        value(response, "ragu_http_requests_total", method="POST", route="unmatched", status="413")
        == 1
    )
    assert (
        value(response, "ragu_http_requests_total", method="GET", route="unmatched", status="400")
        == 1
    )
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_mcp_requests_are_measured_as_a_fixed_transport_route(env):
    runtime, client, *_ = env
    credential = monitoring(runtime)
    assert (
        client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).status_code
        == 401
    )
    response = client.get("/actuator/prometheus", headers=credential)
    assert (
        value(response, "ragu_http_requests_total", method="POST", route="/mcp", status="401") == 1
    )


def test_inflight_and_failed_requests_are_observed_without_masking_errors(env):
    import anyio

    from ragu.management import HttpMetricsMiddleware

    runtime, *_ = env
    credential = monitoring(runtime)["Authorization"].removeprefix("Bearer ")

    async def broken(scope, receive, send):
        response = runtime.management.prometheus(credential)
        assert value(response, "ragu_http_requests_in_progress") == 1
        raise RuntimeError("application error")

    async def unused():
        raise AssertionError("This test never reads the body")

    async def send(message):
        raise AssertionError("The application never produces a response")

    middleware = HttpMetricsMiddleware(broken, runtime.management)
    scope = {"type": "http", "path": "/broken-private-path", "method": "GET"}
    with pytest.raises(RuntimeError, match="application error"):
        anyio.run(middleware, scope, unused, send)
    response = runtime.management.prometheus(credential)
    assert value(response, "ragu_http_requests_in_progress") == 0
    assert (
        value(response, "ragu_http_requests_total", method="GET", route="unmatched", status="500")
        == 1
    )
    assert (
        value(response, "ragu_http_request_duration_seconds_count", method="GET", route="unmatched")
        == 1
    )
    assert "broken-private-path" not in response.body.decode()


def test_durable_ingestion_and_schedule_metrics_cover_separate_worker_state(env):
    runtime, client, _, user_id = env
    credential = monitoring(runtime)
    now = utcnow()
    with runtime.db.session() as session:
        session.add_all(
            [
                Job(
                    requested_by=user_id,
                    source="markdown",
                    status="succeeded",
                    started_at=now - timedelta(seconds=12),
                    finished_at=now - timedelta(seconds=2),
                    page_count=7,
                    chunk_count=19,
                    file_count=9,
                    duplicate_count=2,
                    unchanged=True,
                    error="private-job-error-marker",
                ),
                Job(
                    requested_by=user_id,
                    source="confluence",
                    status="failed",
                    started_at=now - timedelta(seconds=5),
                    finished_at=now,
                ),
                Job(
                    requested_by=user_id,
                    source="markdown",
                    status="pending",
                    created_at=now - timedelta(seconds=15),
                ),
                LocalConsumptionIndexState(
                    id=1,
                    owner_id=user_id,
                    source_root="private-root-marker",
                    collection="private-collection-marker",
                    model=runtime.settings.embedding_model,
                    fingerprint="private-fingerprint-marker",
                    published_at=now - timedelta(days=1),
                    manifest={"private-document-marker": {"path": "private-path-marker.md"}},
                ),
                Schedule(
                    id=1,
                    owner_id=user_id,
                    source="markdown",
                    source_root="private-root-marker",
                    enabled=True,
                    hour_utc=3,
                    minute_utc=0,
                    next_run_at=now + timedelta(hours=1),
                ),
            ]
        )
    response = client.get("/actuator/prometheus", headers=credential)
    assert (
        value(response, "ragu_ingestion_jobs", source="local_consumption", status="succeeded") == 1
    )
    assert value(response, "ragu_ingestion_jobs", source="local_consumption", status="pending") == 1
    assert value(response, "ragu_ingestion_jobs", source="confluence", status="failed") == 1
    assert value(response, "ragu_ingestion_last_duration_seconds", source="local_consumption") == 10
    assert value(response, "ragu_ingestion_last_duration_seconds", source="confluence") == 5
    assert value(response, "ragu_ingestion_last_success_documents", source="local_consumption") == 7
    assert value(response, "ragu_ingestion_last_success_chunks", source="local_consumption") == 19
    assert (
        value(response, "ragu_ingestion_last_success_duplicate_files", source="local_consumption")
        == 2
    )
    assert value(response, "ragu_ingestion_last_success_unchanged", source="local_consumption") == 1
    assert (
        value(response, "ragu_ingestion_last_success_timestamp_seconds", source="local_consumption")
        == (now - timedelta(seconds=2)).timestamp()
    )
    assert (
        value(response, "ragu_index_published_timestamp_seconds", source="local_consumption")
        == (now - timedelta(days=1)).timestamp()
    )
    assert value(response, "ragu_index_published_timestamp_seconds", source="confluence") == 0
    assert value(response, "ragu_schedule_enabled", source="local_consumption") == 1
    assert value(response, "ragu_schedule_enabled", source="confluence") == 0
    assert (
        value(
            response, "ragu_ingestion_oldest_pending_timestamp_seconds", source="local_consumption"
        )
        == (now - timedelta(seconds=15)).timestamp()
    )
    assert "private-" not in response.text
    assert 'source="markdown"' not in response.text


def test_actual_worker_success_failure_and_api_restart_preserve_job_metrics(authorized):
    from ragu.management import ManagementService

    runtime, client, auth, _ = authorized
    ingest(runtime, client, auth)
    with runtime.db.session() as session:
        published = session.get(IndexState, 1).published_at
    runtime.source.failure = True
    failed = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert failed.status_code == 202
    assert runtime.ingestion.tick()
    restarted = ManagementService(
        runtime.db, runtime.auth, runtime.settings, runtime.vectors, runtime.strategies
    )
    token = auth["Authorization"].removeprefix("Bearer ")
    response = restarted.prometheus(token)
    assert value(response, "ragu_ingestion_jobs", source="confluence", status="succeeded") == 1
    assert value(response, "ragu_ingestion_jobs", source="confluence", status="failed") == 1
    assert value(response, "ragu_index_published_timestamp_seconds", source="confluence") == (
        published.replace(tzinfo=utcnow().tzinfo).timestamp()
    )
    assert value(response, "ragu_http_requests_in_progress") == 0
    assert not any(sample.name == "ragu_http_requests_total" for sample in samples(response))


def test_health_distinguishes_liveness_and_dependency_readiness(env, monkeypatch, caplog):
    runtime, client, *_ = env
    credential = monitoring(runtime)
    assert client.get("/actuator/health/liveness").json() == {"status": "UP"}
    assert client.get("/actuator/health", headers=credential).json() == {
        "status": "UP",
        "components": {"database": {"status": "UP"}, "qdrant": {"status": "UP"}},
    }

    def unavailable(**kwargs):
        raise UnexpectedResponse(503, "Unavailable", b"sensitive-body-marker", {})

    monkeypatch.setattr(runtime.vectors.client, "get_collections", unavailable)
    response = client.get("/actuator/health/readiness", headers=credential)
    assert response.status_code == 503
    assert response.json() == {
        "status": "DOWN",
        "components": {"database": {"status": "UP"}, "qdrant": {"status": "DOWN"}},
    }
    assert "sensitive-body-marker" not in response.text + caplog.text
    assert client.get("/actuator/health/liveness").status_code == 200

    def disconnected(**kwargs):
        raise ResponseHandlingException(ConnectionError("private-network-marker"))

    monkeypatch.setattr(runtime.vectors.client, "get_collections", disconnected)
    assert client.get("/actuator/health/readiness", headers=credential).status_code == 503


def test_database_outage_does_not_report_successful_scrape_or_leak_details(env, monkeypatch):
    runtime, client, *_ = env
    credential = monitoring(runtime)

    @contextmanager
    def unavailable():
        raise OperationalError("private-database-marker", {}, ConnectionError())
        yield

    monkeypatch.setattr(runtime.db, "session", unavailable)
    response = client.get("/actuator/health/readiness", headers=credential)
    assert response.status_code == 503
    assert response.json()["components"]["database"] == {"status": "DOWN"}
    assert "private-database-marker" not in response.text
    response = client.get("/actuator/prometheus", headers=credential)
    assert response.status_code == 503
    assert response.json() == {"detail": "Database operation failed."}
    assert client.get("/actuator/health/liveness").json() == {"status": "UP"}
