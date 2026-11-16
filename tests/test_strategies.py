import json
from unittest.mock import Mock

import pytest

from ragu.confluence_cloud import ConfluenceCloudSource
from ragu.database import LocalConsumptionIndexState, utcnow
from ragu.errors import AppError
from ragu.ingestion import IngestionService
from ragu.schemas import DailySchedule, IngestionRequest, SearchQuery
from ragu.search import SearchService
from ragu.strategies import LocalConsumptionStrategy, RetrievalStrategy, StrategyRegistry


def test_current_plan_and_request_defaults_are_local_consumption(env):
    runtime, client, *_ = env
    local = runtime.strategies.get()
    assert isinstance(local, LocalConsumptionStrategy)
    assert runtime.strategies.get("local_consumption") is local
    assert runtime.strategies.get("markdown") is local
    assert runtime.strategies.get("confluence") is not local
    assert IngestionRequest().source == "local_consumption"
    assert SearchQuery(query="policy").source == "local_consumption"
    assert DailySchedule(enabled=True, hour_utc=3, minute_utc=0).source == "local_consumption"
    schema = client.get("/openapi.json").json()
    query = next(
        item
        for item in schema["paths"]["/documents/{page_id}"]["get"]["parameters"]
        if item["name"] == "source"
    )
    assert query["schema"]["default"] == "local_consumption"


@pytest.mark.parametrize("source", ["local_consumption", "markdown"])
def test_source_requests_normalize_legacy_alias(source):
    assert IngestionRequest(source=source).source == "local_consumption"
    assert SearchQuery(query="policy", source=source).source == "local_consumption"
    assert (
        DailySchedule(enabled=False, hour_utc=3, minute_utc=0, source=source).source
        == "local_consumption"
    )


def test_unconfigured_local_consumption_never_falls_back_to_confluence(authorized):
    runtime, client, auth, *_ = authorized
    crawl = Mock(
        side_effect=AssertionError("Default local consumption must not contact Confluence")
    )
    runtime.source.crawl = crawl
    response = client.post("/ingestions", headers=auth)
    assert response.status_code == 503
    assert "RAGU_LOCAL_CONSUMPTION_ROOT" in response.json()["detail"]
    response = client.post("/documents/search", headers=auth, json={"query": "policy"})
    assert response.status_code == 409
    assert "local consumption" in response.json()["detail"]
    crawl.assert_not_called()


def test_unknown_sources_fail_closed_in_api_and_service(authorized):
    runtime, client, auth, *_ = authorized
    for method, path, data in (
        (client.post, "/ingestions", {"source": "not-approved"}),
        (client.post, "/documents/search", {"query": "policy", "source": "not-approved"}),
        (
            client.put,
            "/schedules/daily",
            {"enabled": True, "hour_utc": 3, "minute_utc": 0, "source": "not-approved"},
        ),
    ):
        assert method(path, headers=auth, json=data).status_code == 422
    assert client.get("/documents/123?source=not-approved", headers=auth).status_code == 422
    with pytest.raises(AppError) as error:
        runtime.search.document("123", None, "not-approved")
    assert error.value.status == 422


def test_configuring_cloud_never_instantiates_or_exposes_the_stub(monkeypatch, request):
    monkeypatch.setenv(
        "RAGU_CONFLUENCE_CLOUD",
        json.dumps(
            {
                "site_url": "https://example.atlassian.net",
                "account_email": "fixture@example.invalid",
                "api_token": "synthetic-api-token",
                "space_id": "101",
                "root_page_id": "202",
            }
        ),
    )
    constructor = Mock(side_effect=AssertionError("Cloud must remain unwired."))
    monkeypatch.setattr(ConfluenceCloudSource, "__init__", constructor)
    runtime, client, auth, *_ = request.getfixturevalue("authorized")
    assert runtime.settings.confluence_cloud is not None
    with pytest.raises(AppError) as error:
        runtime.strategies.get("confluence_cloud")
    assert error.value.status == 422
    assert (
        client.post("/ingestions", headers=auth, json={"source": "confluence_cloud"}).status_code
        == 422
    )
    assert (
        client.post(
            "/documents/search",
            headers=auth,
            json={"source": "confluence_cloud", "query": "policy"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/documents/202", headers=auth, params={"source": "confluence_cloud"}
        ).status_code
        == 422
    )
    constructor.assert_not_called()


def test_services_dispatch_to_registered_strategy_without_source_branches(env):
    runtime, _, password, _ = env
    principal = runtime.auth.credentials("admin", password, "test")
    strategy = Mock(spec=RetrievalStrategy)
    strategy.source = "local_consumption"
    strategy.storage_source = "markdown"
    strategy.collection_prefix = "markdown"
    strategy.index_model = LocalConsumptionIndexState
    strategy.bind.return_value = "authorized-root"
    strategy.search.return_value = {"results": [], "source": strategy.source}
    strategy.document.return_value = {"page_id": "example", "source": strategy.source}
    strategies = StrategyRegistry(strategy)
    search = SearchService(strategies)
    query = SearchQuery(query="policy")
    assert search.search(query, principal) == strategy.search.return_value
    strategy.search.assert_called_once_with(query, principal)
    assert search.document("example", principal) == strategy.document.return_value
    strategy.document.assert_called_once_with("example", principal)

    def consume(job, heartbeat, authorize):
        heartbeat()
        with runtime.db.session() as session:
            current = authorize(session, job.id)
            current.status, current.finished_at = "succeeded", utcnow()

    strategy.ingest.side_effect = consume
    ingestion = IngestionService(
        runtime.db, runtime.auth, runtime.settings, runtime.vectors, strategies
    )
    job = ingestion.start(principal)
    assert job.source == "local_consumption"
    strategy.bind.assert_called_once()
    assert ingestion.tick()
    strategy.ingest.assert_called_once()
    consumed = strategy.ingest.call_args.args[0]
    assert consumed.id == job.id
    assert consumed.source_root == "authorized-root"
    assert ingestion.status(job.id, principal).status == "succeeded"


def test_registry_can_dispatch_a_future_explicitly_registered_strategy():
    strategy = Mock(spec=RetrievalStrategy)
    strategy.source = strategy.storage_source = "approved-test-source"
    strategy.document.return_value = {"page_id": "evidence"}
    registry = StrategyRegistry(strategy)
    search = SearchService(registry)
    assert search.document("evidence", None, strategy.source) == {"page_id": "evidence"}
    strategy.document.assert_called_once_with("evidence", None)
    with pytest.raises(AppError):
        search.document("evidence", None, "unregistered-source")
    assert strategy.document.call_count == 1
    with pytest.raises(ValueError, match="Duplicate retrieval strategy"):
        StrategyRegistry(strategy, strategy)
