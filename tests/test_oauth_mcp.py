import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup


def authorize(env):
    runtime, client, password, _ = env
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    params = {
        "client_id": runtime.settings.oauth_client_id,
        "redirect_uri": runtime.settings.oauth_redirect_uris[0],
        "response_type": "code",
        "resource": runtime.settings.mcp_resource,
        "scope": "documents:read",
        "state": "opaque-state",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    response = client.get("/oauth/authorize", params=params)
    assert response.status_code == 200, response.text
    soup = BeautifulSoup(response.text, "html.parser")
    data = {field["name"]: field["value"] for field in soup.select("input[type=hidden]")}
    data.update(username="admin", password=password)
    return runtime, client, params, data, verifier


def token_request(runtime, params, verifier, code):
    return {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": verifier,
        "client_id": runtime.settings.oauth_client_id,
        "redirect_uri": params["redirect_uri"],
        "resource": params["resource"],
    }


def test_oauth_pkce_flow_replay_and_read_only_mcp(env):
    runtime, client, params, data, verifier = authorize(env)
    response = client.post(
        "/oauth/authorize",
        data=data,
        headers={"Origin": runtime.settings.public_url},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["state"] == ["opaque-state"]
    request = token_request(runtime, params, verifier, query["code"][0])
    response = client.post("/oauth/token", data=request)
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    assert client.post("/oauth/token", data=request).status_code == 400
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}
    assert client.post("/ingestions", headers=headers).status_code == 401
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        },
    )
    assert response.status_code == 200, response.text
    assert {tool["name"] for tool in response.json()["result"]["tools"]} == {
        "search_documents",
        "get_document",
        "get_retrieval_capabilities",
        "generate_candidates",
        "rerank_documents",
    }


def test_csrf_unregistered_redirect_and_wrong_pkce_are_rejected(env):
    runtime, client, params, data, _verifier = authorize(env)
    assert client.post("/oauth/authorize", data=data).status_code == 403
    params["redirect_uri"] = "https://attacker.example/callback"
    assert client.get("/oauth/authorize", params=params).status_code == 400
    response = client.post(
        "/oauth/authorize",
        data=data,
        headers={"Origin": runtime.settings.public_url},
        follow_redirects=False,
    )
    code = parse_qs(urlparse(response.headers["location"]).query)["code"][0]
    request = token_request(
        runtime,
        {**params, "redirect_uri": "http://127.0.0.1:7777/callback"},
        secrets.token_urlsafe(48),
        code,
    )
    assert client.post("/oauth/token", data=request).status_code == 400


def test_anonymous_mcp_and_metadata(env):
    _, client, _, _ = env
    response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
    assert response.status_code == 401
    assert "resource_metadata" in response.headers["www-authenticate"]
    assert client.get("/.well-known/oauth-authorization-server").json()[
        "code_challenge_methods_supported"
    ] == ["S256"]
    assert client.get("/.well-known/oauth-protected-resource/mcp").status_code == 200
