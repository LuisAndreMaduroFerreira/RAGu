import secrets

from sqlalchemy import select

from ragu.database import User
from tests.conftest import headers


def test_login_is_case_insensitive_and_never_returns_hash(env):
    _, client, password, _ = env
    auth = headers(client, password, "ADMIN")
    response = client.get("/auth/me", headers=auth)
    assert response.status_code == 200
    assert response.json()["username"] == "admin"
    assert set(response.json()) == {"id", "username", "role"}


def test_anonymous_and_reader_cannot_start_ingestion(env):
    _, client, password, _ = env
    assert client.post("/ingestions").status_code == 401
    assert (
        client.post("/ingestions", headers=headers(client, password, "reader")).status_code == 403
    )


def test_login_failure_is_generic_and_throttled(env):
    _, client, _, _ = env
    assert client.post("/auth/login", json={"username": "admin", "password": "wrong"}).json() == (
        client.post("/auth/login", json={"username": "absent", "password": "wrong"}).json()
    )
    for _ in range(10):
        response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 429


def test_password_change_revokes_old_tokens_and_updates_hash(env):
    runtime, client, password, user_id = env
    auth = headers(client, password)
    new_password = secrets.token_urlsafe(24)
    response = client.post(
        "/auth/password",
        headers=auth,
        json={
            "current_password": password,
            "new_password": new_password,
        },
    )
    assert response.status_code == 200
    assert client.get("/auth/me", headers=auth).status_code == 401
    assert headers(client, new_password)
    with runtime.db.session() as session:
        user = session.get(User, user_id)
        assert user.password_hash.startswith("$argon2id$")
        assert runtime.auth.hasher.verify(new_password, user.password_hash)
        assert user.auth_version == 2


def test_logout_all_and_account_disable_revoke_tokens(env):
    runtime, client, password, user_id = env
    auth = headers(client, password)
    assert client.post("/auth/logout-all", headers=auth).status_code == 200
    assert client.get("/auth/me", headers=auth).status_code == 401
    auth = headers(client, password)
    with runtime.db.session() as session:
        session.get(User, user_id).is_active = False
    assert client.get("/auth/me", headers=auth).status_code == 401


def test_personal_credential_is_encrypted_and_not_returned(authorized):
    runtime, client, auth, user_id = authorized
    token = secrets.token_urlsafe(32)
    response = client.put("/auth/confluence-credential", headers=auth, json={"token": token})
    assert response.status_code == 200
    assert token not in response.text
    with runtime.db.session() as session:
        stored = session.scalar(select(User.confluence_token).where(User.id == user_id))
        assert token not in stored
        assert runtime.auth.cipher.decrypt(stored.encode()).decode() == token


def test_wrong_audience_and_algorithm_are_rejected(authorized):
    import jwt

    runtime, client, auth, _user_id = authorized
    payload = jwt.decode(auth["Authorization"].split()[1], options={"verify_signature": False})
    payload["aud"] = runtime.settings.mcp_resource
    token = jwt.encode(payload, runtime.settings.jwt_secret.get_secret_value(), algorithm="HS256")
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    token = jwt.encode(payload, "", algorithm="none")
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_request_size_and_security_headers(env):
    _, client, _, _ = env
    response = client.post("/auth/login", content="x" * 65537)
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    response = client.get("/auth/me")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_validation_never_echoes_a_password(env):
    _, client, _, _ = env
    password = secrets.token_urlsafe(200)
    response = client.post("/auth/login", json={"username": "admin", "password": password})
    assert response.status_code == 422
    assert password not in response.text


def test_remote_deployment_disables_development_documentation(env):
    from ragu.api import create_app

    runtime, _, _, _ = env
    runtime.settings.public_url = "https://ragu.example.test"
    app = create_app(runtime)
    assert app.docs_url is None
    assert app.openapi_url is None
