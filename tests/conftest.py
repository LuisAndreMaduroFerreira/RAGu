import secrets

import httpx
import numpy as np
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy.pool import StaticPool

from ragu.api import create_app
from ragu.config import ROOT_PAGE_ID, Settings
from ragu.confluence import Page
from ragu.database import Base, Database, Role, User
from ragu.errors import AppError
from ragu.runtime import Runtime
from ragu.vectors import Vectors


class Embeddings:
    def __init__(self):
        self.embedded = 0

    def embed(self, texts):
        for text in texts:
            self.embedded += 1
            yield np.array([1.0, len(text) / 1000, 0.1, 0.2])

    def query_embed(self, query):
        yield np.array([1.0, len(query) / 1000, 0.1, 0.2])


class Source:
    def __init__(self):
        self.client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        )
        self.pages = {
            ROOT_PAGE_ID: Page(ROOT_PAGE_ID, "Gateway", 1, "Project documentation.", "2026-11-10"),
            "123": Page(
                "123", "Retries", 1, "Retry transient failures with backoff.", "2026-11-10"
            ),
        }
        self.denied = set()
        self.failure = False

    def crawl(self, token, heartbeat):
        heartbeat()
        if self.failure:
            raise AppError(503, "Confluence listing incomplete.")
        return list(self.pages.values())

    def page(self, page_id, token):
        if token != "personal-test-token" or page_id in self.denied or page_id not in self.pages:
            raise AppError(404, "Not accessible.")
        return self.pages[page_id]


@pytest.fixture
def env():
    settings = Settings(
        _env_file=None,
        database_password="unused-test-password",
        jwt_secret=secrets.token_urlsafe(48),
        credential_key=Fernet.generate_key().decode(),
        confluence_ingestion_token="ingestion-test-token",
        allow_cloud_context=True,
        oauth_redirect_uris=["http://127.0.0.1:7777/callback"],
    )
    db = Database("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(db.engine)
    source = Source()
    embeddings = Embeddings()
    vectors = Vectors(settings, QdrantClient(":memory:"), embeddings)
    runtime = Runtime(settings, db=db, source=source, vectors=vectors)
    password = secrets.token_urlsafe(24)
    with db.session() as session:
        admin_role = Role(code="admin", name="Administrator")
        reader_role = Role(code="reader", name="Reader")
        session.add_all([admin_role, reader_role])
        session.flush()
        admin = User(
            username="admin",
            password_hash=runtime.auth.hasher.hash(password),
            role_id=admin_role.id,
        )
        reader = User(
            username="reader",
            password_hash=runtime.auth.hasher.hash(password),
            role_id=reader_role.id,
        )
        session.add_all([admin, reader])
        session.flush()
        admin.confluence_token = runtime.auth.cipher.encrypt(b"personal-test-token").decode()
        reader.confluence_token = runtime.auth.cipher.encrypt(b"personal-test-token").decode()
        admin_id = admin.id
    with TestClient(create_app(runtime)) as client:
        yield runtime, client, password, admin_id


def headers(client, password, username="admin"):
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def authorized(env):
    runtime, client, password, user_id = env
    return runtime, client, headers(client, password), user_id


def ingest(runtime, client, auth):
    response = client.post("/ingestions", headers=auth, json={"source": "confluence"})
    assert response.status_code == 202, response.text
    assert runtime.ingestion.tick()
    status = client.get(f"/ingestions/{response.json()['id']}", headers=auth)
    assert status.json()["status"] == "succeeded", status.text
    return response.json()["id"]
