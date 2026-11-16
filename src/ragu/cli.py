import argparse
import base64
import csv
import getpass
import json
import logging
import os
import secrets
import subprocess
import time
import warnings
from pathlib import Path

from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from dotenv import dotenv_values
from sqlalchemy import create_engine, select

from ragu.config import ROOT_PAGE_ID, Settings
from ragu.database import Database, Role, User

ROOT = Path(__file__).resolve().parents[2]


def protect_config(path: Path) -> None:
    if os.name == "nt":
        system = Path(os.environ["SystemRoot"]) / "System32"
        result = subprocess.run(
            [str(system / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
        )
        sid = next(csv.reader([result.stdout.strip()]))[1]
        subprocess.run(
            [str(system / "icacls.exe"), str(path), "/inheritance:r", "/grant:r", f"*{sid}:(F)"],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        path.chmod(0o600)


def write_config(values: dict[str, str]) -> None:
    target, temporary = Path(".env"), Path(".env.setup")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            protect_config(temporary)
            for key, value in values.items():
                handle.write(f"{key}={json.dumps(value, ensure_ascii=False)}\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def configuration_values(settings: Settings) -> dict[str, str]:
    values = {
        key: value
        for key, value in dotenv_values(".env", interpolate=False).items()
        if value is not None
    }
    values.pop("RAGU_DATABASE_PASSWORD", None)
    values["RAGU_DATABASE_PASSWORD_B64"] = base64.b64encode(
        settings.database_password.get_secret_value().encode()
    ).decode()
    if settings.confluence_ingestion_token:
        values.pop("RAGU_CONFLUENCE_INGESTION_TOKEN", None)
        values["RAGU_CONFLUENCE_INGESTION_TOKEN_B64"] = base64.b64encode(
            settings.confluence_ingestion_token.get_secret_value().encode()
        ).decode()
    for suffix in ("ROOT", "MAX_FILE_BYTES", "MAX_TOTAL_BYTES"):
        values.pop(f"RAGU_MARKDOWN_{suffix}", None)
        effective = getattr(settings, f"local_consumption_{suffix.lower()}")
        if effective is not None:
            values[f"RAGU_LOCAL_CONSUMPTION_{suffix}"] = str(effective)
    return values


def configure_monitoring(settings: Settings) -> None:
    if settings.monitoring_token:
        print("Monitoring credential already configured; no credentials were changed.")
        return
    values = configuration_values(settings)
    values["RAGU_MONITORING_TOKEN"] = secrets.token_urlsafe(48)
    write_config(values)
    print("Read-only monitoring credential saved in private .env. No secret was displayed.")


def migrate(settings: Settings, password: str) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    engine = create_engine(
        settings.database_url(user="ragu_migrator", password=password), hide_parameters=True
    )
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    finally:
        engine.dispose()


def setup() -> None:
    path = Path(".env")
    if path.exists():
        raise SystemExit(".env already exists. Use 'ragu migrate' to update the database.")
    migration_password = getpass.getpass("Existing ragu_migrator password (hidden): ")
    runtime_password = getpass.getpass("Existing ragu_app password (hidden): ")
    values = {
        "RAGU_DATABASE_PASSWORD_B64": base64.b64encode(runtime_password.encode()).decode(),
        "RAGU_JWT_SECRET": secrets.token_urlsafe(48),
        "RAGU_CREDENTIAL_KEY": Fernet.generate_key().decode(),
        "RAGU_PUBLIC_URL": "http://localhost:8000",
        "RAGU_OAUTH_REDIRECT_URIS": "[]",
        "RAGU_EMBEDDING_OFFLINE": "true",
        "RAGU_ALLOW_CLOUD_CONTEXT": "false",
    }
    settings = Settings(
        database_password=runtime_password,
        jwt_secret=values["RAGU_JWT_SECRET"],
        credential_key=values["RAGU_CREDENTIAL_KEY"],
    )
    migrate(settings, migration_password)
    db = Database(settings.database_url())
    try:
        with db.session() as session:
            accounts = session.execute(
                select(User.username, Role.code).join(Role).where(User.is_active.is_(True))
            ).all()
    finally:
        db.engine.dispose()
    write_config(values)
    print("Migration committed; existing application users and password hashes preserved.")
    print("Roles:", ", ".join(f"{username} ({role})" for username, role in accounts))
    print("Runtime configuration written to ignored .env; keep it private and back it up securely.")
    print("Remote MCP remains disabled until its public HTTPS URL, redirects and approval are set.")


def configure_confluence(settings: Settings) -> None:
    from ragu.auth import AuthService
    from ragu.confluence import ConfluenceSource
    from ragu.schemas import ConfluenceCredential

    ingestion = getpass.getpass("Confluence ingestion PAT (hidden): ")
    personal = getpass.getpass("Your personal Confluence PAT (hidden): ")
    if not ingestion or not personal:
        raise SystemExit("Both credentials are required. No credentials were changed.")
    source = ConfluenceSource(settings)
    try:
        source.page(ROOT_PAGE_ID, ingestion)
        source.page(ROOT_PAGE_ID, personal)
    finally:
        source.client.close()
    username = input("Application username [admin]: ").strip() or "admin"
    password = getpass.getpass("Application password (hidden): ")
    db = Database(settings.database_url())
    try:
        auth = AuthService(db, settings)
        principal = auth.credentials(username, password, "local-cli")
        with db.session() as session:
            auth.current(session, principal, admin=True)
        auth.set_confluence_token(ConfluenceCredential(token=personal), principal)
    finally:
        db.engine.dispose()
    values = configuration_values(settings)
    values.pop("RAGU_CONFLUENCE_INGESTION_TOKEN", None)
    values["RAGU_CONFLUENCE_INGESTION_TOKEN_B64"] = base64.b64encode(ingestion.encode()).decode()
    write_config(values)
    print(
        "Root-page access confirmed. Ingestion credential saved privately; personal PAT encrypted."
    )


def check_login(settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from ragu.api import create_app

    username = input("Application username [admin]: ").strip() or "admin"
    password = getpass.getpass("Application password (hidden): ")
    with TestClient(create_app(), base_url=settings.public_url) as client:
        response = client.post("/auth/login", json={"username": username, "password": password})
        if response.status_code != 200:
            raise SystemExit(f"Login failed: HTTP {response.status_code}.")
        result = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {response.json()['access_token']}"}
        )
        if result.status_code != 200:
            raise SystemExit("Authenticated account lookup failed.")
        print(f"Login verified: {result.json()['username']} ({result.json()['role']}).")


def ingest_local_consumption(settings: Settings) -> None:
    from ragu.runtime import Runtime

    username = input("Application username [admin]: ").strip() or "admin"
    password = getpass.getpass("Application password (hidden): ")
    runtime = Runtime(settings)
    try:
        principal = runtime.auth.credentials(username, password, "local-cli")
        job = runtime.ingestion.start(principal)
        print(f"Local consumption queued: {job.id}. Run 'ragu worker' to process it.")
    finally:
        runtime.close()


def create_admin(settings: Settings) -> None:
    from sqlalchemy import func

    from ragu.auth import AuthService

    username = input("Admin username: ").strip()
    if not username or len(username) > 128:
        raise SystemExit("Username must contain 1-128 characters.")
    password = getpass.getpass("New admin password (15-128 characters): ")
    confirm = getpass.getpass("Repeat password: ")
    if password != confirm or not 15 <= len(password) <= 128:
        raise SystemExit("Passwords must match and contain 15-128 characters.")
    db = Database(settings.database_url())
    try:
        auth = AuthService(db, settings)
        with db.session() as session:
            Database.lock(session, 1001)
            if session.scalar(select(func.count()).select_from(User)) != 0:
                raise SystemExit("Bootstrap is disabled because application users already exist.")
            role = session.scalar(select(Role).where(Role.code == "admin"))
            if role is None:
                raise SystemExit("Run migrations before bootstrap.")
            session.add(
                User(username=username, password_hash=auth.hasher.hash(password), role_id=role.id)
            )
        print("Initial admin created.")
    finally:
        db.engine.dispose()


def create_reader(settings: Settings) -> None:
    from ragu.auth import AuthService

    administrator = input("Administrator username [admin]: ").strip() or "admin"
    password = getpass.getpass("Administrator password (hidden): ")
    db = Database(settings.database_url())
    try:
        auth = AuthService(db, settings)
        principal = auth.credentials(administrator, password, "local-cli")
        username = input("New reader/agent username: ").strip()
        new_password = getpass.getpass("New reader password (15-128 characters): ")
        if new_password != getpass.getpass("Repeat password: "):
            raise SystemExit("Passwords must match. No reader was created.")
        print(json.dumps(auth.create_reader(username, new_password, principal)))
    finally:
        db.engine.dispose()


def main() -> None:
    warnings.simplefilter("error", getpass.GetPassWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="ragu")
    parser.add_argument(
        "command",
        choices=[
            "setup",
            "migrate",
            "create-admin",
            "create-reader",
            "serve",
            "worker",
            "download-model",
            "download-sparse-model",
            "download-reranker",
            "configure-confluence",
            "protect-config",
            "check-login",
            "ingest-markdown",
            "ingest-local-consumption",
            "configure-monitoring",
        ],
    )
    args = parser.parse_args()
    if args.command == "setup":
        setup()
        return
    if args.command == "protect-config":
        protect_config(Path(".env"))
        print("Private configuration restricted to the current OS account.")
        return
    settings = Settings()
    if args.command == "migrate":
        migrate(settings, getpass.getpass("ragu_migrator password: "))
    elif args.command == "create-admin":
        create_admin(settings)
    elif args.command == "create-reader":
        create_reader(settings)
    elif args.command == "configure-confluence":
        configure_confluence(settings)
    elif args.command == "check-login":
        check_login(settings)
    elif args.command in {"ingest-local-consumption", "ingest-markdown"}:
        ingest_local_consumption(settings)
    elif args.command == "configure-monitoring":
        configure_monitoring(settings)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "ragu.api:create_app",
            factory=True,
            host="127.0.0.1",
            port=8000,
            access_log=False,
            proxy_headers=False,
        )
    elif args.command == "download-model":
        from fastembed import TextEmbedding

        TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=str(settings.embedding_cache),
            local_files_only=False,
        )
        print("Embedding model cached. Runtime remains offline.")
    elif args.command == "download-sparse-model":
        from fastembed import SparseTextEmbedding

        SparseTextEmbedding(
            model_name=settings.sparse_model,
            cache_dir=str(settings.embedding_cache),
            language=settings.sparse_language,
            local_files_only=False,
        )
        print("Sparse model provisioned. Runtime remains offline.")
    elif args.command == "download-reranker":
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        TextCrossEncoder(
            model_name=settings.reranking_model,
            cache_dir=str(settings.embedding_cache),
            local_files_only=False,
            providers=["CPUExecutionProvider"],
            threads=settings.model_threads,
        )
        print("Reranking model provisioned. Runtime remains offline.")
    elif args.command == "worker":
        from ragu.runtime import Runtime

        runtime = Runtime(settings)
        try:
            while True:
                runtime.ingestion.tick()
                time.sleep(settings.worker_poll_seconds)
        except KeyboardInterrupt:
            print("Worker stopped.")
        finally:
            runtime.close()
