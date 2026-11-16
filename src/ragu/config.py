import base64
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlsplit
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

ROOT_PAGE_ID = "41276872"
CONFLUENCE_URL = "https://confluence.weareplanet.com"


class ConfluenceCloudConfig(BaseModel):
    """TODO: Confirm tenant, API-token authentication and document scope before integration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    site_url: str
    account_email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    api_token: SecretStr = Field(min_length=1, max_length=4096)
    cloud_id: UUID | None = None
    space_id: str = Field(pattern=r"^[1-9][0-9]*$")
    root_page_id: str = Field(pattern=r"^[1-9][0-9]*$")
    request_timeout_seconds: int = Field(default=30, ge=1, le=120)

    @field_validator("site_url")
    @classmethod
    def tenant_origin(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.atlassian\.net",
                parsed.netloc,
                re.IGNORECASE,
            )
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Confluence Cloud requires an HTTPS tenant.atlassian.net origin.")
        return f"https://{parsed.hostname}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGU_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        hide_input_in_errors=True,
    )

    database_host: str = "localhost"
    database_port: int = 5886
    database_name: str = "ragu"
    database_user: str = "ragu_app"
    database_password: SecretStr | None = None
    database_password_b64: SecretStr | None = None
    database_ca_bundle: Path | None = None
    jwt_secret: SecretStr = Field(min_length=43)
    credential_key: SecretStr
    public_url: str = "http://localhost:8000"
    access_token_minutes: int = Field(default=15, ge=1, le=60)
    confluence_ingestion_token: SecretStr | None = None
    confluence_ingestion_token_b64: SecretStr | None = None
    confluence_ca_bundle: Path | None = None
    confluence_cloud: ConfluenceCloudConfig | None = None
    local_consumption_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "RAGU_LOCAL_CONSUMPTION_ROOT", "RAGU_MARKDOWN_ROOT", "local_consumption_root"
        ),
    )
    local_consumption_max_file_bytes: int = Field(
        default=5_000_000,
        ge=1,
        validation_alias=AliasChoices(
            "RAGU_LOCAL_CONSUMPTION_MAX_FILE_BYTES",
            "RAGU_MARKDOWN_MAX_FILE_BYTES",
            "local_consumption_max_file_bytes",
        ),
    )
    local_consumption_max_total_bytes: int = Field(
        default=100_000_000,
        ge=1,
        validation_alias=AliasChoices(
            "RAGU_LOCAL_CONSUMPTION_MAX_TOTAL_BYTES",
            "RAGU_MARKDOWN_MAX_TOTAL_BYTES",
            "local_consumption_max_total_bytes",
        ),
    )
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache: Path = Path(".ragu/models")
    embedding_offline: bool = True
    sparse_model: Literal["Qdrant/bm25"] = "Qdrant/bm25"
    sparse_language: Literal["english", "portuguese", "spanish", "french", "german"] = "english"
    reranking_model: Literal["Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"] = (
        "Xenova/ms-marco-MiniLM-L-6-v2"
    )
    model_threads: int = Field(default=2, ge=1, le=32)
    reranking_concurrency: int = Field(default=2, ge=1, le=8)
    reranking_max_characters: int = Field(default=100000, ge=1000, le=200000)
    multi_source_roots: list[Path] = []
    import_redis_url: SecretStr | None = None
    import_cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    mock_confluence_origin: str = "https://confluence.example.invalid"
    oauth_client_id: str = "ragu-copilot"
    oauth_redirect_uris: list[str] = []
    allow_cloud_context: bool = False
    monitoring_token: SecretStr | None = Field(default=None, min_length=32, max_length=512)
    worker_poll_seconds: int = Field(default=5, ge=1, le=60)
    stale_job_seconds: int = Field(default=600, ge=120)
    max_pages: int = Field(default=10000, ge=1, le=100000)
    max_chunks: int = Field(default=100000, ge=1)

    @field_validator("local_consumption_root")
    @classmethod
    def absolute_local_consumption_root(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("Local consumption root must be an absolute directory path.")
        return value

    @field_validator("multi_source_roots")
    @classmethod
    def absolute_import_roots(cls, values: list[Path]) -> list[Path]:
        if any(not value.is_absolute() or ".." in value.parts for value in values):
            raise ValueError("Multi-source allowed roots must be absolute directories.")
        return values

    @model_validator(mode="after")
    def secure_endpoints(self) -> "Settings":
        from cryptography.fernet import Fernet

        if self.database_password and self.database_password_b64:
            raise ValueError("Set one database password setting, not both.")
        if self.database_password_b64:
            self.database_password = SecretStr(
                base64.b64decode(
                    self.database_password_b64.get_secret_value(), validate=True
                ).decode()
            )
        if not self.database_password:
            raise ValueError("A database password is required.")
        if self.confluence_ingestion_token and self.confluence_ingestion_token_b64:
            raise ValueError("Set one ingestion credential setting, not both.")
        if self.confluence_ingestion_token_b64:
            self.confluence_ingestion_token = SecretStr(
                base64.b64decode(
                    self.confluence_ingestion_token_b64.get_secret_value(), validate=True
                ).decode()
            )
        Fernet(self.credential_key.get_secret_value().encode())
        if self.import_redis_url:
            redis = urlparse(self.import_redis_url.get_secret_value())
            if (
                not redis.hostname
                or redis.scheme not in {"redis", "rediss"}
                or redis.query
                or redis.fragment
                or redis.path not in {"", "/", "/0"}
                or (
                    redis.scheme != "rediss"
                    and redis.hostname not in {"localhost", "127.0.0.1", "::1"}
                )
            ):
                raise ValueError("Redis requires database 0 and TLS unless using loopback.")
        origin = urlparse(self.mock_confluence_origin)
        if (
            origin.scheme != "https"
            or not origin.hostname
            or origin.username
            or origin.password
            or origin.path
            or origin.query
            or origin.fragment
        ):
            raise ValueError("Mock Confluence origin must be an HTTPS origin.")
        self.public_url = self.public_url.rstrip("/")
        public = urlparse(self.public_url)
        if (
            public.scheme not in {"https", "http"}
            or not public.hostname
            or public.username
            or public.password
            or public.query
            or public.fragment
            or public.path not in {"", "/"}
        ):
            raise ValueError("Public URL must be an origin without credentials or a path.")
        if public.scheme != "https" and public.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Remote access requires HTTPS.")
        for uri in self.oauth_redirect_uris:
            parsed = urlparse(uri)
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
                or (
                    parsed.scheme != "https"
                    and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"})
                )
            ):
                raise ValueError("OAuth redirects require HTTPS or an explicit loopback HTTP URI.")
        qdrant = urlparse(self.qdrant_url)
        if qdrant.scheme != "https" and not (
            qdrant.scheme == "http" and qdrant.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Remote Qdrant requires HTTPS.")
        if qdrant.username or qdrant.password:
            raise ValueError("Use the Qdrant API key setting, not URL credentials.")
        return self

    @property
    def api_audience(self) -> str:
        return f"{self.public_url}/api"

    @property
    def mcp_resource(self) -> str:
        return f"{self.public_url}/mcp"

    def database_url(self, *, user: str | None = None, password: str | None = None) -> URL:
        if self.database_password is None:
            raise ValueError("A database password is required.")
        query = {
            "sslmode": "prefer"
            if self.database_host in {"localhost", "127.0.0.1", "::1"}
            else "verify-full"
        }
        if self.database_ca_bundle:
            query["sslrootcert"] = str(self.database_ca_bundle)
        return URL.create(
            "postgresql+psycopg",
            username=user or self.database_user,
            password=password
            if password is not None
            else self.database_password.get_secret_value(),
            host=self.database_host,
            port=self.database_port,
            database=self.database_name,
            query=query,
        )
