import base64
import secrets
from unittest.mock import Mock

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from ragu.cli import configuration_values, configure_monitoring, write_config
from ragu.config import ConfluenceCloudConfig, Settings


def test_generated_config_preserves_password_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protected = []

    def protect_empty_file(path):
        assert path.read_text() == ""
        protected.append(path)

    monkeypatch.setattr("ragu.cli.protect_config", protect_empty_file)
    password = secrets.token_urlsafe(24) + '${SHOULD_NOT_EXPAND}"\\\n\u00e9'
    write_config(
        {
            "RAGU_DATABASE_PASSWORD_B64": base64.b64encode(password.encode()).decode(),
            "RAGU_JWT_SECRET": secrets.token_urlsafe(48),
            "RAGU_CREDENTIAL_KEY": Fernet.generate_key().decode(),
        }
    )
    settings = Settings()
    assert settings.database_url().password == password
    assert protected
    assert not (tmp_path / ".env.setup").exists()
    assert password not in (tmp_path / ".env").read_text()


def test_remote_database_requires_certificate_verification():
    settings = Settings(
        _env_file=None,
        database_host="postgres.example.test",
        database_password=secrets.token_urlsafe(24),
        jwt_secret=secrets.token_urlsafe(48),
        credential_key=Fernet.generate_key().decode(),
    )
    assert settings.database_url().query["sslmode"] == "verify-full"


def test_invalid_configuration_does_not_disclose_other_secrets():
    password = secrets.token_urlsafe(24)
    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            database_password=password,
            jwt_secret="too-short",
            credential_key=Fernet.generate_key().decode(),
        )
    assert password not in str(error.value)


def test_local_consumption_root_must_be_absolute():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            local_consumption_root="relative-folder",
            database_password=secrets.token_urlsafe(24),
            jwt_secret=secrets.token_urlsafe(48),
            credential_key=Fernet.generate_key().decode(),
        )


@pytest.fixture
def cloud_connection():
    return {
        "site_url": "https://Example-Tenant.atlassian.net/",
        "account_email": "fixture@example.invalid",
        "api_token": secrets.token_urlsafe(32),
        "cloud_id": "2b6a42cc-4fcf-4be6-a05e-bc1c9701cde1",
        "space_id": "101",
        "root_page_id": "202",
        "request_timeout_seconds": "45",
    }


def test_cloud_configuration_loads_from_nested_environment_without_replacing_data_center(
    env, cloud_connection, monkeypatch
):
    runtime, *_ = env
    for key, value in cloud_connection.items():
        monkeypatch.setenv(f"RAGU_CONFLUENCE_CLOUD__{key.upper()}", value)
    settings = Settings(_env_file=None, **runtime.settings.model_dump(exclude={"confluence_cloud"}))
    assert settings.confluence_cloud.site_url == "https://example-tenant.atlassian.net"
    assert settings.confluence_cloud.api_token.get_secret_value() == cloud_connection["api_token"]
    assert str(settings.confluence_cloud.cloud_id) == cloud_connection["cloud_id"]
    assert settings.confluence_cloud.space_id == "101"
    assert settings.confluence_cloud.root_page_id == "202"
    assert settings.confluence_cloud.request_timeout_seconds == 45
    assert settings.confluence_ingestion_token == runtime.settings.confluence_ingestion_token
    assert cloud_connection["api_token"] not in repr(settings.confluence_cloud)
    assert cloud_connection["api_token"] not in settings.confluence_cloud.model_dump_json()


@pytest.mark.parametrize(
    "site_url",
    [
        "http://example.atlassian.net",
        "https://example.atlassian.net.attacker.invalid",
        "https://attacker.invalid",
        "https://127.0.0.1",
        "https://token@example.atlassian.net",
        "https://example.atlassian.net:8443",
        "https://example.atlassian.net/wiki",
        "https://example.atlassian.net?token=hidden",
        "https://example.atlassian.net#fragment",
    ],
)
def test_cloud_configuration_rejects_unapproved_origins_without_disclosing_tokens(
    cloud_connection, site_url
):
    with pytest.raises(ValidationError) as error:
        ConfluenceCloudConfig(**{**cloud_connection, "site_url": site_url})
    assert cloud_connection["api_token"] not in str(error.value)


def test_partial_cloud_configuration_fails_instead_of_silently_enabling_a_source(
    env, cloud_connection
):
    runtime, *_ = env
    incomplete = {**cloud_connection}
    incomplete.pop("root_page_id")
    with pytest.raises(ValidationError, match="root_page_id") as error:
        Settings(
            _env_file=None,
            **{**runtime.settings.model_dump(), "confluence_cloud": incomplete},
        )
    assert cloud_connection["api_token"] not in str(error.value)


def test_monitoring_configuration_preserves_credentials_and_never_prints_token(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ragu.cli.protect_config", lambda path: None)
    password = secrets.token_urlsafe(24) + '${SHOULD_NOT_EXPAND}"\\\n\u00e9'
    ingestion_token = secrets.token_urlsafe(32)
    secret = secrets.token_urlsafe(48)
    key = Fernet.generate_key().decode()
    root = tmp_path / "source"
    write_config(
        {
            "RAGU_DATABASE_PASSWORD_B64": base64.b64encode(password.encode()).decode(),
            "RAGU_JWT_SECRET": secret,
            "RAGU_CREDENTIAL_KEY": key,
            "RAGU_CONFLUENCE_INGESTION_TOKEN_B64": base64.b64encode(
                ingestion_token.encode()
            ).decode(),
            "RAGU_MARKDOWN_ROOT": str(root),
        }
    )
    configure_monitoring(Settings())
    settings = Settings()
    token = settings.monitoring_token.get_secret_value()
    assert len(token) >= 32
    assert settings.database_url().password == password
    assert settings.jwt_secret.get_secret_value() == secret
    assert settings.credential_key.get_secret_value() == key
    assert settings.confluence_ingestion_token.get_secret_value() == ingestion_token
    assert settings.local_consumption_root == root
    configure_monitoring(settings)
    assert Settings().monitoring_token.get_secret_value() == token
    assert token not in capsys.readouterr().out


def test_monitoring_credential_rejects_short_secrets():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_password=secrets.token_urlsafe(24),
            jwt_secret=secrets.token_urlsafe(48),
            credential_key=Fernet.generate_key().decode(),
            monitoring_token="short",
        )


@pytest.mark.parametrize("prefix", ["RAGU_LOCAL_CONSUMPTION", "RAGU_MARKDOWN"])
def test_local_consumption_configuration_accepts_new_and_legacy_properties(
    tmp_path, monkeypatch, prefix
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ragu.cli.protect_config", lambda path: None)
    root = tmp_path / "source"
    write_config(
        {
            "RAGU_DATABASE_PASSWORD_B64": base64.b64encode(b"test-password").decode(),
            "RAGU_JWT_SECRET": secrets.token_urlsafe(48),
            "RAGU_CREDENTIAL_KEY": Fernet.generate_key().decode(),
            f"{prefix}_ROOT": str(root),
            f"{prefix}_MAX_FILE_BYTES": "1234",
            f"{prefix}_MAX_TOTAL_BYTES": "5678",
        }
    )
    settings = Settings()
    assert settings.local_consumption_root == root
    assert settings.local_consumption_max_file_bytes == 1234
    assert settings.local_consumption_max_total_bytes == 5678
    values = configuration_values(settings)
    assert values["RAGU_LOCAL_CONSUMPTION_ROOT"] == str(root)
    assert values["RAGU_LOCAL_CONSUMPTION_MAX_FILE_BYTES"] == "1234"
    assert values["RAGU_LOCAL_CONSUMPTION_MAX_TOTAL_BYTES"] == "5678"
    assert not any(key.startswith("RAGU_MARKDOWN_") for key in values)
    write_config(values)
    assert Settings().local_consumption_root == root


def test_new_local_consumption_properties_take_precedence(tmp_path):
    settings = Settings(
        _env_file=None,
        database_password="test-password",
        jwt_secret=secrets.token_urlsafe(48),
        credential_key=Fernet.generate_key().decode(),
        RAGU_LOCAL_CONSUMPTION_ROOT=tmp_path / "approved",
        RAGU_MARKDOWN_ROOT=tmp_path / "legacy",
    )
    assert settings.local_consumption_root == tmp_path / "approved"


@pytest.mark.parametrize("command", ["ingest-local-consumption", "ingest-markdown"])
def test_local_consumption_cli_and_legacy_alias_dispatch_to_same_plan(monkeypatch, command):
    from ragu import cli

    settings = object()
    consume = Mock()
    monkeypatch.setattr("sys.argv", ["ragu", command])
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "ingest_local_consumption", consume)
    cli.main()
    consume.assert_called_once_with(settings)
