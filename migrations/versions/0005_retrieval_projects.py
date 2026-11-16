"""Versioned retrieval profiles and explicitly granted project corpora."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.execute(
        "INSERT INTO app_roles (code, name) VALUES ('reader', 'Reader') "
        "ON CONFLICT (code) DO NOTHING"
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "project_members",
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("app_users.id"), primary_key=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.CheckConstraint("role IN ('reader', 'maintainer')"),
    )
    op.create_table(
        "corpora",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("source_root", sa.String(64)),
        sa.Column("collection", sa.String(100)),
        sa.Column("model", sa.String(200)),
        sa.Column("index_mode", sa.String(16), nullable=False, server_default="dense"),
        sa.Column("index_profile", sa.JSON()),
        sa.Column("fingerprint", sa.String(64)),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("source IN ('confluence', 'local_consumption', 'multi_source')"),
        sa.CheckConstraint("index_mode IN ('dense', 'sparse', 'hybrid')"),
    )
    op.create_index("ix_corpora_project_id", "corpora", ["project_id"])
    op.create_table(
        "corpus_grants",
        sa.Column("corpus_id", sa.Uuid(), sa.ForeignKey("corpora.id"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("app_users.id"), primary_key=True),
        sa.Column("permission", sa.String(16), nullable=False),
        sa.CheckConstraint("permission IN ('read', 'ingest')"),
    )
    op.create_table(
        "corpus_schedules",
        sa.Column("corpus_id", sa.Uuid(), sa.ForeignKey("corpora.id"), primary_key=True),
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("source_root", sa.String(64)),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("hour_utc", sa.Integer(), nullable=False),
        sa.Column("minute_utc", sa.Integer(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("index_mode", sa.String(16)),
        sa.CheckConstraint("hour_utc BETWEEN 0 AND 23 AND minute_utc BETWEEN 0 AND 59"),
        sa.CheckConstraint("source IN ('confluence', 'markdown')"),
    )
    op.add_column("ingestion_jobs", sa.Column("corpus_id", sa.Uuid(), sa.ForeignKey("corpora.id")))
    for table in (
        "ingestion_jobs",
        "index_state",
        "markdown_index_state",
        "multi_source_index_state",
    ):
        op.add_column(
            table, sa.Column("index_mode", sa.String(16), nullable=False, server_default="dense")
        )
        op.add_column(table, sa.Column("index_profile", sa.JSON()))
        op.create_check_constraint(
            f"{table}_index_mode_check", table, "index_mode IN ('dense', 'sparse', 'hybrid')"
        )
    op.add_column("daily_schedule", sa.Column("index_mode", sa.String(16)))
    op.execute(
        "REVOKE ALL ON projects, project_members, corpora, corpus_grants, corpus_schedules "
        "FROM PUBLIC; GRANT SELECT, INSERT, UPDATE, DELETE "
        "ON projects, project_members, corpora, corpus_grants, corpus_schedules TO ragu_app;"
    )


def downgrade() -> None:
    raise RuntimeError("Automatic downgrade is disabled to protect corpus isolation.")
