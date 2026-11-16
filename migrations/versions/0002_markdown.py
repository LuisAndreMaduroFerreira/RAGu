"""Add an independently owned, incremental local Markdown source."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    for table in ("ingestion_jobs", "daily_schedule"):
        op.add_column(
            table, sa.Column("source", sa.String(16), nullable=False, server_default="confluence")
        )
        op.add_column(table, sa.Column("source_root", sa.String(64)))
        op.create_check_constraint(
            f"{table}_source_check", table, "source IN ('confluence', 'markdown')"
        )
    op.add_column(
        "ingestion_jobs",
        sa.Column("unchanged", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "ingestion_jobs", sa.Column("file_count", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("duplicate_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "markdown_index_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", UUID(), sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("source_root", sa.String(64), nullable=False),
        sa.Column("collection", sa.String(100), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("manifest", JSONB(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1"),
    )
    op.execute(
        "REVOKE ALL ON markdown_index_state FROM PUBLIC; "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON markdown_index_state TO ragu_app;"
    )


def downgrade() -> None:
    raise RuntimeError("Automatic downgrade is disabled to protect indexed documents.")
