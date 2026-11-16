"""Add an independently owned, exact-chunk multi-source import."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.drop_constraint("ingestion_jobs_source_check", "ingestion_jobs", type_="check")
    op.create_check_constraint(
        "ingestion_jobs_source_check",
        "ingestion_jobs",
        "source IN ('confluence', 'markdown', 'multi_source')",
    )
    op.add_column("ingestion_jobs", sa.Column("parameters", JSONB()))
    for column in ("chunk_reference_count", "reused_chunk_count"):
        op.add_column(
            "ingestion_jobs", sa.Column(column, sa.Integer(), nullable=False, server_default="0")
        )
    op.create_table(
        "multi_source_index_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", UUID(), sa.ForeignKey("app_users.id"), nullable=False),
        sa.Column("collection", sa.String(100), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("manifest", JSONB(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1"),
    )
    op.execute(
        "REVOKE ALL ON multi_source_index_state FROM PUBLIC; "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON multi_source_index_state TO ragu_app;"
    )


def downgrade() -> None:
    raise RuntimeError("Automatic downgrade is disabled to protect published source references.")
