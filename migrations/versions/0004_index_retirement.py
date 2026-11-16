"""Retain retired generations from their actual retirement, not their job's completion."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.create_table(
        "retired_indexes",
        sa.Column("collection", sa.String(100), primary_key=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retired_indexes_retired_at", "retired_indexes", ["retired_at"])
    op.execute(
        "INSERT INTO retired_indexes (collection, retired_at) "
        "SELECT 'ragu_' || CASE WHEN source = 'confluence' THEN '41276872' ELSE source END "
        "|| '_' || replace(id::text, '-', ''), CURRENT_TIMESTAMP "
        "FROM ingestion_jobs WHERE status IN ('succeeded', 'failed') AND NOT unchanged "
        "AND ('ragu_' || CASE WHEN source = 'confluence' THEN '41276872' ELSE source END "
        "|| '_' || replace(id::text, '-', '')) NOT IN "
        "(SELECT collection FROM index_state UNION SELECT collection FROM markdown_index_state "
        "UNION SELECT collection FROM multi_source_index_state)"
    )
    op.execute(
        "REVOKE ALL ON retired_indexes FROM PUBLIC; "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON retired_indexes TO ragu_app;"
    )


def downgrade() -> None:
    raise RuntimeError("Automatic downgrade is disabled to protect generation retention.")
