"""Regenerate SQL references without connecting to any database."""

import io
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

ROOT = Path(__file__).resolve().parents[2]
HEADER = "-- Generated from canonical Alembic migrations; see documentation/README.md.\n"
MIGRATIONS = sorted((ROOT / "migrations" / "versions").glob("*.py"))


def generate(*, fresh=False, revision=None):
    output = io.StringIO()
    output.write(HEADER + "BEGIN;\n\n")
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        if fresh:
            Operations(context).create_table(
                "alembic_version",
                sa.Column("version_num", sa.String(32), nullable=False),
                sa.PrimaryKeyConstraint("version_num", name="alembic_version_pkc"),
            )
        previous = None
        for path in MIGRATIONS:
            migration = runpy.run_path(str(path))
            current = migration["revision"]
            if not fresh and current != revision:
                continue
            with patch(
                "sqlalchemy.inspect",
                return_value=SimpleNamespace(get_table_names=lambda: ["alembic_version"]),
            ):
                migration["upgrade"]()
            if fresh:
                if previous is None:
                    Operations(context).execute(
                        f"INSERT INTO alembic_version (version_num) VALUES ('{current}')"
                    )
                else:
                    Operations(context).execute(
                        f"UPDATE alembic_version SET version_num = '{current}' "
                        f"WHERE version_num = '{previous}'"
                    )
                previous = current
    output.write("COMMIT;\n")
    return output.getvalue()


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    (folder / "fresh-schema.sql").write_text(generate(fresh=True), encoding="utf-8")
    (folder / "0005-retrieval-projects.sql").write_text(generate(revision="0005"), encoding="utf-8")
