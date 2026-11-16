-- Generated from canonical Alembic migrations; see documentation/README.md.
BEGIN;

CREATE TABLE retired_indexes (
    collection VARCHAR(100) NOT NULL, 
    retired_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (collection)
);

CREATE INDEX ix_retired_indexes_retired_at ON retired_indexes (retired_at);

INSERT INTO retired_indexes (collection, retired_at) SELECT 'ragu_' || CASE WHEN source = 'confluence' THEN '41276872' ELSE source END || '_' || replace(id::text, '-', ''), CURRENT_TIMESTAMP FROM ingestion_jobs WHERE status IN ('succeeded', 'failed') AND NOT unchanged AND ('ragu_' || CASE WHEN source = 'confluence' THEN '41276872' ELSE source END || '_' || replace(id::text, '-', '')) NOT IN (SELECT collection FROM index_state UNION SELECT collection FROM markdown_index_state UNION SELECT collection FROM multi_source_index_state);

REVOKE ALL ON retired_indexes FROM PUBLIC; GRANT SELECT, INSERT, UPDATE, DELETE ON retired_indexes TO ragu_app;;

COMMIT;
