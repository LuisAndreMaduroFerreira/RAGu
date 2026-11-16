-- Generated from canonical Alembic migrations; see documentation/README.md.
BEGIN;

ALTER TABLE ingestion_jobs DROP CONSTRAINT ingestion_jobs_source_check;

ALTER TABLE ingestion_jobs ADD CONSTRAINT ingestion_jobs_source_check CHECK (source IN ('confluence', 'markdown', 'multi_source'));

ALTER TABLE ingestion_jobs ADD COLUMN parameters JSONB;

ALTER TABLE ingestion_jobs ADD COLUMN chunk_reference_count INTEGER DEFAULT '0' NOT NULL;

ALTER TABLE ingestion_jobs ADD COLUMN reused_chunk_count INTEGER DEFAULT '0' NOT NULL;

CREATE TABLE multi_source_index_state (
    id SERIAL NOT NULL, 
    owner_id UUID NOT NULL, 
    collection VARCHAR(100) NOT NULL, 
    model VARCHAR(200) NOT NULL, 
    manifest JSONB NOT NULL, 
    published_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    CHECK (id = 1), 
    FOREIGN KEY(owner_id) REFERENCES app_users (id)
);

REVOKE ALL ON multi_source_index_state FROM PUBLIC; GRANT SELECT, INSERT, UPDATE, DELETE ON multi_source_index_state TO ragu_app;;

COMMIT;

