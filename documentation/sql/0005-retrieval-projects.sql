-- Generated from canonical Alembic migrations; see documentation/README.md.
BEGIN;

INSERT INTO app_roles (code, name) VALUES ('reader', 'Reader') ON CONFLICT (code) DO NOTHING;

CREATE TABLE projects (
    id UUID NOT NULL, 
    name VARCHAR(100) NOT NULL, 
    owner_id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(owner_id) REFERENCES app_users (id)
);

CREATE TABLE project_members (
    project_id UUID NOT NULL, 
    user_id UUID NOT NULL, 
    role VARCHAR(16) NOT NULL, 
    PRIMARY KEY (project_id, user_id), 
    CHECK (role IN ('reader', 'maintainer')), 
    FOREIGN KEY(project_id) REFERENCES projects (id), 
    FOREIGN KEY(user_id) REFERENCES app_users (id)
);

CREATE TABLE corpora (
    id UUID NOT NULL, 
    project_id UUID NOT NULL, 
    name VARCHAR(100) NOT NULL, 
    source VARCHAR(32) NOT NULL, 
    owner_id UUID NOT NULL, 
    configuration JSON NOT NULL, 
    source_root VARCHAR(64), 
    collection VARCHAR(100), 
    model VARCHAR(200), 
    index_mode VARCHAR(16) DEFAULT 'dense' NOT NULL, 
    index_profile JSON, 
    fingerprint VARCHAR(64), 
    manifest JSON NOT NULL, 
    published_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CHECK (source IN ('confluence', 'local_consumption', 'multi_source')), 
    CHECK (index_mode IN ('dense', 'sparse', 'hybrid')), 
    FOREIGN KEY(project_id) REFERENCES projects (id), 
    FOREIGN KEY(owner_id) REFERENCES app_users (id)
);

CREATE INDEX ix_corpora_project_id ON corpora (project_id);

CREATE TABLE corpus_grants (
    corpus_id UUID NOT NULL, 
    user_id UUID NOT NULL, 
    permission VARCHAR(16) NOT NULL, 
    PRIMARY KEY (corpus_id, user_id), 
    CHECK (permission IN ('read', 'ingest')), 
    FOREIGN KEY(corpus_id) REFERENCES corpora (id), 
    FOREIGN KEY(user_id) REFERENCES app_users (id)
);

CREATE TABLE corpus_schedules (
    corpus_id UUID NOT NULL, 
    owner_id UUID NOT NULL, 
    source VARCHAR(16) NOT NULL, 
    source_root VARCHAR(64), 
    enabled BOOLEAN NOT NULL, 
    hour_utc INTEGER NOT NULL, 
    minute_utc INTEGER NOT NULL, 
    next_run_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    index_mode VARCHAR(16), 
    PRIMARY KEY (corpus_id), 
    CHECK (hour_utc BETWEEN 0 AND 23 AND minute_utc BETWEEN 0 AND 59), 
    CHECK (source IN ('confluence', 'markdown')), 
    FOREIGN KEY(corpus_id) REFERENCES corpora (id), 
    FOREIGN KEY(owner_id) REFERENCES app_users (id)
);

ALTER TABLE ingestion_jobs ADD COLUMN corpus_id UUID;

ALTER TABLE ingestion_jobs ADD FOREIGN KEY(corpus_id) REFERENCES corpora (id);

ALTER TABLE ingestion_jobs ADD COLUMN index_mode VARCHAR(16) DEFAULT 'dense' NOT NULL;

ALTER TABLE ingestion_jobs ADD COLUMN index_profile JSON;

ALTER TABLE ingestion_jobs ADD CONSTRAINT ingestion_jobs_index_mode_check CHECK (index_mode IN ('dense', 'sparse', 'hybrid'));

ALTER TABLE index_state ADD COLUMN index_mode VARCHAR(16) DEFAULT 'dense' NOT NULL;

ALTER TABLE index_state ADD COLUMN index_profile JSON;

ALTER TABLE index_state ADD CONSTRAINT index_state_index_mode_check CHECK (index_mode IN ('dense', 'sparse', 'hybrid'));

ALTER TABLE markdown_index_state ADD COLUMN index_mode VARCHAR(16) DEFAULT 'dense' NOT NULL;

ALTER TABLE markdown_index_state ADD COLUMN index_profile JSON;

ALTER TABLE markdown_index_state ADD CONSTRAINT markdown_index_state_index_mode_check CHECK (index_mode IN ('dense', 'sparse', 'hybrid'));

ALTER TABLE multi_source_index_state ADD COLUMN index_mode VARCHAR(16) DEFAULT 'dense' NOT NULL;

ALTER TABLE multi_source_index_state ADD COLUMN index_profile JSON;

ALTER TABLE multi_source_index_state ADD CONSTRAINT multi_source_index_state_index_mode_check CHECK (index_mode IN ('dense', 'sparse', 'hybrid'));

ALTER TABLE daily_schedule ADD COLUMN index_mode VARCHAR(16);

REVOKE ALL ON projects, project_members, corpora, corpus_grants, corpus_schedules FROM PUBLIC; GRANT SELECT, INSERT, UPDATE, DELETE ON projects, project_members, corpora, corpus_grants, corpus_schedules TO ragu_app;;

COMMIT;
