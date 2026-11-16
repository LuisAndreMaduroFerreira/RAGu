# RAGU infrastructure and multi-source imports

For dense/BM25/hybrid profiles, local reranking, project permissions, REST/MCP
examples and agent sequence diagrams, see
[retrieval and project agents](retrieval-and-agents.md).

## What this mode does

`POST /ingestions/multi-source` queues an asynchronous import. Without a project
selector it remains administrator-only; an explicitly granted project ingest
identity can import only into its selected project corpus.
For project corpora, a global administrator binds the approved source list on
the first accepted import; delegated writers cannot initialize or change it.
Its `sources` array accepts up to 16 absolute folders or explicitly mocked
Confluence page links. Folders recursively accept `.md`, `.html`, `.htm`, and
`.txt`, case-insensitively. Plain text is UTF-8/UTF-8 BOM, normalized for line
endings and otherwise retained literally. The existing newest-filename-version,
exact-document deduplication, and safe-file checks apply in every folder.

Each successful request **replaces the complete multi-source corpus**. It does
not append to it, modify the default local-consumption corpus, or change the
legacy Confluence corpus. Include every source that should remain in this corpus.
The first importing administrator owns it. Other administrators and readers
cannot retrieve or replace its documents. Passing `corpus_id` instead selects
that independent, explicitly granted project corpus; it never exposes or
replaces the legacy corpus.

The worker validates every source before creating a Qdrant generation. Chunk
IDs derive from SHA-256 of exact embedding input, document task, and model name;
they never depend on approximate vector similarity. Identical chunks across
folders and mocked pages are embedded once and have one Qdrant point **within
that generation**. Separate document references in PostgreSQL retain citations,
source locations, and deletion behavior. Unique document content is not merged
just because vectors are similar.

Each job has a new Redis namespace, `ragu:import:<job-uuid-hex>:*`. Values are only
write-completion markers: no text, model output vectors, filenames, or URLs.
Redis eviction/expiry triggers an authoritative Qdrant point lookup; it cannot
create duplicate points or require re-embedding a point already written by this
job. Redis is not shared across jobs for embedding reuse and is not on the
retrieval path. Model inference uses the same loaded provider during the job;
there is no claim of cross-runtime or cross-model-artifact embedding equivalence.

Markers have a TTL (default one hour, configurable from 60 seconds to one day)
and are removed on success/failure. A worker crash leaves expiring markers, not
a permanent cache. Cleanup never uses `FLUSHDB` or `FLUSHALL`. Cleanup failure
prevents publication; TTL remains the crash/failure safety net. Redis or Qdrant
errors fail the job explicitly and preserve the last published generation.
Existing PostgreSQL worker leases and final authorization checks remain active.

Retrieval uses `source: "multi_source"`. It rechecks current administrator
authorization (or the selected project's current grants) and each source's live contents. Deleted, denied mocked pages, and
changed documents do not provide citations or passages until reindexed.
Filesystem safety errors remain explicit rather than becoming empty results.
Only citations whose current source is accessible are returned. The existing
MCP cloud-context approval gate is unchanged.

This is bounded ingestion, **not a hundreds-of-GB streaming importer**. Existing
defaults are 5 MB/file, 100 MB combined validated input, 10,000 files/documents,
and 100,000 chunk references, including repeated chunks. Per-folder scans also
enforce their original-byte limit, including ignored older versions/duplicates.
The JSON manifest and validated selected text are held in memory. Raising
limits substantially requires a separate streaming/staging design.

## PostgreSQL

Do not reset or recreate the existing database, users, passwords, or roles.

Stop the API and workers before upgrading, then run from the repository root:

```powershell
uv run --locked ragu migrate
```

The migration command prompts for `ragu_migrator`'s password without echoing it.
Migration `0003` adds job parameters/reference counters and
`multi_source_index_state`; it does not add a persistent embedding-cache table.
Migration `0004` adds `retired_indexes` for generation cleanup. Retirement is
recorded in the same transaction as a replacement publication. Replaced
generations retain a full hour from replacement; failed or expired staging
generations receive the same grace period. Unchanged local runs keep their
active generation. Cleanup workers skip rows locked by another cleaner, retain
failed Qdrant deletions for retry, and consume successfully cleaned records.
Historical inactive generations receive a fresh one-hour grace period when
`0004` runs; active generations are excluded from this backfill.
The runtime account receives only the table DML grants it needs.
Migration `0005` adds retrieval profiles, projects/members, corpus grants and
per-corpus schedules, preserving the existing legacy ownership restrictions.
Restart the updated API/workers only after applying pending migrations; adding
these files does not migrate the existing installation.

SQL references:

- `sql/provision-fresh.psql`: database/role setup for a **new installation only**.
  Uses interactive password prompts, not embedded credentials.
- `sql/fresh-schema.sql`: SQL generated from the fresh-install branches of
  migrations `0001` through `0005`, including grants and Alembic version updates.
  Reference only for an existing database; do not run it over existing tables.
- `sql/0003-multi-source.sql`: exact PostgreSQL DDL emitted by migration `0003`.
  Prefer the migration command; manually applying this file does not update
  Alembic's version ledger.
- `sql/0004-index-retirement.sql`: PostgreSQL DDL and conservative backfill
  emitted by migration `0004`, likewise without a version-ledger update.
- `sql/0005-retrieval-projects.sql`: PostgreSQL DDL and runtime table grants from
  migration `0005`, also without a version-ledger update.

Regenerate current SQL without connecting to a database with
`uv run --locked python documentation\sql\generate.py`. Python migrations remain
authoritative; the standalone upgrade SQL deliberately does not advance Alembic.

The original `0001` migration can also adopt the original authentication tables
without changing user IDs/password hashes. Its conditional adoption branch is
not represented by the fresh-install SQL. The Python migrations are authoritative.
The SQL files contain no real passwords or credential hashes.
Real PostgreSQL migration, permission, and concurrency regressions are described
in the root README's Development section. Their disposable cluster is separate
from the application database; SQLite tests alone do not exercise these locks.

## Redis

Use an approved Redis 7.2+ installation dedicated to this disposable import state.
`redis/redis.conf` is a loopback-only, bounded-memory, non-persistent example.
`redis/users.acl.example` denies the default user and grants only the commands
used by the importer under its key prefix.
The Redis/Qdrant configuration examples target Linux services (such as an
approved VM, WSL, or container); their example paths are not Windows shell paths.

Before starting Redis, create a private ACL file from the example, replace the
placeholder with SHA-256 of a strong generated password, and restrict OS access
to the service account. Do not commit the real ACL file or password. Set
`aclfile` to that private file's absolute path. Run Redis as an unprivileged
account and do not expose port 6379 publicly. `maxmemory` bounds Redis-managed
memory, not total process RSS; reserve additional OS/container headroom.

```text
redis-server /absolute/private/path/redis.conf
```

Set `RAGU_IMPORT_REDIS_URL` in the protected, ignored application `.env`:

```dotenv
RAGU_IMPORT_REDIS_URL="redis://ragu_import:REPLACE_PRIVATELY@127.0.0.1:6379/0"
RAGU_IMPORT_CACHE_TTL_SECONDS=3600
RAGU_MULTI_SOURCE_ROOTS='["C:\\RAGU\\Approved"]'
```

Do not paste a real Redis URL into chat, command-line arguments, logs, or this
repository. Remote Redis requires `rediss://`, a valid trusted server
certificate, and private networking. The application allows only database 0 and
does not allow URL query options that could disable TLS validation.

When `RAGU_MULTI_SOURCE_ROOTS` is empty, the configured
`RAGU_LOCAL_CONSUMPTION_ROOT` is the sole allowed root. A request may select only
those roots or their regular subdirectories. Registering roots is an operator
configuration action, not an API permission to read arbitrary server paths.
Filesystem links and opened-file escapes remain forbidden. Configuring Redis
does not install/start its server. The legacy ingestion paths do not require it.

## Qdrant

`qdrant/config.yaml` shows loopback bindings and a private persistent storage
directory. Set `QDRANT__SERVICE__API_KEY` through the service's secret mechanism,
and match it with private `RAGU_QDRANT_API_KEY` configuration. Protect the storage
directory/backups; document payloads are company information.

The worker creates collections named `ragu_multi_source_<job-uuid-hex>` or
`ragu_corpus_<corpus-uuid-hex>_<job-uuid-hex>`. Dense indexes use Cosine distance;
hybrid indexes have named `dense` and `bm25` vectors; sparse indexes have only
`bm25` with IDF weighting. Dense dimensions come from the provider (384 for the
default model); no collection needs to be precreated.
Qdrant contains the actual vectors and passage text; Redis does not.
Cross-job generations are not globally deduplicated.

Run the API and worker using their existing supervised commands after applying
migrations. A reachable Qdrant and Redis are required for an actual batch.
The provided files do not automatically provision infrastructure or change
the protected `.env`.

## Requests and results

An admin API JWT, or an API JWT with ingest permission on the selected project
corpus, is required. Monitoring credentials and MCP-audience tokens cannot
enqueue imports. Example local-only body:

```json
{
  "sources": [
    {"kind": "folder", "path": "C:\\RAGU\\Approved\\Engineering"},
    {"kind": "folder", "path": "C:\\RAGU\\Approved\\Operations"}
  ]
}
```

POST it to `/ingestions/multi-source`, then poll `/ingestions/{id}`.
Jobs report `chunk_count` (unique Qdrant points), `chunk_reference_count`
(occurrences across source documents), and `reused_chunk_count` (occurrences
minus unique points). `file_count` counts validated inputs; `duplicate_count`
still counts ignored full-document duplicates and superseded filename versions
within folder scans, not repeated chunks. These new reference counters are
zero for the older ingestion modes.

```json
{"query": "deployment policy", "source": "multi_source", "limit": 5}
```

POST searches to `/documents/search`. Each unique-passage result includes a
`citations` array of currently accessible source documents. Read a document with
`GET /documents/{page_id}?source=multi_source`. Absolute filesystem locations
are kept in the private manifest, not returned in citations or job responses.

## Confluence: mock only

**No real Atlassian Cloud requests or credentials are implemented for this mode.**
Production `Runtime` has no Confluence page provider, so requests with Confluence
sources fail explicitly before being queued. Existing legacy Data Center
ingestion is unaffected; it is not repurposed as a Cloud connector.

Tests/demos can explicitly inject `MockConfluenceProvider` with synthetic `Page`
objects through `Runtime(..., mock_confluence=provider)`. Its links must match
`RAGU_MOCK_CONFLUENCE_ORIGIN` (default `https://confluence.example.invalid`) and
the `/wiki/spaces/<space>/pages/<numeric-id>[/title]` format. This configuration
is a mock origin restriction, not a switch enabling live network access.

Example source for an injected mock only:

```json
{"kind": "confluence", "url": "https://confluence.example.invalid/wiki/spaces/DEMO/pages/123"}
```

A future approved real connector needs its own credential handling,
origin/project restrictions, bounded pagination and current-source permission
checks. Enabling it is a separate change, not an implicit effect of this example.
