# API testing and an embedding demonstration

## What is possible now

Local documents -> real local embeddings -> Qdrant -> authenticated semantic
retrieval -> live source verification is implemented. Hybrid BM25/dense retrieval
and local cross-encoder reranking are optional. The API returns matching passages,
citations, candidate IDs, generation IDs and scores, not raw embedding arrays.
It does not generate an LLM answer; that belongs to the calling agent.

Confluence Data Center is a separate, live-credential-dependent flow.
Atlassian Cloud is still a TODO-marked, unwired stub and cannot run end to end.
Redis is needed only for multi-source imports, not this local-folder demo.

## Prerequisites

Use the existing installation's protected configuration; do not reset its
database, users or passwords. PostgreSQL must have the current migrations.
If an upgrade is needed, stop API/workers and follow the root README's migration
instructions before starting them again.

Start the approved Qdrant server (see the root README); API and worker must use
the **same Qdrant server and PostgreSQL database**. Embedded Qdrant is for tests
only. This demo does not provision infrastructure, migrate a database or change
configuration.

The dense model must already be provisioned under `RAGU_EMBEDDING_CACHE`
(`.ragu\models` by default). For hybrid/reranking also provision the sparse model
and reranker. Offline runtime never downloads missing models. If artifacts are
missing, use the README's approved model-provisioning procedure.

In two separate PowerShell terminals at the repository root:

```powershell
uv run --locked ragu serve
```

```powershell
uv run --locked ragu worker
```

Use an existing RAGU administrator. Liveness is public at
`http://127.0.0.1:8000/actuator/health/liveness`; authenticated readiness checks
PostgreSQL and Qdrant but **does not prove a worker is alive or a model loads**.
A successful demo job provides that stronger check.

## Executable demo

Pick an existing operator-approved directory accessible to both the script and
the server/worker. It must fall under `RAGU_MULTI_SOURCE_ROOTS`, or under
`RAGU_LOCAL_CONSUMPTION_ROOT` when the former is unset. Do not pass an unrelated
directory or add broader allowlists just to make the demo pass.
Prefer a dedicated approved demo root outside all existing corpus folders:
any other corpus recursively scanning an ancestor can also discover these files
on its next ingestion, even though this script targets only the new corpus.

```powershell
$approvedRoot = Read-Host "Existing absolute operator-approved document directory"
uv run --locked python documentation\demo_e2e.py --approved-root "$approvedRoot" --username admin
```

The password is prompted privately, never supplied on the command line. The
script creates two synthetic files in a unique `ragu-demo-*` subfolder, creates a
new project and corpus, queues dense embeddings, polls the separate worker for
up to ten minutes, retrieves the network policy and verifies the live document.
It prints the job, project/corpus and candidate IDs, index/model profile, vector
similarity score and passage containing `BLUE-482`. An unrelated cafeteria
document demonstrates ranking rather than simply retrieving the only document.

The query deliberately does not contain `BLUE-482`:
`How should a client recover from temporary network failures?`

For dense + BM25 + hybrid comparison and a separate reranking request:

```powershell
uv run --locked python documentation\demo_e2e.py --approved-root "$approvedRoot" --username admin --index-mode hybrid --rerank
```

The default API origin is `http://127.0.0.1:8000`; `--base-url` also supports HTTPS
origins. For a remote API, the demo folder must still be a shared path that the
server can access under the same approved name; this is not a file-upload API.
The script targets no existing corpus and edits no existing documents,
schedules, credentials or `.env`.
Each run leaves a new project/corpus/job and its synthetic folder for inspection;
there is currently no project/corpus deletion endpoint. Keep these files while
retrieving: removing or changing them intentionally invalidates old evidence.

## Postman Desktop

Import `documentation\postman\ragu.postman_collection.json`. Use a local-only
Postman workspace with cloud synchronization disabled for private API data, or
an organization-approved equivalent. Never export/share responses, credentials
or production document content.

Prepare just the synthetic files (no login or server call):

```powershell
uv run --locked python documentation\demo_e2e.py --approved-root "$approvedRoot" --prepare-only
```

1. Set collection variables `base_url`, `username` and `demo_folder` (the absolute
   path just printed, pasted normally, without doubling Windows backslashes).
   Leave `index_mode=dense` and `candidate_method=dense` for the minimum setup.
2. Create the local Postman Vault secret `ragu-password` with the existing admin
   password. Allow this collection's scripts to access Vault. Login stores the
   short-lived JWT in local Vault as `ragu-access-token`; exported collection
   variables contain no password or JWT. Do not log request headers/bodies.
3. Run only the **End-to-end** folder, in order, using Collection Runner with a
   **2000 ms request delay** and a request timeout of at least **120000 ms**.
   The job-status request repeats until success, failure or its ten-minute bound.
   Manual **Send** does not follow `setNextRequest`; repeat the status request
   manually until `succeeded` before proceeding.
4. Inspect the index profile, candidate passages, dense scores and live source.
   The **Inspect demo** folder has read-only project/corpus/job/schedule requests.
   The optional reranking folder requires its additional local model.

To compare hybrid candidates, set `index_mode=hybrid` and
`candidate_method=hybrid_rrf`, then run **End-to-end** again (it creates another
isolated project). Semantic search still explicitly uses `dense` to demonstrate
actual vector similarity. The collection covers this demo and inspection, not
destructive credential management, all permission mutations, OAuth consent or
MCP. It requires Postman Desktop Vault support, not Newman or a cloud runner.

## Interpreting failures

Connection refused: start the API. Readiness DOWN: check PostgreSQL and Qdrant.
An ingestion remaining `pending`: start the separate worker with matching config.
HTTP 401: log in again; API JWTs expire (15 minutes by default). HTTP 403 at corpus
creation: check admin identity and operator-approved paths. HTTP 409 when queueing:
another ingestion is pending/running; inspect it rather than replacing/cancelling
it. A failed job with a model error: provision the required local model. Failed
requests and timeouts are errors, never a successful demonstration.
