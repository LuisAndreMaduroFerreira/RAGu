import argparse
import getpass
import json
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

QUERY = "How should a client recover from temporary network failures?"
EXPECTED = "BLUE-482"


class DemoError(Exception):
    pass


def api_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
    ):
        raise argparse.ArgumentTypeError("Use an HTTPS origin, or HTTP on loopback only.")
    return value.rstrip("/")


def prepare_documents(approved_root: Path) -> Path:
    if not approved_root.is_absolute() or ".." in approved_root.parts:
        raise DemoError("--approved-root must be an absolute path without parent traversal.")
    if not approved_root.is_dir():
        raise DemoError("--approved-root must already exist and be operator-approved.")
    folder = approved_root / f"ragu-demo-{uuid4().hex}"
    folder.mkdir()
    documents = {
        "retries.txt": (
            "Network recovery policy. Retry transient network failures with exponential "
            "backoff and jitter. Stop after three attempts and report the error. "
            "Tracking code BLUE-482."
        ),
        "cafeteria.txt": (
            "The cafeteria serves lunch between noon and two. "
            "The menu changes every week. Vegetarian meals are available."
        ),
    }
    for name, text in documents.items():
        with (folder / name).open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(f"Demo folder: {folder}", flush=True)
    return folder


def request(client, method, path, *, expected=200, **kwargs):
    response = client.request(method, path, **kwargs)
    if response.status_code != expected:
        raise DemoError(
            f"{method} {path}: HTTP {response.status_code}. Check server logs and prerequisites; "
            "401 requires a fresh login, 403 an admin/approved folder, "
            "409 an available queue/index, and 503 healthy dependencies/models."
        )
    return response.json()


def wait_for_job(client, job_id, timeout):
    deadline = time.monotonic() + timeout
    while True:
        job = request(client, "GET", f"/ingestions/{job_id}")
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise DemoError(
                f"Ingestion {job_id} failed. Inspect its status and private worker logs."
            )
        if job["status"] not in {"pending", "running"}:
            raise DemoError(f"Ingestion {job_id} returned an unexpected status.")
        if time.monotonic() >= deadline:
            raise DemoError(
                f"Ingestion {job_id} timed out. Is 'ragu worker' running against the same "
                "PostgreSQL/Qdrant configuration? The queued job was not cancelled."
            )
        time.sleep(2)


def verify_result(batch, generation_id):
    if batch["generation_id"] != generation_id:
        raise DemoError("The retrieved generation does not match the successful ingestion.")
    if not batch["results"] or EXPECTED not in batch["results"][0]["passage"]:
        raise DemoError("Semantic retrieval did not rank the synthetic network policy first.")
    top = batch["results"][0]
    if top["relative_path"] != "retries.txt":
        raise DemoError("Retrieval returned an unexpected document.")
    return top


def run_demo(client, approved_root, username, password, *, mode="dense", rerank=False, timeout=600):
    identity = request(
        client, "POST", "/auth/login", json={"username": username, "password": password}
    )
    client.headers["Authorization"] = f"Bearer {identity['access_token']}"
    user = request(client, "GET", "/auth/me")
    if user["role"] != "admin":
        raise DemoError("Use an existing administrator to create the isolated demo corpus.")
    health = request(client, "GET", "/actuator/health/readiness")
    if health["status"] != "UP":
        raise DemoError("PostgreSQL and Qdrant must both be ready before starting the demo.")
    folder = prepare_documents(approved_root)
    project = request(client, "POST", "/projects", expected=201, json={"name": folder.name})
    print(f"Project ID: {project['id']}", flush=True)
    corpus = request(
        client,
        "POST",
        f"/projects/{project['id']}/corpora",
        expected=201,
        json={"name": "Embedding demo", "source": "local_consumption", "folder": str(folder)},
    )
    corpus_id = corpus["id"]
    print(f"Corpus ID: {corpus_id}", flush=True)
    queued = request(
        client,
        "POST",
        "/ingestions",
        expected=202,
        json={"corpus_id": corpus_id, "index_mode": mode},
    )
    print(f"Ingestion ID: {queued['id']} (waiting for the separate worker)", flush=True)
    job = wait_for_job(client, queued["id"], timeout)
    if job["page_count"] != 2 or job["chunk_count"] < 2:
        raise DemoError("The ingestion did not publish both synthetic documents.")
    capabilities = request(
        client, "GET", "/documents/capabilities", params={"corpus_id": corpus_id}
    )
    print("Index profile: " + json.dumps(capabilities["index_profile"], sort_keys=True))
    for method in capabilities["candidate_methods"]:
        batch = request(
            client,
            "POST",
            "/documents/search",
            json={"corpus_id": corpus_id, "query": QUERY, "candidate_method": method, "limit": 1},
        )
        top = verify_result(batch, job["id"])
        print(f"{method}: score={top['score']}, candidate_id={top['candidate_id']}")
        print(f"Passage: {top['passage']}")
    candidates = request(
        client,
        "POST",
        "/documents/candidates",
        json={
            "corpus_id": corpus_id,
            "query": QUERY,
            "candidate_method": "dense",
            "limit": 2,
        },
    )
    verify_result(candidates, job["id"])
    if rerank:
        ranked = request(
            client,
            "POST",
            "/documents/rerank",
            json={
                "corpus_id": corpus_id,
                "query": QUERY,
                "generation_id": candidates["generation_id"],
                "candidate_ids": [item["candidate_id"] for item in candidates["results"]],
                "rerank_method": "cross_encoder",
                "limit": 1,
            },
        )
        top = verify_result(ranked, job["id"])
        print(f"Cross-encoder rerank score: {top['rerank_score']}")
    document = request(
        client,
        "GET",
        f"/documents/{candidates['results'][0]['page_id']}",
        params={"corpus_id": corpus_id},
    )
    if EXPECTED not in document["text"]:
        raise DemoError("Live document verification did not match the retrieved passage.")
    print(f"PASS: generation {job['id']} retrieved the policy and verified its live source.")
    print("Keep the demo folder while demonstrating retrieval; source files are revalidated.")
    print("Only the new corpus was targeted; no existing documents or configuration were edited.")
    return {"project_id": project["id"], "corpus_id": corpus_id, "generation_id": job["id"]}


def main():
    parser = argparse.ArgumentParser(description="Demonstrate real API ingestion and retrieval.")
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--base-url", type=api_origin, default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--index-mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--rerank", action="store_true", help="Also use the local cross-encoder.")
    parser.add_argument(
        "--timeout-seconds", type=int, choices=range(1, 1801), default=600, metavar="1..1800"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Only create Postman demo files."
    )
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare_documents(args.approved_root)
            return 0
        warnings.simplefilter("error", getpass.GetPassWarning)
        password = getpass.getpass("Existing RAGU administrator password (hidden): ")
        with httpx.Client(
            base_url=args.base_url, timeout=120, follow_redirects=False, trust_env=False
        ) as client:
            run_demo(
                client,
                args.approved_root,
                args.username,
                password,
                mode=args.index_mode,
                rerank=args.rerank,
                timeout=args.timeout_seconds,
            )
        return 0
    except (DemoError, OSError, getpass.GetPassWarning, httpx.HTTPError) as error:
        message = (
            str(error)
            if isinstance(error, DemoError)
            else (
                f"{type(error).__name__}: check local files, a private interactive terminal, "
                "and the API connection. No credentials were printed."
            )
        )
        print(f"Demo failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
