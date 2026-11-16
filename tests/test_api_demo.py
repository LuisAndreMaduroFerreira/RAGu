import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from documentation.demo_e2e import DemoError, api_origin, prepare_documents, run_demo
from ragu.database import Corpus, Job, LocalConsumptionIndexState

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "documentation" / "postman" / "ragu.postman_collection.json"


class PolicyEmbeddings:
    def embed(self, texts):
        for text in texts:
            yield np.array([1.0, 0.0, 0.0, 0.0] if "network" in text else [0.0, 1.0, 0.0, 0.0])

    def query_embed(self, query):
        return self.embed([query])


def test_executable_demo_retrieves_evidence_without_replacing_the_default_corpus(
    env, tmp_path, monkeypatch, capsys
):
    runtime, client, password, *_ = env
    runtime.settings.multi_source_roots = [tmp_path]
    runtime.vectors._embedder = PolicyEmbeddings()
    monkeypatch.setattr("documentation.demo_e2e.time.sleep", lambda _: runtime.ingestion.tick())
    result = run_demo(client, tmp_path, "admin", password)
    with runtime.db.session() as session:
        assert session.get(LocalConsumptionIndexState, 1) is None
        assert len(list(session.scalars(select(Corpus)))) == 1
        assert len(list(session.scalars(select(Job)))) == 1
    output = capsys.readouterr().out
    assert "PASS:" in output and "BLUE-482" in output and "dense: score=" in output
    assert result["generation_id"] in output
    assert password not in output


def test_demo_timeout_is_an_error_and_leaves_the_job_inspectable(env, tmp_path):
    runtime, client, password, *_ = env
    runtime.settings.multi_source_roots = [tmp_path]
    with pytest.raises(DemoError, match="timed out"):
        run_demo(client, tmp_path, "admin", password, timeout=0)
    with runtime.db.session() as session:
        assert session.scalar(select(Job)).status == "pending"


def test_demo_rejects_bad_credentials_before_creating_files(env, tmp_path):
    _, client, *_ = env
    with pytest.raises(DemoError, match="HTTP 401"):
        run_demo(client, tmp_path, "admin", "synthetic-wrong-password")
    assert not list(tmp_path.iterdir())


def test_prepare_only_preserves_existing_files_and_uses_unique_folders(tmp_path):
    existing = tmp_path / "existing.txt"
    existing.write_text("Preserve existing documents.", encoding="utf-8")
    first, second = prepare_documents(tmp_path), prepare_documents(tmp_path)
    assert first != second
    assert {path.name for path in first.iterdir()} == {"retries.txt", "cafeteria.txt"}
    assert existing.read_text(encoding="utf-8") == "Preserve existing documents."
    with pytest.raises(DemoError, match="must already exist"):
        prepare_documents(tmp_path / "missing")


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example.invalid",
        "https://user:secret@example.invalid",
        "https://example.invalid/path",
        "https://example.invalid?token=hidden",
    ],
)
def test_demo_never_sends_credentials_to_an_insecure_or_ambiguous_origin(url):
    with pytest.raises(argparse.ArgumentTypeError):
        api_origin(url)


def sandbox(collection, item, state, stage, response=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise Postman sandbox scripts.")
    completed = subprocess.run(
        [node, str(ROOT / "tests" / "postman_sandbox.cjs")],
        input=json.dumps(
            {
                "collection": collection,
                "item": item,
                "state": state,
                "stage": stage,
                "response": response,
            }
        ),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    return json.loads(completed.stdout)


def test_exact_postman_requests_and_scripts_complete_the_isolated_demo(env, tmp_path):
    runtime, client, password, *_ = env
    runtime.settings.multi_source_roots = [tmp_path]
    runtime.vectors._embedder = PolicyEmbeddings()
    collection = json.loads(COLLECTION.read_text(encoding="utf-8"))
    state = {
        "collection": {item["key"]: item["value"] for item in collection["variable"]},
        "vault": {"ragu-password": password},
        "local": {},
    }
    state["collection"]["demo_folder"] = str(prepare_documents(tmp_path))
    items = collection["item"][0]["item"]
    position = polls = 0
    while position < len(items):
        item = items[position]
        before = sandbox(collection, item, state, "prerequest")
        assert before["error"] is None, before["error"]
        response = client.request(**before["request"])
        after = sandbox(
            collection,
            item,
            before["state"],
            "test",
            {"code": response.status_code, "body": response.json()},
        )
        assert after["error"] is None, after["error"]
        state = after["state"]
        if after["nextSet"]:
            assert after["nextRequest"] == item["name"]
            polls += 1
            assert polls == 1
            assert runtime.ingestion.tick()
        else:
            position += 1
    assert "BLUE-482" in response.json()["text"]
    assert state["vault"]["ragu-access-token"]
    assert state["collection"]["candidate_ids"] != "[]"
    assert password not in json.dumps(state["collection"])
    assert state["vault"]["ragu-access-token"] not in json.dumps(state["collection"])
    with runtime.db.session() as session:
        assert session.get(LocalConsumptionIndexState, 1) is None


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_postman_polling_stops_on_failure_or_timeout(status):
    collection = json.loads(COLLECTION.read_text(encoding="utf-8"))
    item = next(item for item in collection["item"][0]["item"] if "Wait for worker" in item["name"])
    result = sandbox(
        collection,
        item,
        {"collection": {"poll_deadline": "0"}, "local": {}, "vault": {}},
        "test",
        {"code": 200, "body": {"status": status}},
    )
    assert result["error"] is not None
    assert result["nextSet"] and result["nextRequest"] is None
