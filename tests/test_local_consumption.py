import json
import os
from datetime import timedelta
from urllib.parse import parse_qs, urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from sqlalchemy import select

from ragu.database import (
    Job,
    LocalConsumptionIndexState,
    RetiredIndex,
    Role,
    Schedule,
    User,
    utcnow,
)
from ragu.errors import AppError
from tests.conftest import headers
from tests.test_oauth_mcp import authorize, token_request


@pytest.fixture
def local(env, tmp_path):
    runtime, client, password, user_id = env
    root = tmp_path / "documents"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\n\nLocal retry policy.\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "beta.MD").write_text(
        "# Beta\n\nLocal deployment policy.\n", encoding="utf-8"
    )
    (root / "ignored.pdf").write_text("An unsupported document type.", encoding="utf-8")
    runtime.settings.local_consumption_root = root
    runtime.settings.confluence_ingestion_token = None
    with runtime.db.session() as session:
        session.get(User, user_id).confluence_token = None
    return runtime, client, headers(client, password), root, user_id, password


def ingest_local(local, source="local_consumption"):
    runtime, client, auth, *_ = local
    response = client.post("/ingestions", headers=auth, json={"source": source})
    assert response.status_code == 202, response.text
    assert response.json()["source"] == "local_consumption"
    assert runtime.ingestion.tick()
    job = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "succeeded", job
    return job


def state(local):
    with local[0].db.session() as session:
        return session.get(LocalConsumptionIndexState, 1)


def version_file(root, name, text, sequence):
    path = root / name
    path.write_text(text, encoding="utf-8")
    timestamp = 1_700_000_000_000_000_000 + sequence * 1_000_000_000
    os.utime(path, ns=(timestamp, timestamp))
    return path


def test_only_newest_filename_version_is_indexed_across_supported_formats(local):
    runtime, client, auth, root, *_ = local
    version_file(root, "guide_v99.md", "# Guide\n\nOldest policy.", 1)
    version_file(root, "guide.md", "# Guide\n\nPrevious policy.", 2)
    version_file(root, "guide - Copy (2).HTML", "<h1>Guide</h1><p>Latest policy.</p>", 3)
    version_file(root, "guide_v3.TXT", "# Guide\n\nNewest plain-text policy.", 4)
    job = ingest_local(local)
    assert job["file_count"] == 6
    assert job["page_count"] == 3
    assert job["duplicate_count"] == 3
    published = state(local)
    identifier, entry = next(
        (identifier, entry)
        for identifier, entry in published.manifest.items()
        if entry["path"].startswith("guide")
    )
    assert entry["path"] == "guide_v3.TXT"
    assert entry["paths"] == ["guide_v3.TXT"]
    assert set(entry["candidates"]) == {
        "guide_v99.md",
        "guide.md",
        "guide - Copy (2).HTML",
        "guide_v3.TXT",
    }
    assert (root / "guide_v99.md").exists()
    assert (root / "guide.md").exists()
    response = client.get(f"/documents/{identifier}", headers=auth)
    assert response.status_code == 200
    assert "Newest plain-text policy." in response.json()["text"]
    assert "Oldest policy." not in response.json()["text"]
    assert runtime.local_consumption.document("guide_v99.md").relative_path == entry["path"]


def test_newer_sibling_hides_old_passages_before_reindex_and_only_embeds_new_content(local):
    runtime, client, auth, root, *_ = local
    version_file(root, "guide.md", "# Guide\n\nOriginal policy.", 1)
    ingest_local(local)
    old = state(local)
    old_id = next(key for key, entry in old.manifest.items() if entry["path"] == "guide.md")
    embedded = runtime.vectors.embedder.embedded
    version_file(root, "guide_v2.md", "# Guide\n\nReplacement policy.", 2)
    response = client.get(f"/documents/{old_id}", headers=auth)
    assert response.status_code == 409
    response = client.post("/documents/search", headers=auth, json={"query": "policy"})
    assert response.status_code == 200
    assert old_id not in {item["page_id"] for item in response.json()["results"]}
    job = ingest_local(local)
    assert job["page_count"] == 3
    assert job["duplicate_count"] == 1
    assert runtime.vectors.embedder.embedded == embedded + 1
    current = state(local)
    assert old_id not in current.manifest
    new_id = next(key for key, entry in current.manifest.items() if entry["path"] == "guide_v2.md")
    assert "Replacement policy." in client.get(f"/documents/{new_id}", headers=auth).json()["text"]
    (root / "guide_v2.md").unlink()
    assert client.get(f"/documents/{new_id}", headers=auth).status_code == 404
    assert ingest_local(local)["page_count"] == 3
    assert old_id in state(local).manifest
    assert new_id not in state(local).manifest


def test_adding_or_changing_an_older_version_is_an_index_noop(local, monkeypatch):
    runtime, _, _, root, *_ = local
    version_file(root, "guide_v2.md", "# Guide\n\nCurrent policy.", 2)
    ingest_local(local)
    previous = state(local)

    def forbidden(*args, **kwargs):
        raise AssertionError("Changes to a superseded version must not touch vectors")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    monkeypatch.setattr(runtime.vectors, "index_page", forbidden)
    version_file(root, "guide - Copy.md", "# Guide\n\nObsolete policy.", 1)
    job = ingest_local(local)
    assert job["unchanged"] is True
    assert job["file_count"] == 4
    assert job["duplicate_count"] == 1
    version_file(root, "guide - Copy.md", "# Guide\n\nDifferent obsolete policy.", 1)
    assert ingest_local(local)["unchanged"] is True
    assert state(local).collection == previous.collection
    assert state(local).published_at == previous.published_at


@pytest.mark.parametrize("legacy_manifest", [False, True])
def test_new_version_filename_reuses_unchanged_chunks_and_accepts_legacy_manifests(
    local, legacy_manifest
):
    runtime, _, _, root, *_ = local
    version_file(root, "long_v1.md", "\n\n".join(letter * 750 for letter in "ABC"), 1)
    ingest_local(local)
    embedded = runtime.vectors.embedder.embedded
    if legacy_manifest:
        with runtime.db.session() as session:
            published = session.get(LocalConsumptionIndexState, 1)
            published.manifest = {
                identifier: {key: value for key, value in entry.items() if key != "candidates"}
                for identifier, entry in published.manifest.items()
            }
    version_file(root, "long_v2.md", "\n\n".join(letter * 750 for letter in "ABD"), 2)
    result = ingest_local(local)
    assert result["unchanged"] is False
    assert result["page_count"] == 3
    assert result["file_count"] == 4
    assert result["duplicate_count"] == 1
    assert runtime.vectors.embedder.embedded == embedded + 1


def test_version_lookup_ignores_regular_directories_named_like_document_versions(local):
    _, client, auth, root, *_ = local
    directory = root / "alpha_v2.md"
    directory.mkdir()
    (directory / "child.md").write_text("# Child\n\nIndependent nested document.", encoding="utf-8")
    assert ingest_local(local)["page_count"] == 3
    identifier = next(
        key for key, entry in state(local).manifest.items() if entry["path"] == "alpha.md"
    )
    assert client.get(f"/documents/{identifier}", headers=auth).status_code == 200


def test_live_version_lookup_enforces_total_bytes_even_for_one_candidate(local):
    runtime, *_ = local
    runtime.settings.local_consumption_max_total_bytes = 1
    with pytest.raises(AppError) as error:
        runtime.local_consumption.document("alpha.md")
    assert error.value.status == 413


def test_identical_newer_copy_changes_provenance_without_reembedding(local, monkeypatch):
    runtime, client, auth, root, *_ = local
    content = "# Guide\n\nUnchanged policy."
    version_file(root, "guide.md", content, 1)
    ingest_local(local)
    previous = state(local)
    identifier = next(
        key for key, entry in previous.manifest.items() if entry["path"] == "guide.md"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("An identical newer copy must not touch vectors")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    monkeypatch.setattr(runtime.vectors, "index_page", forbidden)
    version_file(root, "guide (1).md", content, 2)
    response = client.get(f"/documents/{identifier}", headers=auth)
    assert response.status_code == 200
    assert response.json()["relative_path"] == "guide (1).md"
    assert ingest_local(local)["unchanged"] is True
    assert state(local).collection == previous.collection
    assert state(local).manifest[identifier]["paths"] == ["guide (1).md"]


def test_conflicting_latest_timestamps_fail_without_publishing_or_hiding_the_error(
    local, monkeypatch
):
    runtime, client, auth, root, *_ = local
    version_file(root, "guide.md", "# Guide\n\nOriginal policy.", 1)
    ingest_local(local)
    previous = state(local)
    identifier = next(
        key for key, entry in previous.manifest.items() if entry["path"] == "guide.md"
    )
    version_file(root, "guide_v2.md", "# Guide\n\nConflicting policy.", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Ambiguous versions must fail before any vector writes")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    response = client.post("/documents/search", headers=auth, json={"query": "policy"})
    assert response.status_code == 409
    assert "same latest modification time" in response.json()["detail"]
    assert client.get(f"/documents/{identifier}", headers=auth).status_code == 409
    response = client.post("/ingestions", headers=auth)
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    job = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "failed"
    assert "same latest modification time" in job["error"]
    assert state(local).collection == previous.collection


def test_invalid_superseded_version_still_fails_complete_validation(local, monkeypatch):
    runtime, client, auth, root, *_ = local
    version_file(root, "guide_v2.md", "# Guide\n\nCurrent policy.", 2)
    ingest_local(local)
    previous = state(local)
    invalid = root / "guide_v1.md"
    invalid.write_bytes(b"\xff")
    os.utime(invalid, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid older version must fail before any vector writes")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    response = client.post("/ingestions", headers=auth)
    assert runtime.ingestion.tick()
    job = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "failed"
    assert state(local).collection == previous.collection


def test_recursive_markdown_import_search_and_document_without_confluence(local):
    runtime, client, auth, root, *_ = local
    result = ingest_local(local)
    assert result["page_count"] == 2
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
    )
    assert response.status_code == 200, response.text
    items = response.json()["results"]
    assert {item["relative_path"] for item in items} == {"alpha.md", "nested/beta.MD"}
    assert all(str(root) not in item["source_url"] for item in items)
    identifier = items[0]["page_id"]
    document = client.get(f"/documents/{identifier}?source=markdown", headers=auth)
    assert document.status_code == 200
    assert document.json()["text"].startswith("# ")
    assert runtime.source.failure is False
    assert (
        client.post(
            "/documents/search", headers=auth, json={"query": "policy", "source": "confluence"}
        ).status_code
        == 409
    )


def test_html_and_htm_are_read_without_active_or_hidden_content(local):
    _, client, auth, root, *_ = local
    (root / "nested" / "deployment.HTML").write_text(
        """<!doctype html><html><head><title>Tab title</title>
        <script>SCRIPT_ONLY_MARKER</script><style>STYLE_ONLY_MARKER</style></head>
        <body><header><h1>Site header</h1></header><main><h1>HTML deployment</h1>
        <p>Deploy foo &amp; bar <a href="https://invalid.example/doc">documentation</a></p>
        <pre>if ready:\n    deploy()</pre>
        <table><tr><th>Status</th><th>Meaning</th></tr><tr><td>OK</td><td>Ready</td></tr></table>
        <nav>NAV_ONLY_MARKER</nav><p hidden>HIDDEN_ONLY_MARKER</p>
        <div aria-hidden="TRUE">ARIA_ONLY_MARKER</div>
        <template>TEMPLATE_ONLY_MARKER</template><noscript>NOSCRIPT_ONLY_MARKER</noscript>
        <iframe src="https://invalid.example/frame">FRAME_ONLY_MARKER</iframe>
        <img src="https://invalid.example/image"><script>SCRIPT_BODY_MARKER</script>
        </main><footer>Site footer</footer></body></html>""",
        encoding="utf-8",
    )
    (root / "fallback.htm").write_text(
        "<html><head><title>Fallback title</title></head><body><p>Fallback body.</p></body></html>",
        encoding="utf-8",
    )
    result = ingest_local(local)
    assert result["file_count"] == 4
    assert result["page_count"] == 4
    manifest = state(local).manifest
    identifier = next(key for key, entry in manifest.items() if entry["path"].endswith(".HTML"))
    response = client.get(f"/documents/{identifier}?source=markdown", headers=auth)
    assert response.status_code == 200
    document = response.json()
    assert document["title"] == "HTML deployment"
    assert "# HTML deployment" in document["text"]
    assert "Deploy foo & bar" in document["text"]
    assert "if ready:\n    deploy()" in document["text"]
    assert "Status | Meaning" in document["text"]
    assert "ONLY_MARKER" not in document["text"]
    assert "SCRIPT_BODY_MARKER" not in document["text"]
    assert "Site header" not in document["text"]
    assert "Site footer" not in document["text"]
    fallback = next(key for key, entry in manifest.items() if entry["path"] == "fallback.htm")
    response = client.get(f"/documents/{fallback}?source=markdown", headers=auth)
    assert response.json()["title"] == "Fallback title"
    assert response.json()["text"] == "Fallback body."
    response = client.post(
        "/documents/search", headers=auth, json={"query": "deployment", "source": "markdown"}
    )
    assert response.status_code == 200
    assert any(item["page_id"] == identifier for item in response.json()["results"])


def test_plain_text_is_read_literally_with_utf8_bom_and_line_ending_normalization(local):
    _, client, auth, root, *_ = local
    (root / "notes.TXT").write_bytes(
        b"\xef\xbb\xbfPlain text policy.\r\n<b>Literal text, not HTML.</b>\r\n"
    )
    result = ingest_local(local)
    assert result["file_count"] == 3
    identifier = next(
        key for key, entry in state(local).manifest.items() if entry["path"] == "notes.TXT"
    )
    response = client.get(f"/documents/{identifier}", headers=auth)
    assert response.status_code == 200
    assert response.json()["title"] == "notes"
    assert response.json()["text"] == "Plain text policy.\n<b>Literal text, not HTML.</b>\n"


def test_invalid_plain_text_preserves_the_published_generation(local):
    _, client, auth, root, *_ = local
    ingest_local(local)
    old = state(local).collection
    (root / "invalid.txt").write_bytes(b"\xff")
    response = client.post("/ingestions", headers=auth)
    assert local[0].ingestion.tick()
    job = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "failed"
    assert state(local).collection == old


def test_plain_text_and_markdown_exact_copies_share_the_local_document(local):
    _, _, _, root, *_ = local
    (root / "nested" / "alpha.TXT").write_bytes(
        b"\xef\xbb\xbf# Alpha\r\n\r\nLocal retry policy.\r\n"
    )
    job = ingest_local(local)
    assert job["file_count"] == 3
    assert job["page_count"] == 2
    assert job["duplicate_count"] == 1
    entry = next(entry for entry in state(local).manifest.values() if "alpha.md" in entry["paths"])
    assert set(entry["paths"]) == {"alpha.md", "nested/alpha.TXT"}


def test_duplicate_extracted_html_is_ignored_and_deleted_copies_are_mirrored(local, monkeypatch):
    runtime, client, auth, root, *_ = local
    original = root / "example.html"
    original.write_text("<h1>Example</h1><p>A &amp; B.</p>", encoding="utf-8")
    ingest_local(local)
    old = state(local)
    embedded = runtime.vectors.embedder.embedded
    identifier = next(key for key, entry in old.manifest.items() if entry["path"] == "example.html")
    duplicate = root / "nested" / "copy.HTM"
    duplicate.write_text(
        "<html><head><style>different CSS</style></head><body><h1>Example</h1>"
        "<p>A &#38; B.</p><script>ignored()</script></body></html>",
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Equivalent extracted HTML must not change the vector index")

    with monkeypatch.context() as patch:
        patch.setattr(runtime.vectors, "create", forbidden)
        patch.setattr(runtime.vectors, "index_page", forbidden)
        result = ingest_local(local)
    assert result["unchanged"] is True
    assert result["file_count"] == 4
    assert result["duplicate_count"] == 1
    assert state(local).collection == old.collection
    original.unlink()
    response = client.get(f"/documents/{identifier}?source=markdown", headers=auth)
    assert response.status_code == 200
    assert response.json()["relative_path"] == "nested/copy.HTM"
    assert ingest_local(local)["unchanged"] is True
    duplicate.unlink()
    assert ingest_local(local)["page_count"] == 2
    assert identifier not in state(local).manifest
    assert runtime.vectors.embedder.embedded == embedded


def test_invalid_html_never_applies_deletions_or_index_changes(local, monkeypatch):
    ingest_local(local)
    runtime, client, auth, root, *_ = local
    old = state(local).collection
    (root / "alpha.md").unlink()
    (root / "invalid.html").write_bytes(b"<body>\xff</body>")

    def forbidden(*args, **kwargs):
        raise AssertionError("Every HTML document must validate before index changes")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    response = client.post("/ingestions", headers=auth, json={"source": "markdown"})
    assert response.status_code == 202
    runtime.ingestion.tick()
    response = client.get(f"/ingestions/{response.json()['id']}", headers=auth)
    assert response.json()["status"] == "failed"
    assert state(local).collection == old


def test_identical_contents_are_a_true_noop_including_bom_and_line_endings(local, monkeypatch):
    ingest_local(local)
    runtime, _, _, root, *_ = local
    old = state(local)
    embedded = runtime.vectors.embedder.embedded
    text = (root / "alpha.md").read_text(encoding="utf-8")
    (root / "alpha.md").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode())

    def forbidden(*args, **kwargs):
        raise AssertionError("An unchanged folder must not touch the vector index")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    monkeypatch.setattr(runtime.vectors, "index_page", forbidden)
    job = ingest_local(local)
    assert job["unchanged"] is True
    assert runtime.vectors.embedder.embedded == embedded
    assert state(local).collection == old.collection
    assert state(local).published_at == old.published_at
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, old.collection) is None


def test_changed_documents_only_embed_new_content(local):
    ingest_local(local)
    runtime, _, _, root, *_ = local
    old = state(local).collection
    count = runtime.vectors.embedder.embedded
    (root / "alpha.md").write_text("# Alpha\n\nA changed retry policy.\n", encoding="utf-8")
    result = ingest_local(local)
    assert result["unchanged"] is False
    assert runtime.vectors.embedder.embedded == count + 1
    assert result["page_count"] == 2
    with runtime.db.session() as session:
        assert session.get(RetiredIndex, old) is not None
        assert (
            session.get(RetiredIndex, session.get(LocalConsumptionIndexState, 1).collection) is None
        )


def test_changed_content_reuses_unchanged_chunks_under_its_previous_path(local):
    runtime, _, _, root, *_ = local
    path = root / "long.md"
    path.write_text("\n\n".join(letter * 750 for letter in "ABC"), encoding="utf-8")
    ingest_local(local)
    embedded = runtime.vectors.embedder.embedded
    path.write_text("\n\n".join(letter * 750 for letter in "ABD"), encoding="utf-8")
    assert ingest_local(local)["page_count"] == 3
    assert runtime.vectors.embedder.embedded == embedded + 1


def test_adding_a_normalized_duplicate_is_a_true_index_noop(local, monkeypatch):
    runtime, _, _, root, *_ = local
    ingest_local(local)
    old = state(local)
    text = (root / "alpha.md").read_text(encoding="utf-8")
    (root / "copy.MD").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode())

    def forbidden(*args, **kwargs):
        raise AssertionError("A duplicate must not rebuild or write the vector index")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    monkeypatch.setattr(runtime.vectors, "index_page", forbidden)
    result = ingest_local(local)
    assert result["file_count"] == 3
    assert result["page_count"] == 2
    assert result["duplicate_count"] == 1
    assert result["unchanged"] is True
    assert state(local).collection == old.collection
    assert state(local).published_at == old.published_at


def test_duplicates_are_ignored_and_alias_changes_do_not_rebuild_index(local):
    runtime, client, auth, root, *_ = local
    contents = (root / "alpha.md").read_text(encoding="utf-8")
    (root / "nested" / "duplicate.md").write_text(contents, encoding="utf-8")
    first = ingest_local(local)
    assert first["file_count"] == 3
    assert first["page_count"] == 2
    assert first["duplicate_count"] == 1
    old = state(local)
    count = runtime.vectors.embedder.embedded
    identifier = next(key for key, entry in old.manifest.items() if entry["path"] == "alpha.md")
    (root / "alpha.md").unlink()
    response = client.get(f"/documents/{identifier}?source=markdown", headers=auth)
    assert response.status_code == 200
    assert response.json()["relative_path"] == "nested/duplicate.md"
    result = ingest_local(local)
    assert result["unchanged"] is True
    assert result["duplicate_count"] == 0
    assert runtime.vectors.embedder.embedded == count
    assert state(local).collection == old.collection
    assert state(local).manifest[identifier]["paths"] == ["nested/duplicate.md"]
    (root / "nested" / "duplicate.md").unlink()
    assert ingest_local(local)["page_count"] == 1
    assert identifier not in state(local).manifest
    assert runtime.vectors.embedder.embedded == count


def test_same_filename_with_different_contents_is_not_a_duplicate(local):
    _, _, _, root, *_ = local
    (root / "nested" / "alpha.md").write_text("# Different\n\nDistinct content.", encoding="utf-8")
    result = ingest_local(local)
    assert result["page_count"] == 3
    assert result["duplicate_count"] == 0


def test_missing_and_changed_local_files_are_not_released_before_reindex(local):
    ingest_local(local)
    _, client, auth, root, *_ = local
    manifest = state(local).manifest
    identifiers = {entry["path"]: key for key, entry in manifest.items()}
    (root / "alpha.md").unlink()
    (root / "nested" / "beta.MD").write_text("# Changed\n\nNew version.", encoding="utf-8")
    assert (
        client.get(
            f"/documents/{identifiers['alpha.md']}?source=markdown", headers=auth
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/documents/{identifiers['nested/beta.MD']}?source=markdown", headers=auth
        ).status_code
        == 409
    )
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
    )
    assert response.json()["results"] == []


def test_deletions_are_mirrored_without_embedding_including_an_empty_folder(local):
    ingest_local(local)
    runtime, client, auth, root, *_ = local
    count = runtime.vectors.embedder.embedded
    (root / "alpha.md").unlink()
    assert ingest_local(local)["page_count"] == 1
    assert runtime.vectors.embedder.embedded == count
    (root / "nested" / "beta.MD").unlink()
    assert ingest_local(local)["page_count"] == 0
    assert state(local).manifest == {}
    assert runtime.vectors.embedder.embedded == count
    assert (
        client.post(
            "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
        ).json()["results"]
        == []
    )
    assert ingest_local(local)["unchanged"] is True


def test_unavailable_root_preserves_published_index_and_returns_explicit_error(local):
    ingest_local(local)
    runtime, client, auth, root, *_ = local
    old = state(local).collection
    root.rename(root.parent / "temporarily-unavailable")
    response = client.post("/ingestions", headers=auth, json={"source": "markdown"})
    runtime.ingestion.tick()
    result = client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()
    assert result["status"] == "failed"
    assert state(local).collection == old
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
    )
    assert response.status_code == 503


def test_partial_or_invalid_scan_never_changes_vectors_or_publishes_deletions(local, monkeypatch):
    ingest_local(local)
    runtime, client, auth, root, *_ = local
    old = state(local).collection
    (root / "alpha.md").unlink()
    (root / "invalid.md").write_bytes(b"\xff")

    def forbidden(*args, **kwargs):
        raise AssertionError("All files must validate before any index changes")

    monkeypatch.setattr(runtime.vectors, "create", forbidden)
    monkeypatch.setattr(runtime.vectors, "index_page", forbidden)
    response = client.post("/ingestions", headers=auth, json={"source": "markdown"})
    runtime.ingestion.tick()
    assert (
        client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()["status"]
        == "failed"
    )
    assert state(local).collection == old


def test_source_errors_are_not_silently_filtered_as_stale_results(local, monkeypatch):
    ingest_local(local)
    runtime, client, auth, *_ = local

    def unsafe(*args, **kwargs):
        raise AppError(409, "Opened document escaped the configured Markdown root.")

    monkeypatch.setattr(runtime.local_consumption, "document", unsafe)
    response = client.post(
        "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
    )
    assert response.status_code == 409
    assert "escaped" in response.json()["detail"]


def test_local_source_is_restricted_to_the_importing_active_admin(local):
    ingest_local(local)
    runtime, client, _, _, user_id, password = local
    with runtime.db.session() as session:
        role = session.scalar(select(Role).where(Role.code == "admin"))
        session.add(
            User(
                username="other", password_hash=runtime.auth.hasher.hash(password), role_id=role.id
            )
        )
    for username in ("reader", "other"):
        auth = headers(client, password, username)
        assert (
            client.post(
                "/documents/search", headers=auth, json={"query": "policy", "source": "markdown"}
            ).status_code
            == 403
        )
        assert (
            client.post("/ingestions", headers=auth, json={"source": "markdown"}).status_code == 403
        )
    with runtime.db.session() as session:
        session.get(User, user_id).role_id = session.scalar(
            select(Role.id).where(Role.code == "reader")
        )
    assert (
        client.post(
            "/documents/search", headers=local[2], json={"query": "policy", "source": "markdown"}
        ).status_code
        == 403
    )


def test_local_schedule_and_root_change_require_authorized_configuration(local, tmp_path):
    runtime, client, auth, *_ = local
    response = client.put(
        "/schedules/daily",
        headers=auth,
        json={
            "enabled": True,
            "hour_utc": 3,
            "minute_utc": 0,
            "source": "markdown",
        },
    )
    assert response.status_code == 200
    assert response.json()["source"] == "local_consumption"
    with runtime.db.session() as session:
        session.get(Schedule, 1).next_run_at = utcnow() - timedelta(minutes=1)
    runtime.ingestion.tick()
    old = state(local).collection
    response = client.post("/ingestions", headers=auth, json={"source": "markdown"})
    runtime.settings.local_consumption_root = tmp_path / "another"
    runtime.ingestion.tick()
    assert (
        client.get(f"/ingestions/{response.json()['id']}", headers=auth).json()["status"]
        == "failed"
    )
    assert state(local).collection == old


def test_due_schedule_is_disabled_if_its_authorized_root_changes(local, tmp_path):
    runtime, client, auth, *_ = local
    assert (
        client.put(
            "/schedules/daily",
            headers=auth,
            json={"enabled": True, "hour_utc": 3, "minute_utc": 0, "source": "markdown"},
        ).status_code
        == 200
    )
    with runtime.db.session() as session:
        session.get(Schedule, 1).next_run_at = utcnow() - timedelta(minutes=1)
    runtime.settings.local_consumption_root = tmp_path / "another"
    assert runtime.ingestion.tick() is False
    with runtime.db.session() as session:
        assert session.get(Schedule, 1).enabled is False
    runtime.settings.local_consumption_root = None
    assert (
        client.put(
            "/schedules/daily",
            headers=auth,
            json={"enabled": False, "hour_utc": 3, "minute_utc": 0, "source": "markdown"},
        ).status_code
        == 200
    )


def test_real_mcp_tools_default_to_local_consumption_and_enforce_current_owner_role(local, env):
    ingest_local(local)
    runtime, client, params, data, verifier = authorize(env)
    response = client.post(
        "/oauth/authorize",
        data=data,
        headers={"Origin": runtime.settings.public_url},
        follow_redirects=False,
    )
    assert response.status_code == 303
    code = parse_qs(urlparse(response.headers["location"]).query)["code"][0]
    response = client.post("/oauth/token", data=token_request(runtime, params, verifier, code))
    assert response.status_code == 200
    auth = {
        "Authorization": f"Bearer {response.json()['access_token']}",
        "Accept": "application/json, text/event-stream",
    }

    def invoke(method, parameters):
        response = client.post(
            "/mcp",
            headers=auth,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": parameters},
        )
        assert response.status_code == 200, response.text
        return response.json()["result"]

    invoke(
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "local-consumption-test", "version": "1"},
        },
    )
    tools = invoke("tools/list", {})["tools"]
    assert all(
        tool["inputSchema"]["properties"]["source"]["default"] == "local_consumption"
        for tool in tools
    )
    result = invoke(
        "tools/call",
        {
            "name": "search_documents",
            "arguments": {"query": "policy"},
        },
    )
    assert result.get("isError") is not True
    assert result["content"][0]["type"] == "text"
    payload = json.loads(result["content"][0]["text"])
    assert payload["source"] == "local_consumption"
    assert len(payload["results"]) == 2
    identifier = payload["results"][0]["page_id"]
    result = invoke(
        "tools/call",
        {
            "name": "get_document",
            "arguments": {"page_id": identifier},
        },
    )
    assert json.loads(result["content"][0]["text"])["text"].startswith("# ")
    for source in ("local_consumption", "markdown"):
        result = invoke(
            "tools/call",
            {
                "name": "get_document",
                "arguments": {"page_id": identifier, "source": source},
            },
        )
        assert result.get("isError") is not True, result
        assert json.loads(result["content"][0]["text"])["source"] == "local_consumption"
    with runtime.db.session() as session:
        session.get(User, local[4]).role_id = session.scalar(
            select(Role.id).where(Role.code == "reader")
        )
    result = invoke(
        "tools/call",
        {
            "name": "get_document",
            "arguments": {"page_id": identifier, "source": "markdown"},
        },
    )
    assert result["isError"] is True
    assert "structuredContent" not in result


@pytest.mark.parametrize("source", [None, "local_consumption", "markdown"])
def test_local_consumption_selectors_preserve_storage_identity_and_permissions(local, source):
    runtime, client, auth, root, *_ = local
    data = {} if source is None else {"source": source}
    response = client.post("/ingestions", headers=auth, json=data)
    assert response.status_code == 202
    assert response.json()["source"] == "local_consumption"
    assert runtime.ingestion.tick()
    first = state(local)
    assert first.collection.startswith("ragu_markdown_")
    for identifier, entry in first.manifest.items():
        assert identifier == str(
            uuid5(NAMESPACE_URL, f"markdown:{runtime.local_consumption.scope}:{entry['digest']}")
        )
    result = client.post("/documents/search", headers=auth, json={"query": "policy", **data})
    assert result.status_code == 200
    assert result.json()["source"] == "local_consumption"
    item = result.json()["results"][0]
    assert item["source"] == "local_consumption"
    assert "?source=local_consumption" in item["source_url"]
    assert str(root) not in item["source_url"]
    params = {} if source is None else {"source": source}
    response = client.get(f"/documents/{item['page_id']}", headers=auth, params=params)
    assert response.status_code == 200
    assert response.json()["source"] == "local_consumption"
    repeated = ingest_local(local, "markdown")
    assert repeated["unchanged"] is True
    assert state(local).collection == first.collection
    assert state(local).manifest == first.manifest
    schedule = client.put(
        "/schedules/daily",
        headers=auth,
        json={"enabled": True, "hour_utc": 3, "minute_utc": 0, **data},
    )
    assert schedule.status_code == 200
    assert schedule.json()["source"] == "local_consumption"
    assert client.get("/schedules/daily", headers=auth).json()["source"] == "local_consumption"
    with runtime.db.session() as session:
        stored = session.get(Schedule, 1)
        assert stored.source == "markdown"
        assert stored.source_root == runtime.local_consumption.scope


@pytest.mark.parametrize("published", [False, True])
def test_local_strategy_rechecks_authorization_after_validation_before_publication(
    local, monkeypatch, published
):
    runtime, client, auth, _, user_id, _ = local
    if published:
        ingest_local(local)
    previous = state(local)
    embedded = runtime.vectors.embedder.embedded
    scan = runtime.local_consumption.scan

    def revoke_after_validation(heartbeat):
        documents = scan(heartbeat)
        with runtime.db.session() as session:
            session.get(User, user_id).is_active = False
        return documents

    monkeypatch.setattr(runtime.local_consumption, "scan", revoke_after_validation)
    response = client.post("/ingestions", headers=auth)
    assert response.status_code == 202
    assert runtime.ingestion.tick()
    with runtime.db.session() as session:
        job = session.get(Job, UUID(response.json()["id"]))
        assert job.status == "failed"
        assert "revoked" in job.error
    current = state(local)
    if published:
        assert current.collection == previous.collection
    else:
        assert current is None
    assert runtime.vectors.embedder.embedded == embedded


def test_arbitrary_paths_and_oversized_files_are_rejected(local):
    runtime, client, auth, root, *_ = local
    assert (
        client.post(
            "/ingestions",
            headers=auth,
            json={
                "source": "markdown",
                "path": str(root.parent),
            },
        ).status_code
        == 422
    )
    with pytest.raises(AppError) as error:
        runtime.local_consumption.document("../outside.md")
    assert error.value.status == 404
    runtime.settings.local_consumption_max_file_bytes = 4
    with pytest.raises(AppError) as error:
        runtime.local_consumption.scan(lambda: None)
    assert error.value.status == 413


def test_total_byte_limit_counts_original_bytes_before_normalization(local):
    runtime, _, _, root, *_ = local
    path = root / "alpha.md"
    path.write_bytes(b"\xef\xbb\xbf" + b"\r\n" * 100)
    runtime.settings.local_consumption_max_total_bytes = 202
    with pytest.raises(AppError) as error:
        runtime.local_consumption.scan(lambda: None)
    assert error.value.status == 413


def test_hard_links_are_rejected(local, tmp_path):
    runtime, _, _, root, *_ = local
    outside = tmp_path / "outside.md"
    outside.write_text("Outside the allowed source.", encoding="utf-8")
    os.link(outside, root / "escape.md")
    with pytest.raises(AppError) as error:
        runtime.local_consumption.scan(lambda: None)
    assert error.value.status == 409


def test_symlinks_are_never_followed(local, tmp_path):
    runtime, _, _, root, *_ = local
    outside = tmp_path / "outside.md"
    outside.write_text("Outside the allowed source.", encoding="utf-8")
    try:
        (root / "escape.md").symlink_to(outside)
    except OSError:
        pytest.skip("This OS account cannot create symbolic links")
    with pytest.raises(AppError) as error:
        runtime.local_consumption.scan(lambda: None)
    assert error.value.status == 409


@pytest.mark.skipif(os.name != "nt", reason="Windows junction containment")
def test_windows_junctions_are_never_followed(local, tmp_path):
    import _winapi

    runtime, _, _, root, *_ = local
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = root / "escape"
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(AppError) as error:
            runtime.local_consumption.scan(lambda: None)
        assert error.value.status == 409
    finally:
        junction.rmdir()
