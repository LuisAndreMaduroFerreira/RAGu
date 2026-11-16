from dataclasses import dataclass

import pytest

from ragu.errors import AppError
from ragu.file_utils import FileUtils


@dataclass(frozen=True)
class Version:
    relative_path: str
    modified_ns: int
    version: str


@pytest.mark.parametrize(
    "filename",
    [
        "Guide.md",
        "Guide (1).md",
        "Guide (12).HTML",
        "Guide - Copy.md",
        "Guide - COPY (2).htm",
        "Guide - Copy (2).TXT",
        "Guide_copy_2.md",
        "Guide_v2.md",
        "Guide-v2.1.md",
        "Guide v2.1.3.md",
        "Guide (v2).md",
        "Guide_version_3.md",
        "Guide (Version 3).md",
        "Guide_v2 - Copy (3).md",
    ],
)
def test_export_copy_and_version_suffixes_share_one_directory_scoped_key(filename):
    assert FileUtils.version_key(f"folder/{filename}") == FileUtils.version_key("folder/Guide.md")


def test_business_numbers_and_distinct_directories_do_not_merge():
    names = (
        "phase_1.md",
        "phase_2.md",
        "Guide 2026-11-12.md",
        "Guide vNext.md",
        "Guide.md",
        "nested/Guide.md",
        "other/Guide_v2.md",
        "Copy.md",
        "(1).md",
    )
    assert len({FileUtils.version_key(name) for name in names}) == len(names)


def test_newest_uses_precise_timestamp_not_version_number_or_input_order():
    old = Version("Guide_v99.md", 100, "old")
    newest = Version("Guide.md", 101, "new")
    for versions in ([old, newest], [newest, old]):
        assert FileUtils.newest_version(versions) is newest
    assert len(FileUtils.group_versions([old, newest])) == 1


def test_identical_ties_are_deterministic_but_conflicting_latest_ties_fail():
    original = Version("Guide.md", 100, "same")
    copy = Version("Guide (1).md", 100, "same")
    assert FileUtils.newest_version([original, copy]) is copy
    assert FileUtils.newest_version([copy, original]) is copy
    conflict = Version("Guide_v2.md", 100, "different")
    with pytest.raises(AppError) as error:
        FileUtils.newest_version([original, conflict])
    assert error.value.status == 409
    assert "same latest modification time" in error.value.message
    newest = Version("Guide_v3.md", 101, "winner")
    assert FileUtils.newest_version([original, conflict, newest]) is newest
