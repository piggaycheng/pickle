from pathlib import Path

import pytest

from pickleball_ai.schema import Project, SourceType, Video
from pickleball_ai.storage import (
    JsonlReadError,
    StorageError,
    UnsafePathError,
    append_jsonl,
    create_project_layout,
    get_data_root,
    read_json,
    read_jsonl,
    safe_join,
    write_json,
)


def test_default_data_root_is_repo_local_datasets(tmp_path, monkeypatch):
    monkeypatch.delenv("PICKLE_DATA_DIR", raising=False)

    assert get_data_root(tmp_path) == tmp_path / "datasets"


def test_pickle_data_dir_override_works(tmp_path, monkeypatch):
    override = tmp_path / "external-data"
    monkeypatch.setenv("PICKLE_DATA_DIR", str(override))

    assert get_data_root(tmp_path) == override


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "..", "outside")


def test_project_directory_layout_is_created(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=1000,
    )

    paths = create_project_layout(tmp_path, project, video)

    assert paths.project_json.exists()
    assert paths.video_json.exists()
    assert paths.processing_jobs_jsonl.exists()
    assert paths.videos_dir.is_dir()
    assert paths.jobs_dir.is_dir()
    assert paths.state_dir.is_dir()
    assert paths.exports_dir.is_dir()


def test_project_directory_does_not_overwrite_existing_project(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    create_project_layout(tmp_path, project)

    with pytest.raises(StorageError):
        create_project_layout(tmp_path, project)


def test_json_write_read_round_trip(tmp_path):
    project = Project(project_id="project-1")
    path = tmp_path / "project.json"

    write_json(path, project)
    loaded = read_json(path, Project)

    assert loaded == project


def test_jsonl_append_read_round_trip(tmp_path):
    first = Project(project_id="project-1")
    second = Project(project_id="project-2")
    path = tmp_path / "projects.jsonl"

    append_jsonl(path, first)
    append_jsonl(path, second)

    assert read_jsonl(path, Project) == [first, second]


def test_empty_jsonl_is_valid(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()

    assert read_jsonl(path, Project) == []


def test_malformed_jsonl_raises_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"project_id": "ok"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(JsonlReadError) as exc_info:
        read_jsonl(path, Project)

    assert exc_info.value.line_number == 2

