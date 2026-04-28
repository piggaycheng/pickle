from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .schema import JobManifest, ProcessingJob, Project, Video

T = TypeVar("T", bound=BaseModel)


class StorageError(Exception):
    pass


class UnsafePathError(StorageError):
    pass


class JsonlReadError(StorageError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        super().__init__(f"{path}:{line_number}: {message}")
        self.path = path
        self.line_number = line_number


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    project_json: Path
    video_json: Path
    processing_jobs_jsonl: Path
    videos_dir: Path
    artifacts_dir: Path
    jobs_dir: Path
    state_dir: Path
    exports_dir: Path

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_manifest(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"


def get_data_root(project_root: Path | None = None) -> Path:
    override = os.environ.get("PICKLE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    base = project_root if project_root is not None else Path.cwd()
    return (base / "datasets").resolve()


def ensure_inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError(f"path escapes data root: {path}") from exc
    return path


def safe_join(root: Path, *parts: str) -> Path:
    for part in parts:
        candidate = Path(part)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise UnsafePathError(f"unsafe path component: {part}")
    return ensure_inside(root, root.joinpath(*parts))


def project_paths(data_root: Path, project_id: str) -> ProjectPaths:
    root = safe_join(data_root, project_id)
    artifacts_dir = root / "artifacts"
    jobs_dir = artifacts_dir / "jobs"
    return ProjectPaths(
        root=root,
        project_json=root / "project.json",
        video_json=root / "video.json",
        processing_jobs_jsonl=root / "processing_jobs.jsonl",
        videos_dir=root / "videos",
        artifacts_dir=artifacts_dir,
        jobs_dir=jobs_dir,
        state_dir=root / "state",
        exports_dir=root / "exports",
    )


def create_project_layout(data_root: Path, project: Project, video: Video | None = None) -> ProjectPaths:
    paths = project_paths(data_root, project.project_id)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise StorageError(f"project already exists: {paths.root}")
    for directory in (
        paths.root,
        paths.videos_dir,
        paths.jobs_dir,
        paths.state_dir,
        paths.exports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(paths.project_json, project)
    if video is not None:
        write_json(paths.video_json, video)
    paths.processing_jobs_jsonl.touch(exist_ok=True)
    return paths


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    data = model.model_dump_json(indent=2)
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def read_json(path: Path, model_cls: type[T]) -> T:
    with path.open("r", encoding="utf-8") as handle:
        return model_cls.model_validate_json(handle.read())


def append_jsonl(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(model.model_dump_json())
        handle.write("\n")
        handle.flush()


def write_jsonl(path: Path, models: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for model in models:
            handle.write(model.model_dump_json())
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def read_jsonl(path: Path, model_cls: type[T]) -> list[T]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(model_cls.model_validate(json.loads(stripped)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise JsonlReadError(path, line_number, str(exc)) from exc
    return records


def write_job_manifest(paths: ProjectPaths, job: ProcessingJob) -> JobManifest:
    job_dir = paths.job_dir(job.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = JobManifest(
        job_id=job.job_id,
        project_id=paths.root.name,
        artifact_ref=f"artifacts/jobs/{job.job_id}",
    )
    write_json(paths.job_manifest(job.job_id), manifest)
    return manifest
