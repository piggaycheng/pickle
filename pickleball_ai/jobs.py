from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import JobStatus, JobType, ProcessingJob, utc_now
from .storage import ProjectPaths, append_jsonl, read_jsonl, write_job_manifest


class JobTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED},
    JobStatus.RUNNING: {JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCELED},
    JobStatus.SUCCESS: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELED: set(),
}


def create_job(
    paths: ProjectPaths,
    *,
    video_id: str,
    job_type: JobType,
    input_refs: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> ProcessingJob:
    job = ProcessingJob(
        video_id=video_id,
        job_type=job_type,
        status=JobStatus.QUEUED,
        input_refs=input_refs or [],
        params=params or {},
    )
    append_jsonl(paths.processing_jobs_jsonl, job)
    write_job_manifest(paths, job)
    return job


def transition_job(
    job: ProcessingJob,
    status: JobStatus,
    *,
    output_refs: list[str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    user_action: str | None = None,
    now: datetime | None = None,
) -> ProcessingJob:
    if status not in ALLOWED_TRANSITIONS[job.status]:
        raise JobTransitionError(f"invalid transition: {job.status} -> {status}")
    timestamp = now or utc_now()
    updates: dict[str, Any] = {"status": status}
    if status == JobStatus.RUNNING:
        updates["started_at"] = timestamp
    if status in {JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCELED}:
        updates["finished_at"] = timestamp
    if output_refs is not None:
        updates["output_refs"] = output_refs
    if status == JobStatus.FAILED:
        updates["error_type"] = error_type
        updates["error_message"] = error_message
        updates["user_action"] = user_action
    data = job.model_dump()
    data.update(updates)
    return ProcessingJob.model_validate(data)


def append_job_state(paths: ProjectPaths, job: ProcessingJob) -> None:
    append_jsonl(paths.processing_jobs_jsonl, job)


def load_job_history(processing_jobs_jsonl: Path) -> list[ProcessingJob]:
    return read_jsonl(processing_jobs_jsonl, ProcessingJob)
