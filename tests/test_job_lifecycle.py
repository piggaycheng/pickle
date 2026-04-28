import pytest

from pickleball_ai.jobs import (
    JobTransitionError,
    append_job_state,
    create_job,
    load_job_history,
    transition_job,
)
from pickleball_ai.schema import JobStatus, JobType, Project
from pickleball_ai.schema import ProcessingJob
from pickleball_ai.storage import create_project_layout


def test_create_job_writes_processing_jobs_jsonl(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)

    job = create_job(paths, video_id="video-1", job_type=JobType.METADATA)

    assert load_job_history(paths.processing_jobs_jsonl) == [job]
    assert paths.job_manifest(job.job_id).exists()


def test_queued_running_success_transition(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)
    job = create_job(paths, video_id="video-1", job_type=JobType.METADATA)

    running = transition_job(job, JobStatus.RUNNING)
    success = transition_job(running, JobStatus.SUCCESS, output_refs=["artifacts/jobs/job-1/output.json"])
    append_job_state(paths, running)
    append_job_state(paths, success)

    assert success.status == JobStatus.SUCCESS
    assert success.output_refs == ["artifacts/jobs/job-1/output.json"]
    assert len(load_job_history(paths.processing_jobs_jsonl)) == 3


def test_queued_failed_transition_exposes_user_action():
    job = create_project_job()

    failed = transition_job(
        job,
        JobStatus.FAILED,
        error_type="VideoOpenError",
        error_message="Video could not be opened",
        user_action="Choose another video",
    )

    assert failed.status == JobStatus.FAILED
    assert failed.error_type == "VideoOpenError"
    assert failed.user_action == "Choose another video"


def test_failed_transition_requires_error_payload():
    job = create_project_job()

    with pytest.raises(ValueError):
        transition_job(job, JobStatus.FAILED)


def test_success_transition_requires_output_refs():
    running = transition_job(create_project_job(), JobStatus.RUNNING)

    with pytest.raises(ValueError):
        transition_job(running, JobStatus.SUCCESS)


def test_invalid_transition_is_rejected():
    job = create_project_job()

    with pytest.raises(JobTransitionError):
        transition_job(job, JobStatus.SUCCESS, output_refs=["artifacts/jobs/job-1/output.json"])


def test_job_artifact_manifest_path_stays_under_project_root(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)
    job = create_job(paths, video_id="video-1", job_type=JobType.METADATA)

    manifest_path = paths.job_manifest(job.job_id).resolve()

    assert manifest_path.is_relative_to(paths.root.resolve())


def create_project_job():
    return ProcessingJob(video_id="video-1", job_type=JobType.METADATA)
