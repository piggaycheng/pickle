import pytest
from pydantic import ValidationError

from pickleball_ai.schema import (
    JobStatus,
    JobType,
    ProcessingJob,
    Project,
    SourceType,
    TimeWindow,
    TimingConfidence,
    Video,
    Annotation,
)


def test_project_accepts_current_schema_version():
    project = Project(schema_version="0.1.0")

    assert project.schema_version == "0.1.0"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        Project(schema_version="0.1.0", unexpected=True)


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(ValidationError):
        Project(schema_version="9.9.9")


def test_video_rejects_negative_duration():
    with pytest.raises(ValidationError):
        Video(
            source_type=SourceType.LOCAL,
            local_path="videos/source.mp4",
            fps=30,
            duration_ms=-1,
        )


def test_video_rejects_absolute_artifact_ref():
    with pytest.raises(ValidationError):
        Video(
            source_type=SourceType.LOCAL,
            local_path="C:/tmp/source.mp4",
            fps=30,
            duration_ms=1000,
        )


def test_processing_job_supports_lifecycle_states():
    job = ProcessingJob(video_id="video-1", job_type=JobType.METADATA)

    assert job.status == JobStatus.QUEUED


def test_failed_job_requires_error_payload():
    with pytest.raises(ValidationError):
        ProcessingJob(
            video_id="video-1",
            job_type=JobType.METADATA,
            status=JobStatus.FAILED,
        )


def test_success_job_requires_output_refs():
    with pytest.raises(ValidationError):
        ProcessingJob(
            video_id="video-1",
            job_type=JobType.METADATA,
            status=JobStatus.SUCCESS,
        )


def test_annotation_time_window_requires_end_after_start():
    with pytest.raises(ValidationError):
        TimeWindow(start_ms=200, end_ms=100)


def test_annotation_validates_clip_range():
    with pytest.raises(ValidationError):
        Annotation(
            video_id="video-1",
            event_time_ms=100,
            time_window=TimeWindow(start_ms=90, end_ms=110),
            timing_confidence=TimingConfidence.EXACT,
            player_id="A",
            action="drive",
            confidence=1,
            source="manual",
            clip_start_ms=200,
            clip_end_ms=100,
        )

