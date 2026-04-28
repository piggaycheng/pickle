from pathlib import Path
from types import SimpleNamespace

from pickleball_ai.jobs import load_job_history
from pickleball_ai.pose import (
    VideoOpenError,
    extract_pose_job,
    extract_pose_jsonl,
    serialize_pose_landmarks,
    timestamp_for_frame,
)
from pickleball_ai.schema import JobStatus, Landmark, PoseFrame, Project
from pickleball_ai.storage import create_project_layout, read_jsonl


class FakeFrameSource:
    fps = 10.0
    width = 640
    height = 360

    def __init__(self, frames=None):
        self.frames = frames or ["frame-0", "frame-1"]
        self.closed = False

    def __iter__(self):
        for index, frame in enumerate(self.frames):
            yield index, frame

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self):
        self.closed = False

    def detect(self, frame, timestamp_ms):
        return [[Landmark(x=0.1, y=0.2, z=0.3, visibility=0.9, presence=0.8)]]

    def close(self):
        self.closed = True


def test_timestamp_for_frame_uses_fps():
    assert timestamp_for_frame(3, 10.0) == 300


def test_serialize_pose_landmarks_keeps_landmark_fields():
    poses = serialize_pose_landmarks(
        [[SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9, presence=0.8)]]
    )

    assert poses[0][0].x == 0.1
    assert poses[0][0].visibility == 0.9


def test_extract_pose_jsonl_writes_pose_frames(tmp_path):
    output_path = tmp_path / "pose.jsonl"

    count = extract_pose_jsonl(
        frame_source=FakeFrameSource(),
        detector=FakeDetector(),
        output_path=output_path,
    )

    records = read_jsonl(output_path, PoseFrame)
    assert count == 2
    assert [record.timestamp_ms for record in records] == [0, 100]
    assert records[0].image_width == 640
    assert records[0].poses[0][0].x == 0.1


def test_extract_pose_job_success_writes_artifact_and_job_states(tmp_path, monkeypatch):
    monkeypatch.setattr("pickleball_ai.pose.download_model", lambda model_path: Path(model_path))
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)

    job = extract_pose_job(
        paths,
        video_id="video-1",
        frame_source_factory=lambda input_path: FakeFrameSource(),
        detector_factory=lambda model_path: FakeDetector(),
    )

    assert job.status == JobStatus.SUCCESS
    assert job.output_refs == [f"artifacts/jobs/{job.job_id}/pose.jsonl"]
    assert job.params["frame_count"] == 2
    assert (paths.root / job.output_refs[0]).exists()
    assert [state.status for state in load_job_history(paths.processing_jobs_jsonl)] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.SUCCESS,
    ]


def test_extract_pose_job_failure_records_failed_state(tmp_path, monkeypatch):
    monkeypatch.setattr("pickleball_ai.pose.download_model", lambda model_path: Path(model_path))
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)

    def failing_frame_source(input_path):
        raise VideoOpenError("no video")

    job = extract_pose_job(
        paths,
        video_id="video-1",
        frame_source_factory=failing_frame_source,
        detector_factory=lambda model_path: FakeDetector(),
    )

    assert job.status == JobStatus.FAILED
    assert job.error_type == "VideoOpenError"
    assert "retry" in (job.user_action or "").lower()
    assert [state.status for state in load_job_history(paths.processing_jobs_jsonl)] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.FAILED,
    ]
