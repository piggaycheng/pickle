from pathlib import Path

from pickleball_ai.events import (
    HitCandidateConfig,
    extract_hit_candidates_job,
    extract_hit_candidates_jsonl,
    generate_hit_candidates,
)
from pickleball_ai.jobs import load_job_history
from pickleball_ai.schema import HitCandidate, JobStatus, Landmark, PoseFrame, Project
from pickleball_ai.storage import create_project_layout, read_jsonl, safe_join, write_jsonl


def make_pose_frame(frame_index: int, wrist_x: float | None) -> PoseFrame:
    landmarks = [Landmark(x=0.0, y=0.0, visibility=1.0) for _ in range(17)]
    if wrist_x is None:
        poses = []
    else:
        landmarks[15] = Landmark(x=wrist_x, y=0.5, visibility=1.0)
        landmarks[16] = Landmark(x=wrist_x, y=0.5, visibility=1.0)
        poses = [landmarks]
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        image_width=100,
        image_height=100,
        poses=poses,
    )


def test_generate_hit_candidates_detects_motion_peak():
    frames = [
        make_pose_frame(0, 0.10),
        make_pose_frame(1, 0.20),
        make_pose_frame(2, 0.90),
        make_pose_frame(3, 0.91),
        make_pose_frame(4, 0.92),
    ]

    candidates = generate_hit_candidates(
        frames,
        video_id="video-1",
        config=HitCandidateConfig(score_threshold_px=50, min_gap_ms=250),
    )

    assert len(candidates) == 1
    assert candidates[0].timestamp_ms == 200
    assert candidates[0].time_window.start_ms == 0
    assert candidates[0].features["score_px"] >= 50


def test_threshold_suppresses_small_motion():
    frames = [
        make_pose_frame(0, 0.10),
        make_pose_frame(1, 0.12),
        make_pose_frame(2, 0.14),
        make_pose_frame(3, 0.16),
    ]

    assert (
        generate_hit_candidates(
            frames,
            video_id="video-1",
            config=HitCandidateConfig(score_threshold_px=50),
        )
        == []
    )


def test_missing_landmarks_do_not_create_candidates():
    frames = [make_pose_frame(index, None) for index in range(4)]

    assert generate_hit_candidates(frames, video_id="video-1") == []


def test_nearby_peaks_are_clustered_by_time():
    frames = [
        make_pose_frame(0, 0.10),
        make_pose_frame(1, 0.70),
        make_pose_frame(2, 0.71),
        make_pose_frame(3, 0.15),
        make_pose_frame(4, 0.90),
        make_pose_frame(5, 0.91),
    ]

    candidates = generate_hit_candidates(
        frames,
        video_id="video-1",
        config=HitCandidateConfig(score_threshold_px=50, min_gap_ms=350),
    )

    assert len(candidates) == 1
    assert candidates[0].timestamp_ms == 400


def test_extract_hit_candidates_jsonl_round_trip(tmp_path):
    pose_path = tmp_path / "pose.jsonl"
    output_path = tmp_path / "hit_candidates.jsonl"
    write_jsonl(
        pose_path,
        [
            make_pose_frame(0, 0.10),
            make_pose_frame(1, 0.20),
            make_pose_frame(2, 0.90),
            make_pose_frame(3, 0.91),
        ],
    )

    count = extract_hit_candidates_jsonl(
        pose_path=pose_path,
        output_path=output_path,
        video_id="video-1",
        source_job_id="pose-job-1",
        source_pose_ref="artifacts/jobs/pose-job-1/pose.jsonl",
        config=HitCandidateConfig(score_threshold_px=50),
    )

    records = read_jsonl(output_path, HitCandidate)
    assert count == 1
    assert records[0].source_job_id == "pose-job-1"
    assert records[0].source_pose_ref == "artifacts/jobs/pose-job-1/pose.jsonl"


def test_extract_hit_candidates_job_records_success(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    paths = create_project_layout(tmp_path, project)
    pose_ref = "artifacts/jobs/pose-job-1/pose.jsonl"
    pose_path = safe_join(paths.root, *pose_ref.split("/"))
    write_jsonl(
        pose_path,
        [
            make_pose_frame(0, 0.10),
            make_pose_frame(1, 0.20),
            make_pose_frame(2, 0.90),
            make_pose_frame(3, 0.91),
        ],
    )

    job = extract_hit_candidates_job(
        paths,
        video_id="video-1",
        pose_ref=pose_ref,
        source_job_id="pose-job-1",
        config=HitCandidateConfig(score_threshold_px=50),
    )

    assert job.status == JobStatus.SUCCESS
    assert job.params["candidate_count"] == 1
    assert Path(paths.root / job.output_refs[0]).exists()
    assert [state.status for state in load_job_history(paths.processing_jobs_jsonl)] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.SUCCESS,
    ]


def test_extract_hit_candidates_job_records_failure(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    job = extract_hit_candidates_job(
        paths,
        video_id="video-1",
        pose_ref="artifacts/jobs/missing/pose.jsonl",
    )

    assert job.status == JobStatus.FAILED
    assert job.error_type in {"FileNotFoundError", "JsonlReadError"}
    assert "pose artifact" in (job.user_action or "")
