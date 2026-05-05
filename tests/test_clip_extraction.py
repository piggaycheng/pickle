import subprocess

from pickleball_ai.clips import (
    CLIPS_DIR_REF,
    ClipExtractionError,
    build_ffmpeg_clip_command,
    clip_ref_for_annotation,
    extract_clip,
    extract_clips_job,
)
from pickleball_ai.jobs import load_job_history
from pickleball_ai.schema import Annotation, JobStatus, JobType, Project, SourceType, TimeWindow, TimingConfidence, Video
from pickleball_ai.storage import create_project_layout


def make_annotation(annotation_id: str = "ann-1") -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        video_id="video-1",
        event_time_ms=1000,
        time_window=TimeWindow(start_ms=900, end_ms=1100),
        timing_confidence=TimingConfidence.EXACT,
        player_id="A",
        action="drive",
        confidence=1,
        source="manual",
        clip_start_ms=800,
        clip_end_ms=1200,
    )


def make_paths(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    source = paths.root / "videos" / "source.mp4"
    source.write_bytes(b"fake video")
    return paths


def test_build_ffmpeg_clip_command_uses_argv_without_shell(tmp_path):
    command = build_ffmpeg_clip_command(
        ffmpeg_path="ffmpeg",
        source_path=tmp_path / "source.mp4",
        output_path=tmp_path / "clip.mp4",
        start_ms=800,
        end_ms=1200,
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-ss",
        "0.800",
        "-i",
        str(tmp_path / "source.mp4"),
        "-t",
        "0.400",
        "-c",
        "copy",
        str(tmp_path / "clip.mp4"),
    ]


def test_clip_ref_for_annotation_stays_under_exports_clips():
    assert clip_ref_for_annotation(make_annotation("ann-1")) == f"{CLIPS_DIR_REF}/ann-1.mp4"


def test_extract_clip_writes_expected_ref_with_fake_runner(tmp_path):
    paths = make_paths(tmp_path)

    def fake_run(command, **kwargs):
        output_path = command[-1]
        with open(output_path, "wb") as handle:
            handle.write(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    clip_ref = extract_clip(
        paths,
        annotation=make_annotation(),
        source_video_ref="videos/source.mp4",
        run_command=fake_run,
    )

    assert clip_ref == "exports/clips/ann-1.mp4"
    assert (paths.root / clip_ref).read_bytes() == b"clip"


def test_extract_clip_rejects_missing_source_video(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    try:
        extract_clip(paths, annotation=make_annotation(), source_video_ref="videos/missing.mp4")
    except ClipExtractionError as exc:
        assert "source video not found" in str(exc)
    else:
        raise AssertionError("missing source video should fail")


def test_extract_clip_reports_missing_ffmpeg(tmp_path):
    paths = make_paths(tmp_path)

    def missing_ffmpeg(command, **kwargs):
        raise FileNotFoundError("ffmpeg")

    try:
        extract_clip(
            paths,
            annotation=make_annotation(),
            source_video_ref="videos/source.mp4",
            ffmpeg_path="missing-ffmpeg",
            run_command=missing_ffmpeg,
        )
    except ClipExtractionError as exc:
        assert "ffmpeg executable not found" in str(exc)
    else:
        raise AssertionError("missing ffmpeg should fail")


def test_extract_clip_reports_nonzero_ffmpeg_exit(tmp_path):
    paths = make_paths(tmp_path)

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad range")

    try:
        extract_clip(paths, annotation=make_annotation(), source_video_ref="videos/source.mp4", run_command=failed_run)
    except ClipExtractionError as exc:
        assert "bad range" in str(exc)
    else:
        raise AssertionError("nonzero ffmpeg exit should fail")


def test_extract_clips_job_records_success(tmp_path):
    paths = make_paths(tmp_path)

    def fake_run(command, **kwargs):
        with open(command[-1], "wb") as handle:
            handle.write(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    job = extract_clips_job(
        paths,
        video_id="video-1",
        source_video_ref="videos/source.mp4",
        annotations=[make_annotation()],
        run_command=fake_run,
    )

    assert job.status == JobStatus.SUCCESS
    assert job.job_type == JobType.CLIP_EXTRACTION
    assert job.output_refs == ["exports/clips/ann-1.mp4"]
    assert [item.status for item in load_job_history(paths.processing_jobs_jsonl)] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.SUCCESS,
    ]


def test_extract_clips_job_records_failed_user_action(tmp_path):
    paths = make_paths(tmp_path)

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="codec unavailable")

    job = extract_clips_job(
        paths,
        video_id="video-1",
        source_video_ref="videos/source.mp4",
        annotations=[make_annotation()],
        run_command=failed_run,
    )

    assert job.status == JobStatus.FAILED
    assert job.error_type == "ClipExtractionError"
    assert "codec unavailable" in job.error_message
    assert "Install ffmpeg" in job.user_action
