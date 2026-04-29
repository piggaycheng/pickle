from pickleball_ai.annotations import update_annotation_event
from pickleball_ai.jobs import append_job_state, create_job, transition_job
from pickleball_ai.metrics import compute_metrics, metrics_path, rebuild_metrics
from pickleball_ai.queue import write_review_queue
from pickleball_ai.schema import (
    Annotation,
    CoverageSpan,
    CoverageState,
    JobStatus,
    JobType,
    Project,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    SourceType,
    TargetRef,
    TimeWindow,
    TimingConfidence,
    Video,
)
from pickleball_ai.storage import create_project_layout, read_json, write_jsonl
from pickleball_ai.summary import build_project_summary, project_summary_path, rebuild_project_summary


def make_annotation(player_id: str = "A", action: str = "drive") -> Annotation:
    return Annotation(
        annotation_id="ann-1",
        video_id="video-1",
        event_time_ms=1000,
        time_window=TimeWindow(start_ms=900, end_ms=1100),
        timing_confidence=TimingConfidence.EXACT,
        player_id=player_id,
        action=action,
        confidence=1,
        source="manual",
        clip_start_ms=800,
        clip_end_ms=1200,
    )


def test_metrics_use_reviewed_duration_denominator():
    coverage = [
        CoverageSpan(start_ms=0, end_ms=60_000, state=CoverageState.REVIEWED, source_event_id="cov-1"),
        CoverageSpan(start_ms=60_000, end_ms=120_000, state=CoverageState.SKIPPED_NON_GAME, source_event_id="cov-2"),
        CoverageSpan(start_ms=120_000, end_ms=180_000, state=CoverageState.UNREVIEWED, source_event_id="cov-3"),
    ]

    metrics = compute_metrics(annotations=[make_annotation(), make_annotation()], coverage=coverage)

    assert metrics.reviewed_duration_ms == 60_000
    assert metrics.skipped_non_game_duration_ms == 60_000
    assert metrics.unreviewed_duration_ms == 60_000
    assert metrics.annotations_per_reviewed_minute == 2


def test_metrics_distinguish_unavailable_from_zero():
    metrics = compute_metrics(annotations=[], coverage=[])

    assert metrics.annotation_count == 0
    assert metrics.annotations_per_reviewed_minute is None


def test_metrics_count_identity_and_action_corrections():
    before = make_annotation(player_id="B", action="drive")
    player_fixed = update_annotation_event(before, {"player_id": "A", "action": "drive"})
    action_fixed = update_annotation_event(before, {"player_id": "B", "action": "slice"})

    metrics = compute_metrics(annotation_events=[player_fixed, action_fixed])

    assert metrics.identity_corrections_count == 1
    assert metrics.action_corrections_by_label == {"slice": 1}


def test_rebuild_metrics_writes_metrics_json(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    metrics = rebuild_metrics(paths, annotations=[make_annotation()])

    assert metrics_path(paths).exists()
    assert read_json(metrics_path(paths), type(metrics)) == metrics


def test_project_summary_reports_jobs_failures_queue_coverage_and_artifacts(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        title="Test Match",
        fps=30,
        duration_ms=1000,
    )
    paths = create_project_layout(tmp_path, project, video)
    job = create_job(paths, video_id="video-1", job_type=JobType.METADATA)
    running = transition_job(job, JobStatus.RUNNING)
    failed = transition_job(
        running,
        JobStatus.FAILED,
        error_type="MetadataError",
        error_message="bad metadata",
        user_action="retry",
    )
    append_job_state(paths, running)
    append_job_state(paths, failed)
    write_review_queue(
        paths,
        [
            ReviewQueueItem(
                reason=QueueReason.HIT_CANDIDATE,
                target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
                status=QueueStatus.OPEN,
            ),
            ReviewQueueItem(
                reason=QueueReason.COVERAGE_GAP,
                target_ref=TargetRef(type="coverage_span", id="0-1000"),
                status=QueueStatus.STALE,
            ),
        ],
    )
    write_jsonl(
        paths.state_dir / "coverage.jsonl",
        [
            CoverageSpan(start_ms=0, end_ms=500, state=CoverageState.REVIEWED, source_event_id="cov-1"),
            CoverageSpan(start_ms=500, end_ms=1000, state=CoverageState.NEEDS_RECHECK, source_event_id="cov-2"),
        ],
    )
    artifact = paths.jobs_dir / "job-1" / "output.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")

    summary = build_project_summary(paths)

    assert summary.project_id == "project-1"
    assert summary.video_title == "Test Match"
    assert summary.job_count == 3
    assert len(summary.failed_jobs) == 1
    assert summary.failed_jobs[0].error_type == "MetadataError"
    assert summary.artifact_count == 2
    assert summary.queue_counts.open == 1
    assert summary.queue_counts.stale == 1
    assert summary.coverage.reviewed_duration_ms == 500
    assert summary.coverage.needs_recheck_duration_ms == 500


def test_rebuild_project_summary_writes_json(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    summary = rebuild_project_summary(paths)

    assert project_summary_path(paths).exists()
    assert read_json(project_summary_path(paths), type(summary)) == summary

