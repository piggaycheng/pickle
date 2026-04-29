from __future__ import annotations

from collections import Counter

from .coverage import load_coverage
from .jobs import load_job_history
from .queue import load_review_queue
from .schema import (
    CoverageState,
    CoverageStatusSummary,
    JobFailureSummary,
    JobStatus,
    Project,
    ProjectSummary,
    QueueStatus,
    QueueStatusCounts,
    Video,
)
from .storage import ProjectPaths, read_json, write_json


def project_summary_path(paths: ProjectPaths):
    return paths.state_dir / "project_summary.json"


def build_project_summary(paths: ProjectPaths) -> ProjectSummary:
    project = read_json(paths.project_json, Project)
    video = read_json(paths.video_json, Video) if paths.video_json.exists() else None
    jobs = load_job_history(paths.processing_jobs_jsonl)
    queue_items = load_review_queue(paths)
    coverage = load_coverage(paths)

    return ProjectSummary(
        project_id=project.project_id,
        video_id=video.video_id if video is not None else project.video_id,
        video_title=video.title if video is not None else None,
        job_count=len(jobs),
        failed_jobs=[
            JobFailureSummary(
                job_id=job.job_id,
                job_type=job.job_type,
                error_type=job.error_type,
                error_message=job.error_message,
                user_action=job.user_action,
            )
            for job in jobs
            if job.status == JobStatus.FAILED
        ],
        artifact_count=_artifact_count(paths),
        queue_counts=_queue_counts(queue_items),
        coverage=CoverageStatusSummary(
            reviewed_duration_ms=_coverage_duration(coverage, CoverageState.REVIEWED),
            skipped_non_game_duration_ms=_coverage_duration(coverage, CoverageState.SKIPPED_NON_GAME),
            unreviewed_duration_ms=_coverage_duration(coverage, CoverageState.UNREVIEWED),
            needs_recheck_duration_ms=_coverage_duration(coverage, CoverageState.NEEDS_RECHECK),
        ),
    )


def write_project_summary(paths: ProjectPaths, summary: ProjectSummary) -> None:
    write_json(project_summary_path(paths), summary)


def rebuild_project_summary(paths: ProjectPaths) -> ProjectSummary:
    summary = build_project_summary(paths)
    write_project_summary(paths, summary)
    return summary


def _artifact_count(paths: ProjectPaths) -> int:
    if not paths.artifacts_dir.exists():
        return 0
    return sum(1 for path in paths.artifacts_dir.rglob("*") if path.is_file())


def _queue_counts(queue_items) -> QueueStatusCounts:
    counts = Counter(item.status for item in queue_items)
    return QueueStatusCounts(
        open=counts[QueueStatus.OPEN],
        accepted=counts[QueueStatus.ACCEPTED],
        corrected=counts[QueueStatus.CORRECTED],
        dismissed=counts[QueueStatus.DISMISSED],
        stale=counts[QueueStatus.STALE],
    )


def _coverage_duration(coverage, state: CoverageState) -> int:
    return sum(span.end_ms - span.start_ms for span in coverage if span.state == state)

