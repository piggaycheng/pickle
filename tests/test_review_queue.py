from pickleball_ai.queue import (
    load_review_queue,
    materialize_review_queue,
    rebuild_review_queue,
    review_queue_path,
)
from pickleball_ai.schema import (
    CoverageSpan,
    CoverageState,
    HitCandidate,
    IdentityConfirmedBy,
    IdentitySession,
    Project,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    TargetRef,
    TimeWindow,
)
from pickleball_ai.storage import create_project_layout


def make_hit_candidate(candidate_id: str = "candidate-1") -> HitCandidate:
    return HitCandidate(
        candidate_id=candidate_id,
        video_id="video-1",
        timestamp_ms=1000,
        time_window=TimeWindow(start_ms=900, end_ms=1100),
        confidence=0.8,
        source_job_id="event-job-1",
        pose_frame_index=10,
    )


def make_coverage_span(state: CoverageState = CoverageState.UNREVIEWED) -> CoverageSpan:
    return CoverageSpan(
        start_ms=0,
        end_ms=1000,
        state=state,
        source_event_id="coverage-event-1",
    )


def make_identity_session(confidence: float = 0.5, invalidated_by_cut: bool = False) -> IdentitySession:
    return IdentitySession(
        identity_session_id="identity-1",
        player_id="A",
        track_id="track-1",
        start_ms=0,
        end_ms=1000,
        confidence=confidence,
        confirmed_by=IdentityConfirmedBy.MODEL,
        invalidated_by_cut=invalidated_by_cut,
    )


def test_queue_materializes_hit_candidates_coverage_gaps_and_low_identity():
    items = materialize_review_queue(
        hit_candidates=[make_hit_candidate()],
        coverage=[make_coverage_span()],
        identity_sessions=[make_identity_session()],
    )

    assert [item.reason for item in items] == [
        QueueReason.HIT_CANDIDATE,
        QueueReason.LOW_IDENTITY_CONFIDENCE,
        QueueReason.COVERAGE_GAP,
    ]
    assert all(item.status == QueueStatus.OPEN for item in items)
    assert items[0].created_from_job_id == "event-job-1"


def test_queue_rebuild_is_idempotent_for_same_inputs():
    first = materialize_review_queue(hit_candidates=[make_hit_candidate()])
    second = materialize_review_queue(hit_candidates=[make_hit_candidate()])

    assert first == second


def test_queue_preserves_existing_user_status_for_active_item():
    existing = ReviewQueueItem(
        queue_item_id="queue-1",
        reason=QueueReason.HIT_CANDIDATE,
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        status=QueueStatus.ACCEPTED,
        priority=100,
    )

    items = materialize_review_queue(
        hit_candidates=[make_hit_candidate()],
        existing_items=[existing],
    )

    assert len(items) == 1
    assert items[0].queue_item_id == "queue-1"
    assert items[0].status == QueueStatus.ACCEPTED


def test_queue_marks_missing_upstream_items_stale():
    existing = ReviewQueueItem(
        queue_item_id="queue-1",
        reason=QueueReason.HIT_CANDIDATE,
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        status=QueueStatus.OPEN,
        priority=100,
    )

    items = materialize_review_queue(existing_items=[existing])

    assert len(items) == 1
    assert items[0].reason == QueueReason.STALE_ITEM
    assert items[0].status == QueueStatus.STALE
    assert items[0].target_ref == existing.target_ref


def test_queue_does_not_materialize_reviewed_coverage_or_valid_identity():
    items = materialize_review_queue(
        coverage=[make_coverage_span(CoverageState.REVIEWED)],
        identity_sessions=[
            make_identity_session(confidence=0.9),
            make_identity_session(confidence=0.1, invalidated_by_cut=True),
        ],
    )

    assert items == []


def test_rebuild_review_queue_writes_materialized_state(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    rebuilt = rebuild_review_queue(
        paths,
        hit_candidates=[make_hit_candidate()],
        coverage=[make_coverage_span()],
    )

    assert review_queue_path(paths).exists()
    assert load_review_queue(paths) == rebuilt

