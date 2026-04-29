from __future__ import annotations

from collections.abc import Iterable

from .schema import (
    CoverageSpan,
    CoverageState,
    HitCandidate,
    IdentitySession,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    TargetRef,
    stable_id,
)
from .storage import ProjectPaths, read_jsonl, write_jsonl


def review_queue_path(paths: ProjectPaths):
    return paths.state_dir / "review_queue.jsonl"


def load_review_queue(paths: ProjectPaths) -> list[ReviewQueueItem]:
    return read_jsonl(review_queue_path(paths), ReviewQueueItem)


def write_review_queue(paths: ProjectPaths, items: list[ReviewQueueItem]) -> None:
    write_jsonl(review_queue_path(paths), items)


def rebuild_review_queue(
    paths: ProjectPaths,
    *,
    hit_candidates: Iterable[HitCandidate] = (),
    coverage: Iterable[CoverageSpan] = (),
    identity_sessions: Iterable[IdentitySession] = (),
    low_identity_threshold: float = 0.75,
) -> list[ReviewQueueItem]:
    rebuilt = materialize_review_queue(
        hit_candidates=hit_candidates,
        coverage=coverage,
        identity_sessions=identity_sessions,
        existing_items=load_review_queue(paths),
        low_identity_threshold=low_identity_threshold,
    )
    write_review_queue(paths, rebuilt)
    return rebuilt


def materialize_review_queue(
    *,
    hit_candidates: Iterable[HitCandidate] = (),
    coverage: Iterable[CoverageSpan] = (),
    identity_sessions: Iterable[IdentitySession] = (),
    existing_items: Iterable[ReviewQueueItem] = (),
    low_identity_threshold: float = 0.75,
) -> list[ReviewQueueItem]:
    existing_by_key = {_item_key(item): item for item in existing_items}
    active_items = [
        *_hit_candidate_items(hit_candidates, existing_by_key),
        *_coverage_gap_items(coverage, existing_by_key),
        *_low_identity_items(identity_sessions, existing_by_key, low_identity_threshold),
    ]
    active_keys = {_item_key(item) for item in active_items}
    stale_items = [
        _mark_stale(item)
        for key, item in existing_by_key.items()
        if key not in active_keys and item.status != QueueStatus.STALE
    ]
    return sorted(
        [*active_items, *stale_items],
        key=lambda item: (-item.priority, item.reason.value, item.target_ref.type, item.target_ref.id),
    )


def _hit_candidate_items(
    hit_candidates: Iterable[HitCandidate],
    existing_by_key: dict[tuple[str, str, str], ReviewQueueItem],
) -> list[ReviewQueueItem]:
    return [
        _reuse_or_create(
            existing_by_key,
            reason=QueueReason.HIT_CANDIDATE,
            target_ref=TargetRef(type="hit_candidate", id=candidate.candidate_id),
            priority=100,
            created_from_job_id=candidate.source_job_id,
        )
        for candidate in hit_candidates
    ]


def _coverage_gap_items(
    coverage: Iterable[CoverageSpan],
    existing_by_key: dict[tuple[str, str, str], ReviewQueueItem],
) -> list[ReviewQueueItem]:
    return [
        _reuse_or_create(
            existing_by_key,
            reason=QueueReason.COVERAGE_GAP,
            target_ref=TargetRef(type="coverage_span", id=f"{span.start_ms}-{span.end_ms}"),
            priority=60,
        )
        for span in coverage
        if span.state == CoverageState.UNREVIEWED
    ]


def _low_identity_items(
    identity_sessions: Iterable[IdentitySession],
    existing_by_key: dict[tuple[str, str, str], ReviewQueueItem],
    low_identity_threshold: float,
) -> list[ReviewQueueItem]:
    return [
        _reuse_or_create(
            existing_by_key,
            reason=QueueReason.LOW_IDENTITY_CONFIDENCE,
            target_ref=TargetRef(type="identity_session", id=session.identity_session_id),
            priority=80,
        )
        for session in identity_sessions
        if not session.invalidated_by_cut and session.confidence < low_identity_threshold
    ]


def _reuse_or_create(
    existing_by_key: dict[tuple[str, str, str], ReviewQueueItem],
    *,
    reason: QueueReason,
    target_ref: TargetRef,
    priority: int,
    created_from_job_id: str | None = None,
) -> ReviewQueueItem:
    key = (reason.value, target_ref.type, target_ref.id)
    existing = existing_by_key.get(key)
    status = existing.status if existing and existing.status != QueueStatus.STALE else QueueStatus.OPEN
    data = {
        "queue_item_id": existing.queue_item_id if existing else _queue_item_id(reason, target_ref),
        "reason": reason,
        "target_ref": target_ref,
        "status": status,
        "priority": priority,
        "created_from_job_id": (
            created_from_job_id
            if created_from_job_id is not None
            else existing.created_from_job_id
            if existing
            else None
        ),
    }
    if existing:
        data["created_at"] = existing.created_at
    return ReviewQueueItem(**data)


def _mark_stale(item: ReviewQueueItem) -> ReviewQueueItem:
    return ReviewQueueItem(
        queue_item_id=item.queue_item_id,
        reason=QueueReason.STALE_ITEM,
        target_ref=item.target_ref,
        status=QueueStatus.STALE,
        priority=0,
        created_from_job_id=item.created_from_job_id,
        stale_reason=f"upstream item no longer produced {item.reason.value}",
        created_at=item.created_at,
    )


def _queue_item_id(reason: QueueReason, target_ref: TargetRef) -> str:
    return stable_id("review_queue", f"{reason.value}:{target_ref.type}:{target_ref.id}")


def _item_key(item: ReviewQueueItem) -> tuple[str, str, str]:
    reason = item.reason
    if reason == QueueReason.STALE_ITEM:
        reason = _original_reason_from_stale(item)
    return (reason.value, item.target_ref.type, item.target_ref.id)


def _original_reason_from_stale(item: ReviewQueueItem) -> QueueReason:
    if item.target_ref.type == "hit_candidate":
        return QueueReason.HIT_CANDIDATE
    if item.target_ref.type == "coverage_span":
        return QueueReason.COVERAGE_GAP
    if item.target_ref.type == "identity_session":
        return QueueReason.LOW_IDENTITY_CONFIDENCE
    return QueueReason.STALE_ITEM
