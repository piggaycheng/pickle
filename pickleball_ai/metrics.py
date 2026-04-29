from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .schema import (
    Annotation,
    AnnotationEvent,
    CoverageSpan,
    CoverageState,
    ProjectMetrics,
)
from .storage import ProjectPaths, write_json


def metrics_path(paths: ProjectPaths):
    return paths.state_dir / "metrics.json"


def compute_metrics(
    *,
    annotations: Iterable[Annotation] = (),
    coverage: Iterable[CoverageSpan] = (),
    annotation_events: Iterable[AnnotationEvent] = (),
) -> ProjectMetrics:
    annotation_list = list(annotations)
    coverage_list = list(coverage)
    reviewed_duration_ms = _duration_for_states(coverage_list, {CoverageState.REVIEWED})
    skipped_non_game_duration_ms = _duration_for_states(coverage_list, {CoverageState.SKIPPED_NON_GAME})
    unreviewed_duration_ms = _duration_for_states(coverage_list, {CoverageState.UNREVIEWED})
    annotations_per_reviewed_minute = None
    if reviewed_duration_ms > 0:
        reviewed_minutes = reviewed_duration_ms / 60_000
        annotations_per_reviewed_minute = len(annotation_list) / reviewed_minutes

    identity_corrections_count = 0
    action_corrections: Counter[str] = Counter()
    for event in annotation_events:
        before = event.before or {}
        after = event.after or {}
        before_player = before.get("player_id")
        after_player = after.get("player_id")
        if before_player is not None and after_player is not None and before_player != after_player:
            identity_corrections_count += 1
        before_action = before.get("action")
        after_action = after.get("action")
        if before_action is not None and after_action is not None and before_action != after_action:
            action_corrections[str(after_action)] += 1

    return ProjectMetrics(
        reviewed_duration_ms=reviewed_duration_ms,
        skipped_non_game_duration_ms=skipped_non_game_duration_ms,
        unreviewed_duration_ms=unreviewed_duration_ms,
        annotation_count=len(annotation_list),
        annotations_per_reviewed_minute=annotations_per_reviewed_minute,
        identity_corrections_count=identity_corrections_count,
        action_corrections_by_label=dict(sorted(action_corrections.items())),
    )


def write_metrics(paths: ProjectPaths, metrics: ProjectMetrics) -> None:
    write_json(metrics_path(paths), metrics)


def rebuild_metrics(
    paths: ProjectPaths,
    *,
    annotations: Iterable[Annotation] = (),
    coverage: Iterable[CoverageSpan] = (),
    annotation_events: Iterable[AnnotationEvent] = (),
) -> ProjectMetrics:
    metrics = compute_metrics(
        annotations=annotations,
        coverage=coverage,
        annotation_events=annotation_events,
    )
    write_metrics(paths, metrics)
    return metrics


def _duration_for_states(coverage: Iterable[CoverageSpan], states: set[CoverageState]) -> int:
    return sum(span.end_ms - span.start_ms for span in coverage if span.state in states)

