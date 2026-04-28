from __future__ import annotations

from collections.abc import Iterable

from .schema import Annotation, CoverageEvent, CoverageSpan, CoverageState
from .storage import ProjectPaths, append_jsonl, read_jsonl, write_jsonl


def coverage_events_path(paths: ProjectPaths):
    return paths.state_dir / "coverage_events.jsonl"


def coverage_path(paths: ProjectPaths):
    return paths.state_dir / "coverage.jsonl"


def load_coverage_events(paths: ProjectPaths) -> list[CoverageEvent]:
    return read_jsonl(coverage_events_path(paths), CoverageEvent)


def load_coverage(paths: ProjectPaths) -> list[CoverageSpan]:
    return read_jsonl(coverage_path(paths), CoverageSpan)


def append_coverage_event(paths: ProjectPaths, event: CoverageEvent) -> None:
    append_jsonl(coverage_events_path(paths), event)


def project_coverage(events: Iterable[CoverageEvent]) -> list[CoverageSpan]:
    ordered = list(events)
    boundaries = sorted({boundary for event in ordered for boundary in (event.start_ms, event.end_ms)})
    spans: list[CoverageSpan] = []

    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        if start_ms == end_ms:
            continue
        winner = next(
            (
                event
                for event in reversed(ordered)
                if event.start_ms <= start_ms and end_ms <= event.end_ms
            ),
            None,
        )
        if winner is None:
            continue
        candidate = CoverageSpan(
            start_ms=start_ms,
            end_ms=end_ms,
            state=winner.state,
            source_event_id=winner.coverage_event_id,
            segment_id=winner.segment_id,
            reason=winner.reason,
        )
        if spans and _can_merge(spans[-1], candidate):
            previous = spans[-1]
            spans[-1] = CoverageSpan(
                start_ms=previous.start_ms,
                end_ms=candidate.end_ms,
                state=previous.state,
                source_event_id=previous.source_event_id,
                segment_id=previous.segment_id,
                reason=previous.reason,
            )
        else:
            spans.append(candidate)

    return spans


def rebuild_coverage(paths: ProjectPaths) -> list[CoverageSpan]:
    coverage = project_coverage(load_coverage_events(paths))
    write_jsonl(coverage_path(paths), coverage)
    return coverage


def annotation_is_allowed(annotation: Annotation, coverage: Iterable[CoverageSpan], *, draft: bool = False) -> bool:
    if draft:
        return True
    allowed_states = {CoverageState.REVIEWED, CoverageState.NEEDS_RECHECK}
    for span in coverage:
        if span.state not in allowed_states:
            continue
        if span.start_ms <= annotation.time_window.start_ms and annotation.time_window.end_ms <= span.end_ms:
            return True
    return False


def _can_merge(left: CoverageSpan, right: CoverageSpan) -> bool:
    return (
        left.end_ms == right.start_ms
        and left.state == right.state
        and left.source_event_id == right.source_event_id
        and left.segment_id == right.segment_id
        and left.reason == right.reason
    )

