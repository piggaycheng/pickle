from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .schema import Annotation, AnnotationEvent, AnnotationEventType
from .storage import ProjectPaths, append_jsonl, read_jsonl, write_jsonl


class AnnotationProjectionError(ValueError):
    pass


def annotation_events_path(paths: ProjectPaths):
    return paths.state_dir / "annotation_events.jsonl"


def annotations_path(paths: ProjectPaths):
    return paths.state_dir / "annotations.jsonl"


def load_annotation_events(paths: ProjectPaths) -> list[AnnotationEvent]:
    return read_jsonl(annotation_events_path(paths), AnnotationEvent)


def load_annotations(paths: ProjectPaths) -> list[Annotation]:
    return read_jsonl(annotations_path(paths), Annotation)


def append_annotation_event(paths: ProjectPaths, event: AnnotationEvent) -> None:
    append_jsonl(annotation_events_path(paths), event)


def create_annotation_event(annotation: Annotation, reason: str | None = None) -> AnnotationEvent:
    return AnnotationEvent(
        annotation_id=annotation.annotation_id,
        event_type=AnnotationEventType.CREATED,
        after=annotation.model_dump(mode="json"),
        reason=reason,
    )


def update_annotation_event(
    current: Annotation,
    changes: Mapping[str, Any],
    *,
    event_type: AnnotationEventType = AnnotationEventType.UPDATED,
    reason: str | None = None,
) -> AnnotationEvent:
    if event_type not in (
        AnnotationEventType.UPDATED,
        AnnotationEventType.ACCEPTED_SUGGESTION,
        AnnotationEventType.CORRECTED_SUGGESTION,
    ):
        raise ValueError(f"unsupported annotation update event type: {event_type}")
    return AnnotationEvent(
        annotation_id=current.annotation_id,
        event_type=event_type,
        before=current.model_dump(mode="json"),
        after=dict(changes),
        reason=reason,
    )


def delete_annotation_event(current: Annotation, reason: str | None = None) -> AnnotationEvent:
    return AnnotationEvent(
        annotation_id=current.annotation_id,
        event_type=AnnotationEventType.DELETED,
        before=current.model_dump(mode="json"),
        reason=reason,
    )


def project_annotations(events: Iterable[AnnotationEvent]) -> list[Annotation]:
    current: dict[str, Annotation] = {}
    for event in events:
        if event.event_type == AnnotationEventType.DELETED:
            if event.annotation_id not in current:
                raise AnnotationProjectionError(f"cannot delete unknown annotation: {event.annotation_id}")
            del current[event.annotation_id]
            continue

        if event.after is None:
            raise AnnotationProjectionError(f"event requires after payload: {event.event_id}")

        if event.event_type == AnnotationEventType.CREATED:
            if event.annotation_id in current:
                raise AnnotationProjectionError(f"annotation already exists: {event.annotation_id}")
            current[event.annotation_id] = Annotation.model_validate(event.after)
            continue

        existing = current.get(event.annotation_id)
        if existing is None:
            raise AnnotationProjectionError(f"cannot update unknown annotation: {event.annotation_id}")
        merged = existing.model_dump(mode="json")
        merged.update(event.after)
        merged["annotation_id"] = event.annotation_id
        current[event.annotation_id] = Annotation.model_validate(merged)

    return sorted(current.values(), key=lambda item: (item.event_time_ms, item.annotation_id))


def rebuild_annotations(paths: ProjectPaths) -> list[Annotation]:
    annotations = project_annotations(load_annotation_events(paths))
    write_jsonl(annotations_path(paths), annotations)
    return annotations

