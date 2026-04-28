import pytest
from pydantic import ValidationError

from pickleball_ai.annotations import (
    AnnotationProjectionError,
    create_annotation_event,
    delete_annotation_event,
    project_annotations,
    rebuild_annotations,
    update_annotation_event,
)
from pickleball_ai.schema import Annotation, AnnotationEventType, TimeWindow, TimingConfidence
from pickleball_ai.storage import create_project_layout, read_jsonl
from pickleball_ai.annotations import annotation_events_path, annotations_path, append_annotation_event
from pickleball_ai.schema import Project


def make_annotation(annotation_id: str = "ann-1", event_time_ms: int = 1000) -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        video_id="video-1",
        event_time_ms=event_time_ms,
        time_window=TimeWindow(start_ms=900, end_ms=1100),
        timing_confidence=TimingConfidence.EXACT,
        player_id="A",
        action="drive",
        confidence=1,
        source="manual",
        clip_start_ms=700,
        clip_end_ms=1300,
    )


def test_annotation_projection_applies_updates_and_deletes():
    first = make_annotation()
    created = create_annotation_event(first)
    updated = update_annotation_event(first, {"action": "slice", "confidence": 0.9})
    projected = project_annotations([created, updated])

    assert len(projected) == 1
    assert projected[0].action == "slice"
    assert projected[0].confidence == 0.9

    deleted = delete_annotation_event(projected[0], reason="wrong event")
    assert project_annotations([created, updated, deleted]) == []


def test_annotation_projection_rejects_update_before_create():
    annotation = make_annotation()
    event = update_annotation_event(annotation, {"action": "lob"})

    with pytest.raises(AnnotationProjectionError):
        project_annotations([event])


def test_annotation_event_rejects_mismatched_after_id():
    annotation = make_annotation()

    with pytest.raises(ValidationError):
        update_annotation_event(annotation, {"annotation_id": "other", "action": "lob"})


def test_rebuild_annotations_writes_materialized_state(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    first = make_annotation(annotation_id="ann-1", event_time_ms=2000)
    second = make_annotation(annotation_id="ann-2", event_time_ms=1000)

    append_annotation_event(paths, create_annotation_event(first))
    append_annotation_event(paths, create_annotation_event(second))

    rebuilt = rebuild_annotations(paths)

    assert [item.annotation_id for item in rebuilt] == ["ann-2", "ann-1"]
    assert annotation_events_path(paths).exists()
    assert read_jsonl(annotations_path(paths), Annotation) == rebuilt

