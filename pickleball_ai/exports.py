from __future__ import annotations

import csv
import os
from pathlib import Path

from .annotations import load_annotations
from .schema import (
    Annotation,
    ExportManifest,
    Project,
    TimelineExportRow,
    TrainingExample,
    Video,
    stable_id,
)
from .storage import ProjectPaths, read_json, safe_join, write_json, write_jsonl

TIMELINE_REF = "exports/timeline.csv"
TRAINING_EXAMPLES_REF = "exports/training_examples.jsonl"
EXPORT_MANIFEST_REF = "exports/export_manifest.json"


def build_timeline_rows(annotations: list[Annotation]) -> list[TimelineExportRow]:
    return [
        TimelineExportRow(
            annotation_id=annotation.annotation_id,
            video_id=annotation.video_id,
            timestamp_ms=annotation.event_time_ms,
            player_id=annotation.player_id,
            action=annotation.action,
            timing_confidence=annotation.timing_confidence,
            clip_start_ms=annotation.clip_start_ms,
            clip_end_ms=annotation.clip_end_ms,
            source=annotation.source,
        )
        for annotation in _trusted_annotations(annotations)
    ]


def build_training_examples(annotations: list[Annotation], *, video_ref: str) -> list[TrainingExample]:
    return [
        TrainingExample(
            example_id=stable_id("training_example", annotation.annotation_id),
            annotation_id=annotation.annotation_id,
            video_id=annotation.video_id,
            video_ref=video_ref,
            player_id=annotation.player_id,
            action=annotation.action,
            event_time_ms=annotation.event_time_ms,
            clip_start_ms=annotation.clip_start_ms,
            clip_end_ms=annotation.clip_end_ms,
            timing_confidence=annotation.timing_confidence,
            bbox=annotation.bbox,
            pose_landmarks_ref=annotation.pose_landmarks_ref,
            source_tool=annotation.source_tool,
            external_refs=annotation.external_refs,
        )
        for annotation in _trusted_annotations(annotations)
    ]


def export_dataset(paths: ProjectPaths, annotations: list[Annotation] | None = None) -> ExportManifest:
    project = read_json(paths.project_json, Project)
    video = read_json(paths.video_json, Video)
    annotation_list = annotations if annotations is not None else load_annotations(paths)
    timeline_rows = build_timeline_rows(annotation_list)
    training_examples = build_training_examples(annotation_list, video_ref=video.local_path)

    timeline_path = safe_join(paths.root, *TIMELINE_REF.split("/"))
    training_examples_path = safe_join(paths.root, *TRAINING_EXAMPLES_REF.split("/"))
    manifest_path = safe_join(paths.root, *EXPORT_MANIFEST_REF.split("/"))

    write_timeline_csv(timeline_path, timeline_rows)
    write_jsonl(training_examples_path, training_examples)
    manifest = ExportManifest(
        project_id=project.project_id,
        video_id=video.video_id,
        timeline_ref=TIMELINE_REF,
        training_examples_ref=TRAINING_EXAMPLES_REF,
        annotation_count=len(annotation_list),
        training_example_count=len(training_examples),
    )
    write_json(manifest_path, manifest)
    return manifest


def write_timeline_csv(path: Path, rows: list[TimelineExportRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    fieldnames = [
        "annotation_id",
        "video_id",
        "timestamp_ms",
        "player_id",
        "action",
        "timing_confidence",
        "clip_start_ms",
        "clip_end_ms",
        "source",
    ]
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.model_dump(mode="json")
            writer.writerow(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _trusted_annotations(annotations: list[Annotation]) -> list[Annotation]:
    return sorted(
        [annotation for annotation in annotations if annotation.source != "model_suggestion"],
        key=lambda item: (item.event_time_ms, item.annotation_id),
    )

