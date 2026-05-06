from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

from .annotations import load_annotations
from .clips import (
    CLIPS_DIR_REF,
    CommandRunner,
    ClipExtractionError,
    clip_ref_for_annotation,
    extract_clips_job,
)
from .schema import (
    Annotation,
    ExportManifest,
    JobStatus,
    Project,
    TimelineExportRow,
    TrainingExample,
    Video,
    new_id,
    stable_id,
)
from .storage import ProjectPaths, read_json, read_jsonl, safe_join, write_json, write_jsonl

LATEST_EXPORT_DIR_REF = "exports/latest"
EXPORT_RUNS_DIR_REF = "exports/runs"
TIMELINE_REF = f"{LATEST_EXPORT_DIR_REF}/timeline.csv"
TRAINING_EXAMPLES_REF = f"{LATEST_EXPORT_DIR_REF}/training_examples.jsonl"
EXPORT_MANIFEST_REF = f"{LATEST_EXPORT_DIR_REF}/export_manifest.json"
LEGACY_TIMELINE_REF = "exports/timeline.csv"
LEGACY_TRAINING_EXAMPLES_REF = "exports/training_examples.jsonl"
LEGACY_EXPORT_MANIFEST_REF = "exports/export_manifest.json"
EXPORT_OUTPUT_REFS = (TIMELINE_REF, TRAINING_EXAMPLES_REF, EXPORT_MANIFEST_REF)


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


def build_training_examples(
    annotations: list[Annotation],
    *,
    video_ref: str,
    clip_refs: dict[str, str] | None = None,
) -> list[TrainingExample]:
    clip_refs = clip_refs or {}
    return [
        TrainingExample(
            example_id=stable_id("training_example", annotation.annotation_id),
            annotation_id=annotation.annotation_id,
            video_id=annotation.video_id,
            video_ref=video_ref,
            clip_ref=clip_refs.get(annotation.annotation_id),
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


def export_dataset(
    paths: ProjectPaths,
    annotations: list[Annotation] | None = None,
    *,
    extract_clips: bool = False,
    ffmpeg_path: str = "ffmpeg",
    run_command: CommandRunner = subprocess.run,
) -> ExportManifest:
    project = read_json(paths.project_json, Project)
    video = read_json(paths.video_json, Video)
    annotation_list = annotations if annotations is not None else load_annotations(paths)
    trusted_annotations = _trusted_annotations(annotation_list)
    export_id = new_id()
    run_dir_ref = export_run_dir_ref(export_id)
    run_timeline_ref = export_timeline_ref(run_dir_ref)
    run_training_examples_ref = export_training_examples_ref(run_dir_ref)
    run_manifest_ref = export_manifest_ref(run_dir_ref)
    run_clips_dir_ref = export_clips_dir_ref(run_dir_ref)
    latest_clips_dir_ref = export_clips_dir_ref(LATEST_EXPORT_DIR_REF)
    clip_extraction_job_id: str | None = None
    clip_refs: dict[str, str] = {}
    if extract_clips and trusted_annotations:
        clip_job = extract_clips_job(
            paths,
            video_id=video.video_id,
            source_video_ref=video.local_path,
            annotations=trusted_annotations,
            clips_dir_ref=run_clips_dir_ref,
            ffmpeg_path=ffmpeg_path,
            run_command=run_command,
        )
        clip_extraction_job_id = clip_job.job_id
        if clip_job.status != JobStatus.SUCCESS:
            _remove_ref(paths, run_dir_ref)
            raise ClipExtractionError(clip_job.error_message or "clip extraction failed")
        clip_refs = {
            annotation.annotation_id: clip_ref_for_annotation(annotation, clips_dir_ref=run_clips_dir_ref)
            for annotation in trusted_annotations
        }
    timeline_rows = build_timeline_rows(annotation_list)
    run_training_examples = build_training_examples(
        annotation_list,
        video_ref=video.local_path,
        clip_refs=clip_refs,
    )

    run_timeline_path = safe_join(paths.root, *run_timeline_ref.split("/"))
    run_training_examples_path = safe_join(paths.root, *run_training_examples_ref.split("/"))
    run_manifest_path = safe_join(paths.root, *run_manifest_ref.split("/"))

    write_timeline_csv(run_timeline_path, timeline_rows)
    write_jsonl(run_training_examples_path, run_training_examples)
    run_manifest = ExportManifest(
        export_id=export_id,
        project_id=project.project_id,
        video_id=video.video_id,
        timeline_ref=run_timeline_ref,
        training_examples_ref=run_training_examples_ref,
        clips_dir_ref=run_clips_dir_ref if clip_refs else None,
        clip_extraction_job_id=clip_extraction_job_id,
        clip_count=len(clip_refs),
        annotation_count=len(annotation_list),
        training_example_count=len(run_training_examples),
    )
    write_json(run_manifest_path, run_manifest)

    _clear_latest_export_outputs(paths)
    latest_clip_refs = {
        annotation.annotation_id: clip_ref_for_annotation(annotation, clips_dir_ref=latest_clips_dir_ref)
        for annotation in trusted_annotations
    } if clip_refs else {}
    latest_training_examples = build_training_examples(
        annotation_list,
        video_ref=video.local_path,
        clip_refs=latest_clip_refs,
    )
    write_timeline_csv(safe_join(paths.root, *TIMELINE_REF.split("/")), timeline_rows)
    write_jsonl(safe_join(paths.root, *TRAINING_EXAMPLES_REF.split("/")), latest_training_examples)
    if clip_refs:
        shutil.copytree(
            safe_join(paths.root, *run_clips_dir_ref.split("/")),
            safe_join(paths.root, *latest_clips_dir_ref.split("/")),
        )
    latest_manifest = ExportManifest(
        export_id=export_id,
        project_id=project.project_id,
        video_id=video.video_id,
        timeline_ref=TIMELINE_REF,
        training_examples_ref=TRAINING_EXAMPLES_REF,
        clips_dir_ref=latest_clips_dir_ref if clip_refs else None,
        clip_extraction_job_id=clip_extraction_job_id,
        clip_count=len(clip_refs),
        annotation_count=len(annotation_list),
        training_example_count=len(latest_training_examples),
        generated_at=run_manifest.generated_at,
    )
    write_json(safe_join(paths.root, *EXPORT_MANIFEST_REF.split("/")), latest_manifest)
    return latest_manifest


def export_run_dir_ref(export_id: str) -> str:
    return f"{EXPORT_RUNS_DIR_REF}/{export_id}"


def export_timeline_ref(export_dir_ref: str) -> str:
    return f"{export_dir_ref}/timeline.csv"


def export_training_examples_ref(export_dir_ref: str) -> str:
    return f"{export_dir_ref}/training_examples.jsonl"


def export_manifest_ref(export_dir_ref: str) -> str:
    return f"{export_dir_ref}/export_manifest.json"


def export_clips_dir_ref(export_dir_ref: str) -> str:
    return f"{export_dir_ref}/clips"


def latest_or_legacy_export_manifest_ref(paths: ProjectPaths) -> str | None:
    for ref in (EXPORT_MANIFEST_REF, LEGACY_EXPORT_MANIFEST_REF):
        if safe_join(paths.root, *ref.split("/")).exists():
            return ref
    return None


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


def clear_export_outputs(paths: ProjectPaths) -> list[str]:
    removed_refs: list[str] = []
    for ref in (*EXPORT_OUTPUT_REFS, LEGACY_TIMELINE_REF, LEGACY_TRAINING_EXAMPLES_REF, LEGACY_EXPORT_MANIFEST_REF):
        if _remove_ref(paths, ref):
            removed_refs.append(ref)

    for clips_ref in (export_clips_dir_ref(LATEST_EXPORT_DIR_REF), CLIPS_DIR_REF):
        if _remove_ref(paths, clips_ref):
            removed_refs.append(clips_ref)
    return removed_refs


def export_staleness_warnings(paths: ProjectPaths, annotations: list[Annotation]) -> list[str]:
    manifest_ref = latest_or_legacy_export_manifest_ref(paths)
    if manifest_ref is None:
        return []

    manifest_path = safe_join(paths.root, *manifest_ref.split("/"))
    manifest = read_json(manifest_path, ExportManifest)
    examples_path = safe_join(paths.root, *manifest.training_examples_ref.split("/"))
    if not examples_path.exists():
        return [f"Export manifest exists, but {manifest.training_examples_ref} is missing."]

    current = {
        annotation.annotation_id: _annotation_export_snapshot(annotation)
        for annotation in _trusted_annotations(annotations)
    }
    exported_examples = read_jsonl(examples_path, TrainingExample)
    exported = {
        example.annotation_id: _training_example_snapshot(example)
        for example in exported_examples
    }

    warnings: list[str] = []
    missing_from_export = sorted(set(current) - set(exported))
    if missing_from_export:
        warnings.append(
            f"Export is stale: {len(missing_from_export)} trusted annotation(s) are not exported."
        )

    removed_from_current = sorted(set(exported) - set(current))
    if removed_from_current:
        warnings.append(
            f"Export is stale: {len(removed_from_current)} exported example(s) no longer have trusted annotations."
        )

    changed = sorted(
        annotation_id
        for annotation_id in set(current) & set(exported)
        if current[annotation_id] != exported[annotation_id]
    )
    if changed:
        warnings.append(
            f"Export is stale: {len(changed)} exported example(s) differ from current annotations."
        )
    return warnings


def _clear_latest_export_outputs(paths: ProjectPaths) -> None:
    for ref in EXPORT_OUTPUT_REFS:
        _remove_ref(paths, ref)
    _remove_ref(paths, export_clips_dir_ref(LATEST_EXPORT_DIR_REF))


def _remove_ref(paths: ProjectPaths, ref: str) -> bool:
    path = safe_join(paths.root, *ref.split("/"))
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _annotation_export_snapshot(annotation: Annotation) -> dict[str, object]:
    return {
        "annotation_id": annotation.annotation_id,
        "video_id": annotation.video_id,
        "player_id": annotation.player_id,
        "action": annotation.action,
        "event_time_ms": annotation.event_time_ms,
        "clip_start_ms": annotation.clip_start_ms,
        "clip_end_ms": annotation.clip_end_ms,
        "timing_confidence": annotation.timing_confidence,
        "bbox": annotation.bbox,
        "pose_landmarks_ref": annotation.pose_landmarks_ref,
        "source_tool": annotation.source_tool,
        "external_refs": annotation.external_refs,
    }


def _training_example_snapshot(example: TrainingExample) -> dict[str, object]:
    return {
        "annotation_id": example.annotation_id,
        "video_id": example.video_id,
        "player_id": example.player_id,
        "action": example.action,
        "event_time_ms": example.event_time_ms,
        "clip_start_ms": example.clip_start_ms,
        "clip_end_ms": example.clip_end_ms,
        "timing_confidence": example.timing_confidence,
        "bbox": example.bbox,
        "pose_landmarks_ref": example.pose_landmarks_ref,
        "source_tool": example.source_tool,
        "external_refs": example.external_refs,
    }


def _trusted_annotations(annotations: list[Annotation]) -> list[Annotation]:
    return sorted(
        [annotation for annotation in annotations if annotation.source != "model_suggestion"],
        key=lambda item: (item.event_time_ms, item.annotation_id),
    )
