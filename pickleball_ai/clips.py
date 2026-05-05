from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from .jobs import append_job_state, create_job, transition_job
from .schema import Annotation, JobStatus, JobType, ProcessingJob
from .storage import ProjectPaths, safe_join

CLIPS_DIR_REF = "exports/clips"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ClipExtractionError(RuntimeError):
    pass


def clip_ref_for_annotation(annotation: Annotation) -> str:
    return f"{CLIPS_DIR_REF}/{annotation.annotation_id}.mp4"


def build_ffmpeg_clip_command(
    *,
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
) -> list[str]:
    if end_ms <= start_ms:
        raise ClipExtractionError("clip end_ms must be greater than start_ms")
    duration_seconds = (end_ms - start_ms) / 1000
    return [
        ffmpeg_path,
        "-y",
        "-ss",
        _seconds(start_ms),
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]


def extract_clip(
    paths: ProjectPaths,
    *,
    annotation: Annotation,
    source_video_ref: str,
    ffmpeg_path: str = "ffmpeg",
    run_command: CommandRunner = subprocess.run,
) -> str:
    source_path = safe_join(paths.root, *source_video_ref.split("/"))
    if not source_path.exists():
        raise ClipExtractionError(f"source video not found: {source_video_ref}")

    clip_ref = clip_ref_for_annotation(annotation)
    output_path = safe_join(paths.root, *clip_ref.split("/"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_clip_command(
        ffmpeg_path=ffmpeg_path,
        source_path=source_path,
        output_path=output_path,
        start_ms=annotation.clip_start_ms,
        end_ms=annotation.clip_end_ms,
    )
    try:
        result = run_command(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ClipExtractionError(f"ffmpeg executable not found: {ffmpeg_path}") from exc

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "ffmpeg exited with no output").strip()
        raise ClipExtractionError(f"ffmpeg failed for {annotation.annotation_id}: {details}")
    if not output_path.exists():
        raise ClipExtractionError(f"ffmpeg did not create clip: {clip_ref}")
    return clip_ref


def extract_clips(
    paths: ProjectPaths,
    *,
    annotations: Iterable[Annotation],
    source_video_ref: str,
    ffmpeg_path: str = "ffmpeg",
    run_command: CommandRunner = subprocess.run,
) -> list[str]:
    return [
        extract_clip(
            paths,
            annotation=annotation,
            source_video_ref=source_video_ref,
            ffmpeg_path=ffmpeg_path,
            run_command=run_command,
        )
        for annotation in annotations
    ]


def extract_clips_job(
    paths: ProjectPaths,
    *,
    video_id: str,
    source_video_ref: str,
    annotations: Iterable[Annotation],
    ffmpeg_path: str = "ffmpeg",
    run_command: CommandRunner = subprocess.run,
) -> ProcessingJob:
    annotation_list = list(annotations)
    job = create_job(
        paths,
        video_id=video_id,
        job_type=JobType.CLIP_EXTRACTION,
        input_refs=[source_video_ref],
        params={
            "annotation_count": len(annotation_list),
            "clips_dir_ref": CLIPS_DIR_REF,
            "ffmpeg_path": ffmpeg_path,
        },
    )
    running = transition_job(job, JobStatus.RUNNING)
    append_job_state(paths, running)

    try:
        clip_refs = extract_clips(
            paths,
            annotations=annotation_list,
            source_video_ref=source_video_ref,
            ffmpeg_path=ffmpeg_path,
            run_command=run_command,
        )
        success = transition_job(running, JobStatus.SUCCESS, output_refs=clip_refs)
        append_job_state(paths, success)
        return success
    except Exception as exc:
        failed = transition_job(
            running,
            JobStatus.FAILED,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            user_action="Install ffmpeg, verify the source video path, then rerun clip extraction.",
        )
        append_job_state(paths, failed)
        return failed


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"
