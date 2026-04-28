from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from pickleball_ai.pose import extract_pose_job
from pickleball_ai.schema import Project, SourceType, Video
from pickleball_ai.storage import create_project_layout, get_data_root


class VideoMetadataError(Exception):
    pass


def read_video_metadata(input_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(input_path))
    try:
        if not cap.isOpened():
            raise VideoMetadataError(f"Could not open video file: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int(1000 * frame_count / fps) if frame_count > 0 else 0
        return fps, duration_ms
    finally:
        cap.release()


def process_video(input_path: str, project_id: str | None = None) -> Path:
    source = Path(input_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    fps, duration_ms = read_video_metadata(source)
    project = Project(project_id=project_id) if project_id else Project()
    video = Video(
        video_id=project.video_id,
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        title=source.name,
        fps=fps,
        duration_ms=duration_ms,
    )
    paths = create_project_layout(get_data_root(), project, video)
    target = paths.videos_dir / "source.mp4"
    shutil.copy2(source, target)

    job = extract_pose_job(paths, video_id=project.video_id)
    if not job.output_refs:
        raise RuntimeError(f"Pose extraction failed: {job.error_type}: {job.error_message}")
    return paths.root / job.output_refs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose landmarks as a replayable job artifact.")
    parser.add_argument("input", nargs="?", default="videos/input.mp4", help="Input video path")
    parser.add_argument("--project-id", help="Optional project id for datasets/{project_id}")
    args = parser.parse_args()

    output_path = process_video(args.input, project_id=args.project_id)
    print(f"Pose JSONL saved to: {output_path}")


if __name__ == "__main__":
    main()
