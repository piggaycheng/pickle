from __future__ import annotations

import urllib.request
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .jobs import append_job_state, create_job, transition_job
from .schema import JobStatus, JobType, Landmark, PoseFrame, ProcessingJob
from .storage import ProjectPaths, append_jsonl, safe_join

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
DEFAULT_MODEL_PATH = Path("tasks/pose_landmarker.task")


class VideoOpenError(Exception):
    pass


class FrameSource(Protocol):
    fps: float
    width: int
    height: int

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        ...

    def close(self) -> None:
        ...


class PoseDetector(Protocol):
    def detect(self, frame: Any, timestamp_ms: int) -> list[list[Landmark]]:
        ...

    def close(self) -> None:
        ...


class OpenCvFrameSource:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        self.cap = cv2.VideoCapture(str(input_path))
        if not self.cap.isOpened():
            raise VideoOpenError(f"OpenCV could not open {input_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30.0

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        frame_index = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_index, frame
            frame_index += 1

    def close(self) -> None:
        self.cap.release()


class MediaPipePoseDetector:
    def __init__(self, model_path: Path) -> None:
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: Any, timestamp_ms: int) -> list[list[Landmark]]:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return serialize_pose_landmarks(result.pose_landmarks or [])

    def close(self) -> None:
        self.landmarker.close()


def download_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def timestamp_for_frame(frame_index: int, fps: float) -> int:
    if fps <= 0:
        fps = 30.0
    return int(1000 * frame_index / fps)


def serialize_pose_landmarks(raw_poses: Iterable[Iterable[Any]]) -> list[list[Landmark]]:
    poses: list[list[Landmark]] = []
    for raw_pose in raw_poses:
        pose: list[Landmark] = []
        for landmark in raw_pose:
            pose.append(
                Landmark(
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z) if getattr(landmark, "z", None) is not None else None,
                    visibility=(
                        float(landmark.visibility)
                        if getattr(landmark, "visibility", None) is not None
                        else None
                    ),
                    presence=(
                        float(landmark.presence)
                        if getattr(landmark, "presence", None) is not None
                        else None
                    ),
                )
            )
        poses.append(pose)
    return poses


def extract_pose_jsonl(
    *,
    frame_source: FrameSource,
    detector: PoseDetector,
    output_path: Path,
) -> int:
    frame_count = 0
    try:
        for frame_index, frame in frame_source:
            timestamp_ms = timestamp_for_frame(frame_index, frame_source.fps)
            record = PoseFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                image_width=frame_source.width,
                image_height=frame_source.height,
                poses=detector.detect(frame, timestamp_ms),
            )
            append_jsonl(output_path, record)
            frame_count += 1
    finally:
        detector.close()
        frame_source.close()
    return frame_count


def extract_pose_job(
    paths: ProjectPaths,
    *,
    video_id: str,
    input_ref: str = "videos/source.mp4",
    model_path: Path = DEFAULT_MODEL_PATH,
    frame_source_factory: Callable[[Path], FrameSource] = OpenCvFrameSource,
    detector_factory: Callable[[Path], PoseDetector] = MediaPipePoseDetector,
) -> ProcessingJob:
    job = create_job(
        paths,
        video_id=video_id,
        job_type=JobType.POSE_EXTRACTION,
        input_refs=[input_ref],
        params={
            "model": "pose_landmarker_heavy",
            "model_path": str(model_path),
        },
    )
    running = transition_job(job, JobStatus.RUNNING)
    append_job_state(paths, running)

    output_ref = f"artifacts/jobs/{job.job_id}/pose.jsonl"
    output_path = safe_join(paths.root, *output_ref.split("/"))
    input_path = safe_join(paths.root, *input_ref.split("/"))

    try:
        resolved_model_path = download_model(model_path)
        frame_source = frame_source_factory(input_path)
        detector = detector_factory(resolved_model_path)
        frame_count = extract_pose_jsonl(
            frame_source=frame_source,
            detector=detector,
            output_path=output_path,
        )
        success = transition_job(
            running,
            JobStatus.SUCCESS,
            output_refs=[output_ref],
        )
        success_data = success.model_dump()
        success_data["params"] = {**success.params, "frame_count": frame_count}
        success = ProcessingJob.model_validate(success_data)
        append_job_state(paths, success)
        return success
    except Exception as exc:
        failed = transition_job(
            running,
            JobStatus.FAILED,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            user_action="Check the video path, model file, and available memory, then retry.",
        )
        append_job_state(paths, failed)
        return failed
