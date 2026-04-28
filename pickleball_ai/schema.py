from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"
APP_VERSION = "0.1.0"


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _validate_relative_ref(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ":" in normalized:
        raise ValueError("artifact refs must be relative paths")
    parts = [part for part in normalized.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError("artifact refs must stay inside the project")
    return normalized


class SourceType(StrEnum):
    YOUTUBE = "youtube"
    LOCAL = "local"


class JobType(StrEnum):
    DOWNLOAD = "download"
    METADATA = "metadata"
    POSE_EXTRACTION = "pose_extraction"
    SEGMENT_DETECTION = "segment_detection"
    HIT_CANDIDATE_DETECTION = "hit_candidate_detection"
    METRICS = "metrics"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"


class TimingConfidence(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class Landmark(StrictModel):
    x: float
    y: float
    z: float | None = None
    visibility: float | None = Field(default=None, ge=0, le=1)
    presence: float | None = Field(default=None, ge=0, le=1)


class PoseFrame(StrictModel):
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    poses: list[list[Landmark]] = Field(default_factory=list)


class Project(StrictModel):
    project_id: str = Field(default_factory=new_id)
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    app_version: str = APP_VERSION
    video_id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=utc_now)


class Video(StrictModel):
    video_id: str = Field(default_factory=new_id)
    source_type: SourceType
    source_url: HttpUrl | None = None
    local_path: str
    title: str | None = None
    fps: float = Field(gt=0)
    duration_ms: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, value: str) -> str:
        return _validate_relative_ref(value)

    @model_validator(mode="after")
    def validate_source(self) -> "Video":
        if self.source_type == SourceType.YOUTUBE and self.source_url is None:
            raise ValueError("youtube videos require source_url")
        return self


class TimeWindow(StrictModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TimeWindow":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class Annotation(StrictModel):
    annotation_id: str = Field(default_factory=new_id)
    video_id: str
    event_time_ms: int = Field(ge=0)
    time_window: TimeWindow
    timing_confidence: TimingConfidence
    player_id: str
    action: str
    confidence: float = Field(ge=0, le=1)
    source: str
    source_tool: str = "pickle"
    external_refs: dict[str, Any] = Field(default_factory=dict)
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(ge=0)
    bbox: tuple[int, int, int, int] | None = None
    pose_landmarks_ref: str | None = None

    @field_validator("pose_landmarks_ref")
    @classmethod
    def validate_pose_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_relative_ref(value)

    @model_validator(mode="after")
    def validate_clip_range(self) -> "Annotation":
        if self.clip_end_ms <= self.clip_start_ms:
            raise ValueError("clip_end_ms must be greater than clip_start_ms")
        return self


class ProcessingJob(StrictModel):
    job_id: str = Field(default_factory=new_id)
    video_id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None
    recoverable: bool = True
    user_action: str | None = None

    @field_validator("input_refs", "output_refs")
    @classmethod
    def validate_refs(cls, value: list[str]) -> list[str]:
        return [_validate_relative_ref(ref) for ref in value]

    @model_validator(mode="after")
    def validate_status_payload(self) -> "ProcessingJob":
        if self.status == JobStatus.FAILED:
            if not self.error_type or not self.error_message:
                raise ValueError("failed jobs require error_type and error_message")
        if self.status == JobStatus.SUCCESS and not self.output_refs:
            raise ValueError("successful jobs require output_refs")
        return self


class JobManifest(StrictModel):
    job_id: str
    project_id: str
    artifact_ref: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("artifact_ref")
    @classmethod
    def validate_artifact_ref(cls, value: str) -> str:
        return _validate_relative_ref(value)
