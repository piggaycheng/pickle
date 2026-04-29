from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

from .jobs import append_job_state, create_job, transition_job
from .schema import HitCandidate, Landmark, JobStatus, JobType, PoseFrame, ProcessingJob, TimeWindow
from .storage import ProjectPaths, read_jsonl, safe_join, write_jsonl

LEFT_WRIST = 15
RIGHT_WRIST = 16


@dataclass(frozen=True)
class HitCandidateConfig:
    score_threshold_px: float = 30.0
    min_gap_ms: int = 250
    window_before_ms: int = 250
    window_after_ms: int = 250
    min_visibility: float = 0.2


def hit_candidates_path(paths: ProjectPaths, job_id: str) -> Path:
    return paths.job_dir(job_id) / "hit_candidates.jsonl"


def generate_hit_candidates(
    pose_frames: list[PoseFrame],
    *,
    video_id: str,
    source_job_id: str | None = None,
    source_pose_ref: str | None = None,
    config: HitCandidateConfig | None = None,
) -> list[HitCandidate]:
    config = config or HitCandidateConfig()
    scored = [
        _score_frame(pose_frames, index, config.min_visibility)
        for index in range(1, max(len(pose_frames) - 1, 1))
    ]
    peaks = [
        score
        for score in scored
        if score is not None
        and score.score_px >= config.score_threshold_px
        and _is_local_peak(scored, score.frame_position)
    ]
    clustered = _cluster_peaks(peaks, pose_frames, config.min_gap_ms)
    candidates: list[HitCandidate] = []
    for peak in clustered:
        frame = pose_frames[peak.frame_position]
        start_ms = max(0, frame.timestamp_ms - config.window_before_ms)
        end_ms = frame.timestamp_ms + config.window_after_ms
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        candidates.append(
            HitCandidate(
                video_id=video_id,
                timestamp_ms=frame.timestamp_ms,
                time_window=TimeWindow(start_ms=start_ms, end_ms=end_ms),
                confidence=min(1.0, peak.score_px / max(config.score_threshold_px * 2, 1.0)),
                source_job_id=source_job_id,
                source_pose_ref=source_pose_ref,
                pose_frame_index=frame.frame_index,
                features={
                    "score_px": peak.score_px,
                    "wrist_speed_px": peak.wrist_speed_px,
                    "wrist_acceleration_px": peak.wrist_acceleration_px,
                },
            )
        )
    return candidates


def extract_hit_candidates_jsonl(
    *,
    pose_path: Path,
    output_path: Path,
    video_id: str,
    source_job_id: str | None = None,
    source_pose_ref: str | None = None,
    config: HitCandidateConfig | None = None,
) -> int:
    if not pose_path.exists():
        raise FileNotFoundError(f"pose artifact not found: {pose_path}")
    pose_frames = read_jsonl(pose_path, PoseFrame)
    candidates = generate_hit_candidates(
        pose_frames,
        video_id=video_id,
        source_job_id=source_job_id,
        source_pose_ref=source_pose_ref,
        config=config,
    )
    write_jsonl(output_path, candidates)
    return len(candidates)


def extract_hit_candidates_job(
    paths: ProjectPaths,
    *,
    video_id: str,
    pose_ref: str,
    source_job_id: str | None = None,
    config: HitCandidateConfig | None = None,
) -> ProcessingJob:
    params = {
        "heuristic": "wrist_motion_peak",
        "score_threshold_px": (config or HitCandidateConfig()).score_threshold_px,
        "min_gap_ms": (config or HitCandidateConfig()).min_gap_ms,
    }
    job = create_job(
        paths,
        video_id=video_id,
        job_type=JobType.HIT_CANDIDATE_DETECTION,
        input_refs=[pose_ref],
        params=params,
    )
    running = transition_job(job, JobStatus.RUNNING)
    append_job_state(paths, running)

    output_ref = f"artifacts/jobs/{job.job_id}/hit_candidates.jsonl"
    pose_path = safe_join(paths.root, *pose_ref.split("/"))
    output_path = safe_join(paths.root, *output_ref.split("/"))

    try:
        candidate_count = extract_hit_candidates_jsonl(
            pose_path=pose_path,
            output_path=output_path,
            video_id=video_id,
            source_job_id=source_job_id,
            source_pose_ref=pose_ref,
            config=config,
        )
        success = transition_job(running, JobStatus.SUCCESS, output_refs=[output_ref])
        success_data = success.model_dump()
        success_data["params"] = {**success.params, "candidate_count": candidate_count}
        success = ProcessingJob.model_validate(success_data)
        append_job_state(paths, success)
        return success
    except Exception as exc:
        failed = transition_job(
            running,
            JobStatus.FAILED,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            user_action="Check the pose artifact path and rerun hit candidate generation.",
        )
        append_job_state(paths, failed)
        return failed


@dataclass(frozen=True)
class _FrameScore:
    frame_position: int
    score_px: float
    wrist_speed_px: float
    wrist_acceleration_px: float


def _score_frame(frames: list[PoseFrame], position: int, min_visibility: float) -> _FrameScore | None:
    if position <= 0 or position >= len(frames) - 1:
        return None
    prev_frame = frames[position - 1]
    frame = frames[position]
    next_frame = frames[position + 1]

    scores: list[tuple[float, float]] = []
    for landmark_index in (LEFT_WRIST, RIGHT_WRIST):
        prev_point = _landmark_point(prev_frame, landmark_index, min_visibility)
        point = _landmark_point(frame, landmark_index, min_visibility)
        next_point = _landmark_point(next_frame, landmark_index, min_visibility)
        if prev_point is None or point is None or next_point is None:
            continue
        prev_speed = _distance(prev_point, point)
        next_speed = _distance(point, next_point)
        acceleration = abs(next_speed - prev_speed)
        wrist_speed = max(prev_speed, next_speed)
        scores.append((wrist_speed, acceleration))

    if not scores:
        return None

    wrist_speed_px = max(speed for speed, _ in scores)
    wrist_acceleration_px = max(acceleration for _, acceleration in scores)
    score_px = wrist_speed_px + wrist_acceleration_px
    return _FrameScore(
        frame_position=position,
        score_px=score_px,
        wrist_speed_px=wrist_speed_px,
        wrist_acceleration_px=wrist_acceleration_px,
    )


def _landmark_point(frame: PoseFrame, landmark_index: int, min_visibility: float) -> tuple[float, float] | None:
    if not frame.poses:
        return None
    pose = frame.poses[0]
    if landmark_index >= len(pose):
        return None
    landmark = pose[landmark_index]
    if not _is_visible(landmark, min_visibility):
        return None
    return (landmark.x * frame.image_width, landmark.y * frame.image_height)


def _is_visible(landmark: Landmark, min_visibility: float) -> bool:
    if landmark.visibility is None:
        return True
    return landmark.visibility >= min_visibility


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(right[0] - left[0], right[1] - left[1])


def _is_local_peak(scores: list[_FrameScore | None], frame_position: int) -> bool:
    score = scores[frame_position - 1]
    if score is None:
        return False
    previous_score = scores[frame_position - 2] if frame_position >= 2 else None
    next_score = scores[frame_position] if frame_position < len(scores) else None
    return (
        (previous_score is None or score.score_px >= previous_score.score_px)
        and (next_score is None or score.score_px >= next_score.score_px)
    )


def _cluster_peaks(peaks: list[_FrameScore], frames: list[PoseFrame], min_gap_ms: int) -> list[_FrameScore]:
    clustered: list[_FrameScore] = []
    for peak in sorted(peaks, key=lambda item: frames[item.frame_position].timestamp_ms):
        if not clustered:
            clustered.append(peak)
            continue
        last = clustered[-1]
        elapsed_ms = frames[peak.frame_position].timestamp_ms - frames[last.frame_position].timestamp_ms
        if elapsed_ms >= min_gap_ms:
            clustered.append(peak)
        elif peak.score_px > last.score_px:
            clustered[-1] = peak
    return clustered
