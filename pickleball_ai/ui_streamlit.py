from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .annotations import (
    append_annotation_event,
    create_annotation_event,
    delete_annotation_event,
    load_annotations,
    rebuild_annotations,
)
from .coverage import append_coverage_event, load_coverage, rebuild_coverage
from .events import hit_candidates_path
from .exports import export_dataset
from .metrics import compute_metrics, rebuild_metrics
from .players import default_players, load_players, players_path, write_players
from .queue import load_review_queue
from .schema import (
    Annotation,
    CoverageEvent,
    CoverageSpan,
    CoverageState,
    HitCandidate,
    Player,
    Project,
    ProjectMetrics,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    TimeWindow,
    TimingConfidence,
    Video,
)
from .storage import ProjectPaths, get_data_root, project_paths, read_json, read_jsonl, write_jsonl
from .summary import build_project_summary, rebuild_project_summary

ACTION_LABELS = ["drive", "slice", "volley", "dink", "lob", "serve", "unknown", "not-hit"]
DUPLICATE_TOLERANCE_MS = 300


@dataclass(frozen=True)
class WorkspaceState:
    paths: ProjectPaths
    project: Project
    video: Video | None
    players: list[Player]
    annotations: list[Annotation]
    coverage: list[CoverageSpan]
    queue_items: list[ReviewQueueItem]
    metrics: ProjectMetrics


def discover_projects(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / "project.json").exists()
    )


def load_workspace(data_root: Path, project_id: str) -> WorkspaceState:
    paths = project_paths(data_root, project_id)
    project = read_json(paths.project_json, Project)
    video = read_json(paths.video_json, Video) if paths.video_json.exists() else None
    players = load_players(paths) if players_path(paths).exists() else default_players()
    annotations = load_annotations(paths)
    coverage = load_coverage(paths)
    queue_items = load_review_queue(paths)
    metrics = compute_metrics(annotations=annotations, coverage=coverage)
    return WorkspaceState(
        paths=paths,
        project=project,
        video=video,
        players=players,
        annotations=annotations,
        coverage=coverage,
        queue_items=queue_items,
        metrics=metrics,
    )


def queue_rows(queue_items: list[ReviewQueueItem]) -> list[dict[str, object]]:
    return [
        {
            "status": item.status.value,
            "reason": item.reason.value,
            "target": f"{item.target_ref.type}:{item.target_ref.id}",
            "priority": item.priority,
        }
        for item in queue_items
    ]


def queue_option(item: ReviewQueueItem) -> str:
    return (
        f"{item.reason.value} | {item.status.value} | "
        f"{item.target_ref.type}:{item.target_ref.id}"
    )


def find_duplicate_annotations(
    annotations: list[Annotation],
    *,
    event_time_ms: int,
    player_id: str,
    action: str,
    tolerance_ms: int = DUPLICATE_TOLERANCE_MS,
) -> list[Annotation]:
    return [
        annotation
        for annotation in annotations
        if annotation.player_id == player_id
        and annotation.action == action
        and abs(annotation.event_time_ms - event_time_ms) <= tolerance_ms
    ]


def load_hit_candidate_for_queue_item(paths: ProjectPaths, item: ReviewQueueItem) -> HitCandidate | None:
    if item.reason != QueueReason.HIT_CANDIDATE or item.target_ref.type != "hit_candidate":
        return None
    if item.created_from_job_id is None:
        return None
    candidate_path = hit_candidates_path(paths, item.created_from_job_id)
    candidates = read_jsonl(candidate_path, HitCandidate)
    return next((candidate for candidate in candidates if candidate.candidate_id == item.target_ref.id), None)


def queue_item_defaults(paths: ProjectPaths, item: ReviewQueueItem) -> dict[str, object]:
    if item.reason == QueueReason.HIT_CANDIDATE:
        candidate = load_hit_candidate_for_queue_item(paths, item)
        if candidate is None:
            return {}
        return {
            "event_time_ms": candidate.timestamp_ms,
            "action": "unknown",
        }
    if item.reason == QueueReason.COVERAGE_GAP:
        start_text, _, end_text = item.target_ref.id.partition("-")
        if start_text.isdigit() and end_text.isdigit():
            start_ms = int(start_text)
            end_ms = int(end_text)
            return {
                "coverage_start_ms": start_ms,
                "coverage_end_ms": end_ms,
            }
    return {}


def update_queue_item_status(
    paths: ProjectPaths,
    queue_item_id: str,
    status: QueueStatus,
) -> ReviewQueueItem:
    items = load_review_queue(paths)
    updated_items: list[ReviewQueueItem] = []
    updated: ReviewQueueItem | None = None
    for item in items:
        if item.queue_item_id != queue_item_id:
            updated_items.append(item)
            continue
        data = item.model_dump()
        data["status"] = status
        updated = ReviewQueueItem.model_validate(data)
        updated_items.append(updated)
    if updated is None:
        raise ValueError(f"queue item not found: {queue_item_id}")
    write_jsonl(paths.state_dir / "review_queue.jsonl", updated_items)
    return updated


def annotation_rows(annotations: list[Annotation]) -> list[dict[str, object]]:
    return [
        {
            "annotation_id": annotation.annotation_id,
            "time_ms": annotation.event_time_ms,
            "player": annotation.player_id,
            "action": annotation.action,
            "confidence": annotation.confidence,
            "window": f"{annotation.time_window.start_ms}-{annotation.time_window.end_ms}",
            "source": annotation.source,
        }
        for annotation in sorted(annotations, key=lambda item: (item.event_time_ms, item.annotation_id))
    ]


def annotation_option(annotation: Annotation) -> str:
    return (
        f"{annotation.event_time_ms}ms | {annotation.player_id} | "
        f"{annotation.action} | {annotation.annotation_id}"
    )


def coverage_rows(coverage: list[CoverageSpan]) -> list[dict[str, object]]:
    return [
        {
            "start_ms": span.start_ms,
            "end_ms": span.end_ms,
            "state": span.state.value,
            "duration_ms": span.end_ms - span.start_ms,
        }
        for span in coverage
    ]


def add_manual_annotation(
    paths: ProjectPaths,
    *,
    video_id: str,
    event_time_ms: int,
    player_id: str,
    action: str,
    window_radius_ms: int = 250,
) -> Annotation:
    start_ms = max(0, event_time_ms - window_radius_ms)
    end_ms = event_time_ms + window_radius_ms
    annotation = Annotation(
        video_id=video_id,
        event_time_ms=event_time_ms,
        time_window=TimeWindow(start_ms=start_ms, end_ms=end_ms),
        timing_confidence=TimingConfidence.EXACT,
        player_id=player_id,
        action=action,
        confidence=1,
        source="manual",
        clip_start_ms=start_ms,
        clip_end_ms=end_ms,
    )
    append_annotation_event(paths, create_annotation_event(annotation, reason="streamlit_manual_entry"))
    rebuild_annotations(paths)
    return annotation


def add_annotation_from_queue(
    paths: ProjectPaths,
    *,
    video_id: str,
    event_time_ms: int,
    player_id: str,
    action: str,
    queue_item_id: str,
    window_radius_ms: int = 250,
) -> Annotation:
    annotation = add_manual_annotation(
        paths,
        video_id=video_id,
        event_time_ms=event_time_ms,
        player_id=player_id,
        action=action,
        window_radius_ms=window_radius_ms,
    )
    update_queue_item_status(paths, queue_item_id, QueueStatus.ACCEPTED)
    return annotation


def delete_annotation(paths: ProjectPaths, annotation_id: str) -> Annotation:
    annotations = load_annotations(paths)
    annotation = next(
        (item for item in annotations if item.annotation_id == annotation_id),
        None,
    )
    if annotation is None:
        raise ValueError(f"annotation not found: {annotation_id}")
    append_annotation_event(
        paths,
        delete_annotation_event(annotation, reason="streamlit_delete_annotation"),
    )
    rebuild_annotations(paths)
    return annotation


def mark_coverage(
    paths: ProjectPaths,
    *,
    start_ms: int,
    end_ms: int,
    state: CoverageState,
    reason: str = "streamlit_manual_entry",
) -> CoverageEvent:
    event = CoverageEvent(start_ms=start_ms, end_ms=end_ms, state=state, reason=reason)
    append_coverage_event(paths, event)
    rebuild_coverage(paths)
    return event


def run() -> None:
    import streamlit as st

    st.set_page_config(page_title="Pickleball Annotation", layout="wide")
    st.title("Pickleball Annotation")

    data_root = Path(st.sidebar.text_input("Data root", str(get_data_root()))).expanduser().resolve()
    project_ids = discover_projects(data_root)
    if not project_ids:
        st.warning("No projects found.")
        return

    project_id = st.sidebar.selectbox("Project", project_ids)
    state = load_workspace(data_root, project_id)
    if not players_path(state.paths).exists():
        write_players(state.paths, state.players)

    video_path = state.paths.root / state.video.local_path if state.video is not None else None
    if video_path is not None and video_path.exists():
        st.video(str(video_path))

    queue_col, inspector_col = st.columns([1, 1], gap="medium")
    with queue_col:
        st.subheader("Review Queue")
        st.dataframe(queue_rows(state.queue_items), use_container_width=True, hide_index=True)
        active_queue_items = [
            item
            for item in state.queue_items
            if item.status == QueueStatus.OPEN
        ]
        selected_queue_item: ReviewQueueItem | None = None
        selected_queue_defaults: dict[str, object] = {}
        if active_queue_items:
            queue_options = {queue_option(item): item for item in active_queue_items}
            selected_queue_label = st.selectbox("Use queue item", list(queue_options))
            selected_queue_item = queue_options[selected_queue_label]
            selected_queue_defaults = queue_item_defaults(state.paths, selected_queue_item)

    with inspector_col:
        st.subheader("Annotation Inspector")
        with st.form("manual_annotation"):
            default_event_time_ms = int(selected_queue_defaults.get("event_time_ms", 0))
            default_action = str(selected_queue_defaults.get("action", ACTION_LABELS[0]))
            default_action_index = ACTION_LABELS.index(default_action) if default_action in ACTION_LABELS else 0
            event_time_ms = st.number_input("Time", min_value=0, step=100, value=default_event_time_ms)
            player_id = st.selectbox("Player", [player.player_id for player in state.players])
            action = st.selectbox("Action", ACTION_LABELS, index=default_action_index)
            duplicates = find_duplicate_annotations(
                state.annotations,
                event_time_ms=int(event_time_ms),
                player_id=player_id,
                action=action,
            )
            allow_duplicate = False
            if duplicates:
                duplicate_labels = ", ".join(
                    f"{annotation.event_time_ms}ms/{annotation.annotation_id}"
                    for annotation in duplicates
                )
                st.warning(f"Possible duplicate annotation: {duplicate_labels}")
                allow_duplicate = st.checkbox("Add anyway")
            submitted = st.form_submit_button("Add")
        if submitted:
            duplicates = find_duplicate_annotations(
                state.annotations,
                event_time_ms=int(event_time_ms),
                player_id=player_id,
                action=action,
            )
            if duplicates and not allow_duplicate:
                st.error("Duplicate not added. Check Add anyway to keep both.")
            else:
                if selected_queue_item is not None and selected_queue_item.reason == QueueReason.HIT_CANDIDATE:
                    add_annotation_from_queue(
                        state.paths,
                        video_id=state.project.video_id,
                        event_time_ms=int(event_time_ms),
                        player_id=player_id,
                        action=action,
                        queue_item_id=selected_queue_item.queue_item_id,
                    )
                else:
                    add_manual_annotation(
                        state.paths,
                        video_id=state.project.video_id,
                        event_time_ms=int(event_time_ms),
                        player_id=player_id,
                        action=action,
                    )
                rebuild_metrics(
                    state.paths,
                    annotations=load_annotations(state.paths),
                    coverage=load_coverage(state.paths),
                )
                rebuild_project_summary(state.paths)
                st.rerun()

        with st.form("coverage_update"):
            default_start_ms = int(selected_queue_defaults.get("coverage_start_ms", 0))
            default_end_ms = int(selected_queue_defaults.get("coverage_end_ms", 1000))
            start_ms = st.number_input("Start", min_value=0, step=100, value=default_start_ms)
            end_ms = st.number_input("End", min_value=1, step=100, value=default_end_ms)
            coverage_state = st.selectbox("Coverage", [item.value for item in CoverageState])
            coverage_submitted = st.form_submit_button("Mark")
        if coverage_submitted:
            if int(end_ms) <= int(start_ms):
                st.error("End must be greater than Start.")
            else:
                mark_coverage(
                    state.paths,
                    start_ms=int(start_ms),
                    end_ms=int(end_ms),
                    state=CoverageState(coverage_state),
                )
                rebuild_metrics(
                    state.paths,
                    annotations=load_annotations(state.paths),
                    coverage=load_coverage(state.paths),
                )
                if selected_queue_item is not None and selected_queue_item.reason == QueueReason.COVERAGE_GAP:
                    update_queue_item_status(state.paths, selected_queue_item.queue_item_id, QueueStatus.ACCEPTED)
                rebuild_project_summary(state.paths)
                st.rerun()

        if state.annotations:
            with st.form("delete_annotation"):
                sorted_annotations = sorted(
                    state.annotations,
                    key=lambda item: (item.event_time_ms, item.annotation_id),
                )
                delete_options = {
                    annotation_option(annotation): annotation.annotation_id
                    for annotation in sorted_annotations
                }
                selected_annotation = st.selectbox("Delete annotation", list(delete_options))
                delete_submitted = st.form_submit_button("Delete")
            if delete_submitted:
                delete_annotation(state.paths, delete_options[selected_annotation])
                rebuild_metrics(
                    state.paths,
                    annotations=load_annotations(state.paths),
                    coverage=load_coverage(state.paths),
                )
                rebuild_project_summary(state.paths)
                st.rerun()

    st.subheader("Timeline")
    st.dataframe(annotation_rows(state.annotations), use_container_width=True, hide_index=True)

    st.subheader("Coverage")
    st.dataframe(coverage_rows(state.coverage), use_container_width=True, hide_index=True)

    metrics_col, summary_col = st.columns([1, 1], gap="medium")
    with metrics_col:
        st.metric("Reviewed ms", state.metrics.reviewed_duration_ms)
        st.metric("Annotations", state.metrics.annotation_count)
        st.metric(
            "Annotations / reviewed min",
            "n/a"
            if state.metrics.annotations_per_reviewed_minute is None
            else round(state.metrics.annotations_per_reviewed_minute, 2),
        )
    with summary_col:
        summary = build_project_summary(state.paths)
        st.metric("Jobs", summary.job_count)
        st.metric("Failed jobs", len(summary.failed_jobs))
        st.metric("Artifacts", summary.artifact_count)
        if st.button("Export Dataset"):
            manifest = export_dataset(state.paths)
            st.success(
                f"Exported {manifest.training_example_count} training examples to "
                f"{manifest.training_examples_ref}."
            )


if __name__ == "__main__":
    run()
