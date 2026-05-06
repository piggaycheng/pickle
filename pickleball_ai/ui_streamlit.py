from __future__ import annotations

from collections import Counter
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

from .annotations import (
    append_annotation_event,
    create_annotation_event,
    delete_annotation_event,
    load_annotation_events,
    load_annotations,
    rebuild_annotations,
    update_annotation_event,
)
from .clips import ClipExtractionError
from .coverage import append_coverage_event, load_coverage, rebuild_coverage
from .events import hit_candidates_path
from .exports import (
    clear_export_outputs,
    compare_export_runs,
    export_dataset,
    export_run_detail,
    export_run_rows,
    export_staleness_warnings,
    latest_or_legacy_export_manifest_ref,
    promote_export_run_to_latest,
)
from .metrics import compute_metrics, rebuild_metrics
from .players import default_players, load_players, players_path, write_players
from .queue import load_review_queue
from .schema import (
    Annotation,
    CoverageEvent,
    CoverageSpan,
    CoverageState,
    ExportManifest,
    HitCandidate,
    Player,
    Project,
    ProjectMetrics,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    TimeWindow,
    TimingConfidence,
    TrainingExample,
    Video,
)
from .storage import ProjectPaths, get_data_root, project_paths, read_json, read_jsonl, safe_join, write_jsonl
from .summary import build_project_summary, rebuild_project_summary

ACTION_LABELS = ["drive", "slice", "volley", "dink", "lob", "serve", "unknown", "not-hit"]
REQUIRED_ACTION_PLACEHOLDER = "Select action..."
DUPLICATE_TOLERANCE_MS = 300
MANUAL_ANNOTATION_STATE_PREFIX = "manual_annotation:"


def manual_annotation_form_key(scope: str, field: str) -> str:
    return f"{MANUAL_ANNOTATION_STATE_PREFIX}{scope}:{field}"


def clear_manual_annotation_form_state(session_state: MutableMapping[str, object]) -> None:
    for key in list(session_state):
        if key.startswith(MANUAL_ANNOTATION_STATE_PREFIX):
            session_state.pop(key, None)


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


def export_precheck_warnings(
    *,
    annotations: list[Annotation],
    coverage: list[CoverageSpan],
    queue_items: list[ReviewQueueItem],
) -> list[str]:
    trusted_annotations = [
        annotation
        for annotation in annotations
        if annotation.source != "model_suggestion"
    ]
    warnings: list[str] = []

    coverage_gaps = [span for span in coverage if span.state == CoverageState.UNREVIEWED]
    if coverage_gaps:
        gap_duration_ms = sum(span.end_ms - span.start_ms for span in coverage_gaps)
        warnings.append(
            f"{len(coverage_gaps)} coverage gap(s), {gap_duration_ms}ms unreviewed."
        )

    unknown_annotations = [
        annotation
        for annotation in trusted_annotations
        if annotation.action == "unknown"
    ]
    if unknown_annotations:
        warnings.append(f"{len(unknown_annotations)} annotation(s) still use unknown action.")

    duplicate_pairs = _duplicate_annotation_pairs(trusted_annotations)
    if duplicate_pairs:
        warnings.append(
            f"{len(duplicate_pairs)} possible duplicate annotation pair(s) within "
            f"{DUPLICATE_TOLERANCE_MS}ms."
        )

    open_queue_items = [
        item
        for item in queue_items
        if item.status == QueueStatus.OPEN
    ]
    if open_queue_items:
        warnings.append(f"{len(open_queue_items)} unresolved review queue item(s).")

    return warnings


def training_readiness_report(
    paths: ProjectPaths,
    *,
    annotations: list[Annotation],
    coverage: list[CoverageSpan],
    queue_items: list[ReviewQueueItem],
    minimum_examples_per_action: int = 20,
    action_labels: list[str] | None = None,
) -> dict[str, object]:
    labels = action_labels if action_labels is not None else ACTION_LABELS
    trainable_labels = [label for label in labels if label != "unknown"]
    trusted_annotations = [
        annotation
        for annotation in annotations
        if annotation.source != "model_suggestion"
    ]
    trusted_counts = Counter(annotation.action for annotation in trusted_annotations)

    manifest_ref = latest_or_legacy_export_manifest_ref(paths)
    exported_examples: list[TrainingExample] = []
    manifest_status = "missing"
    if manifest_ref is not None:
        manifest_path = safe_join(paths.root, *manifest_ref.split("/"))
        manifest = read_json(manifest_path, ExportManifest)
        examples_path = safe_join(paths.root, *manifest.training_examples_ref.split("/"))
        if examples_path.exists():
            exported_examples = read_jsonl(examples_path, TrainingExample)
            manifest_status = "present"
        else:
            manifest_status = "missing_training_examples"

    exported_counts = Counter(example.action for example in exported_examples)
    existing_clip_counts: Counter[str] = Counter()
    missing_clip_counts: Counter[str] = Counter()
    for example in exported_examples:
        if example.clip_ref is None:
            missing_clip_counts[example.action] += 1
            continue
        clip_path = safe_join(paths.root, *example.clip_ref.split("/"))
        if clip_path.exists():
            existing_clip_counts[example.action] += 1
        else:
            missing_clip_counts[example.action] += 1

    action_rows: list[dict[str, object]] = []
    for label in trainable_labels:
        trusted_count = trusted_counts[label]
        exported_count = exported_counts[label]
        existing_clips = existing_clip_counts[label]
        missing_clips = missing_clip_counts[label]
        needed = max(0, minimum_examples_per_action - trusted_count)
        if trusted_count == 0:
            status = "missing"
        elif missing_clips:
            status = "blocked_missing_clips"
        elif trusted_count < minimum_examples_per_action:
            status = "needs_examples"
        else:
            status = "ready"
        action_rows.append(
            {
                "action": label,
                "trusted_annotations": trusted_count,
                "exported_examples": exported_count,
                "existing_clips": existing_clips,
                "missing_clips": missing_clips,
                "minimum_examples": minimum_examples_per_action,
                "needed_for_minimum": needed,
                "status": status,
            }
        )

    coverage_gaps = [span for span in coverage if span.state == CoverageState.UNREVIEWED]
    open_queue_items = [item for item in queue_items if item.status == QueueStatus.OPEN]
    duplicate_pairs = _duplicate_annotation_pairs(trusted_annotations)
    stale_warnings = export_staleness_warnings(paths, annotations)
    unknown_count = trusted_counts["unknown"]
    missing_clip_count = sum(missing_clip_counts.values())
    ready_actions = [
        row
        for row in action_rows
        if row["status"] == "ready"
    ]

    blockers: list[str] = []
    improvements: list[str] = []
    if not trusted_annotations:
        blockers.append("No trusted annotations yet.")
    if manifest_status == "missing":
        blockers.append("No latest export found.")
    elif manifest_status == "missing_training_examples":
        blockers.append("Export manifest exists, but training examples are missing.")
    if stale_warnings:
        blockers.extend(stale_warnings)
    if missing_clip_count:
        blockers.append(f"{missing_clip_count} exported training example(s) are missing clips.")
    if unknown_count:
        blockers.append(f"{unknown_count} trusted annotation(s) still use unknown action.")
    if duplicate_pairs:
        blockers.append(
            f"{len(duplicate_pairs)} possible duplicate annotation pair(s) within "
            f"{DUPLICATE_TOLERANCE_MS}ms."
        )
    if coverage_gaps:
        gap_duration_ms = sum(span.end_ms - span.start_ms for span in coverage_gaps)
        blockers.append(f"{len(coverage_gaps)} coverage gap(s), {gap_duration_ms}ms unreviewed.")
    if open_queue_items:
        blockers.append(f"{len(open_queue_items)} unresolved review queue item(s).")
    if len(ready_actions) < 2:
        improvements.append(
            f"Need at least 2 actions with {minimum_examples_per_action}+ examples; "
            f"{len(ready_actions)} ready now."
        )

    if blockers:
        readiness = "Not ready"
    elif improvements:
        readiness = "Almost ready"
    else:
        readiness = "Ready for baseline"

    return {
        "readiness": readiness,
        "minimum_examples_per_action": minimum_examples_per_action,
        "trusted_annotation_count": len(trusted_annotations),
        "exported_example_count": len(exported_examples),
        "ready_action_count": len(ready_actions),
        "unknown_count": unknown_count,
        "duplicate_pair_count": len(duplicate_pairs),
        "coverage_gap_count": len(coverage_gaps),
        "open_queue_count": len(open_queue_items),
        "missing_clip_count": missing_clip_count,
        "export_status": "stale" if stale_warnings else manifest_status,
        "blockers": blockers,
        "improvements": improvements,
        "action_rows": action_rows,
    }


def _duplicate_annotation_pairs(annotations: list[Annotation]) -> list[tuple[Annotation, Annotation]]:
    pairs: list[tuple[Annotation, Annotation]] = []
    ordered = sorted(annotations, key=lambda item: (item.event_time_ms, item.annotation_id))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if right.event_time_ms - left.event_time_ms > DUPLICATE_TOLERANCE_MS:
                break
            if left.player_id == right.player_id and left.action == right.action:
                pairs.append((left, right))
    return pairs


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


def coverage_option(span: CoverageSpan) -> str:
    return f"{span.start_ms}-{span.end_ms}ms | {span.state.value} | {span.source_event_id}"


def editable_coverage_spans(coverage: list[CoverageSpan]) -> list[CoverageSpan]:
    return [
        span
        for span in sorted(
            coverage,
            key=lambda item: (item.start_ms, item.end_ms, item.state.value),
        )
        if span.state != CoverageState.UNREVIEWED
    ]


def export_clip_preview_rows(
    paths: ProjectPaths,
    annotations: list[Annotation] | None = None,
) -> list[dict[str, object]]:
    manifest_ref = latest_or_legacy_export_manifest_ref(paths)
    if manifest_ref is None:
        return []
    manifest_path = safe_join(paths.root, *manifest_ref.split("/"))
    manifest = read_json(manifest_path, ExportManifest)
    if manifest.clips_dir_ref is None or manifest.clip_count == 0:
        return []

    examples_path = safe_join(paths.root, *manifest.training_examples_ref.split("/"))
    examples = read_jsonl(examples_path, TrainingExample)
    annotation_by_id = {
        annotation.annotation_id: annotation
        for annotation in annotations or []
    }
    rows: list[dict[str, object]] = []
    for example in examples:
        if example.clip_ref is None:
            continue
        clip_path = safe_join(paths.root, *example.clip_ref.split("/"))
        current_annotation = annotation_by_id.get(example.annotation_id)
        rows.append(
            {
                "annotation_id": example.annotation_id,
                "time_ms": example.event_time_ms,
                "player": example.player_id,
                "action": current_annotation.action if current_annotation is not None else example.action,
                "clip_ref": example.clip_ref,
                "exists": clip_path.exists(),
            }
        )
    return rows


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
    if action == REQUIRED_ACTION_PLACEHOLDER:
        raise ValueError("action must be selected before accepting a queue item")
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


def correct_annotation_action(paths: ProjectPaths, annotation_id: str, action: str) -> Annotation:
    if action == REQUIRED_ACTION_PLACEHOLDER:
        raise ValueError("action must be selected before correcting an annotation")
    annotations = load_annotations(paths)
    annotation = next(
        (item for item in annotations if item.annotation_id == annotation_id),
        None,
    )
    if annotation is None:
        raise ValueError(f"annotation not found: {annotation_id}")
    if annotation.action == action:
        return annotation
    append_annotation_event(
        paths,
        update_annotation_event(
            annotation,
            {"action": action},
            reason="streamlit_clip_preview_action_correction",
        ),
    )
    updated_annotations = rebuild_annotations(paths)
    return next(item for item in updated_annotations if item.annotation_id == annotation_id)


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


def correct_coverage(
    paths: ProjectPaths,
    span: CoverageSpan,
    *,
    state: CoverageState,
) -> CoverageEvent:
    return mark_coverage(
        paths,
        start_ms=span.start_ms,
        end_ms=span.end_ms,
        state=state,
        reason="streamlit_correct_coverage",
    )


def delete_coverage(paths: ProjectPaths, span: CoverageSpan) -> CoverageEvent:
    return mark_coverage(
        paths,
        start_ms=span.start_ms,
        end_ms=span.end_ms,
        state=CoverageState.UNREVIEWED,
        reason="streamlit_delete_coverage",
    )


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
            queue_requires_action = (
                selected_queue_item is not None
                and selected_queue_item.reason == QueueReason.HIT_CANDIDATE
            )
            action_labels = (
                [REQUIRED_ACTION_PLACEHOLDER, *ACTION_LABELS]
                if queue_requires_action
                else ACTION_LABELS
            )
            default_action = str(selected_queue_defaults.get("action", action_labels[0]))
            default_action_index = action_labels.index(default_action) if default_action in action_labels else 0
            manual_form_scope = (
                selected_queue_item.queue_item_id
                if selected_queue_item is not None
                else "manual"
            )
            event_time_ms = st.number_input(
                "Time",
                min_value=0,
                step=100,
                value=default_event_time_ms,
                key=manual_annotation_form_key(manual_form_scope, "event_time_ms"),
            )
            player_id = st.selectbox(
                "Player",
                [player.player_id for player in state.players],
                key=manual_annotation_form_key(manual_form_scope, "player_id"),
            )
            action = st.selectbox(
                "Action",
                action_labels,
                index=default_action_index,
                key=manual_annotation_form_key(manual_form_scope, "action"),
            )
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
                allow_duplicate = st.checkbox(
                    "Add anyway",
                    key=manual_annotation_form_key(manual_form_scope, "allow_duplicate"),
                )
            submitted = st.form_submit_button("Add")
        if submitted:
            missing_required_action = (
                selected_queue_item is not None
                and selected_queue_item.reason == QueueReason.HIT_CANDIDATE
                and action == REQUIRED_ACTION_PLACEHOLDER
            )
            duplicates = []
            if not missing_required_action:
                duplicates = find_duplicate_annotations(
                    state.annotations,
                    event_time_ms=int(event_time_ms),
                    player_id=player_id,
                    action=action,
                )
            if missing_required_action:
                st.error("Choose an action before accepting this hit candidate.")
            elif duplicates and not allow_duplicate:
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
                clear_manual_annotation_form_state(st.session_state)
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

        if state.coverage:
            editable_coverage = editable_coverage_spans(state.coverage)
            if editable_coverage:
                with st.form("coverage_correction"):
                    coverage_options = {
                        coverage_option(span): span
                        for span in editable_coverage
                    }
                    selected_coverage = st.selectbox("Correct coverage segment", list(coverage_options))
                    corrected_coverage_state = st.selectbox(
                        "Correction",
                        [item.value for item in CoverageState],
                    )
                    correct_coverage_submitted = st.form_submit_button("Correct")
                    delete_coverage_submitted = st.form_submit_button("Clear to unreviewed")
            else:
                correct_coverage_submitted = False
                delete_coverage_submitted = False
                coverage_options = {}
                selected_coverage = ""
                corrected_coverage_state = CoverageState.UNREVIEWED.value
            if correct_coverage_submitted or delete_coverage_submitted:
                span = coverage_options[selected_coverage]
                if correct_coverage_submitted:
                    correct_coverage(
                        state.paths,
                        span,
                        state=CoverageState(corrected_coverage_state),
                    )
                else:
                    delete_coverage(state.paths, span)
                rebuild_metrics(
                    state.paths,
                    annotations=load_annotations(state.paths),
                    coverage=load_coverage(state.paths),
                )
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
        readiness = training_readiness_report(
            state.paths,
            annotations=state.annotations,
            coverage=state.coverage,
            queue_items=state.queue_items,
        )
        st.subheader("Training Readiness")
        readiness_cols = st.columns([1, 1, 1], gap="small")
        readiness_cols[0].metric("Readiness", readiness["readiness"])
        readiness_cols[1].metric("Ready actions", readiness["ready_action_count"])
        readiness_cols[2].metric("Missing clips", readiness["missing_clip_count"])
        if readiness["blockers"]:
            st.warning("Training blockers found.")
            for blocker in readiness["blockers"]:
                st.write(f"- {blocker}")
        elif readiness["improvements"]:
            st.info("Training is close, but needs more labeled clips.")
            for improvement in readiness["improvements"]:
                st.write(f"- {improvement}")
        else:
            st.success("Ready for a baseline training run.")
        st.dataframe(readiness["action_rows"], use_container_width=True, hide_index=True)

        export_warnings = export_precheck_warnings(
            annotations=state.annotations,
            coverage=state.coverage,
            queue_items=state.queue_items,
        )
        export_anyway = False
        if export_warnings:
            st.warning("Export pre-check found issues.")
            for warning in export_warnings:
                st.write(f"- {warning}")
            export_anyway = st.checkbox("Export anyway")
        stale_export_warnings = export_staleness_warnings(state.paths, state.annotations)
        if stale_export_warnings:
            st.warning("Export is stale. Re-export to refresh training_examples.jsonl.")
            for warning in stale_export_warnings:
                st.write(f"- {warning}")
        extract_export_clips = st.checkbox("Extract clips with ffmpeg")
        if st.button("Export Dataset", disabled=bool(export_warnings and not export_anyway)):
            try:
                manifest = export_dataset(state.paths, extract_clips=extract_export_clips)
            except ClipExtractionError as exc:
                rebuild_project_summary(state.paths)
                st.error(f"Clip extraction failed: {exc}")
            else:
                clip_text = (
                    f" and {manifest.clip_count} clips"
                    if manifest.clip_count
                    else ""
                )
                st.success(
                    f"Exported {manifest.training_example_count} training examples"
                    f"{clip_text} to {manifest.training_examples_ref}."
                )
        if st.button("Clear export outputs"):
            removed_refs = clear_export_outputs(state.paths)
            rebuild_project_summary(state.paths)
            if removed_refs:
                st.success(f"Cleared {len(removed_refs)} export output(s).")
            else:
                st.info("No export outputs to clear.")
            st.rerun()

        export_history_rows = export_run_rows(state.paths, state.annotations)
        if export_history_rows:
            st.subheader("Export History")
            st.dataframe(export_history_rows, use_container_width=True, hide_index=True)
            export_options = {
                f"{row['generated_at']} | {row['training_examples']} examples | {row['clips']} clips": row
                for row in export_history_rows
            }
            selected_export = st.selectbox("View export run", list(export_options))
            selected_export_id = str(export_options[selected_export]["export_id"])
            export_detail = export_run_detail(state.paths, selected_export_id)
            manifest_detail = export_detail["manifest"]
            clip_summary = export_detail["clip_summary"]
            detail_cols = st.columns([1, 1, 1], gap="small")
            detail_cols[0].metric("Run examples", manifest_detail["training_examples"])
            detail_cols[1].metric("Run clips", manifest_detail["clips"])
            detail_cols[2].metric("Missing clips", clip_summary["missing"])
            st.dataframe(export_detail["action_counts"], use_container_width=True, hide_index=True)
            st.dataframe(export_detail["training_examples"], use_container_width=True, hide_index=True)
            if st.button("Promote to latest"):
                promote_export_run_to_latest(
                    state.paths,
                    selected_export_id,
                )
                rebuild_project_summary(state.paths)
                st.success("Promoted export run to latest.")
                st.rerun()
            if len(export_options) >= 2:
                st.subheader("Export Diff")
                base_export = st.selectbox("Base export run", list(export_options), key="export_diff_base")
                compare_export = st.selectbox(
                    "Compare export run",
                    list(export_options),
                    key="export_diff_compare",
                )
                if base_export != compare_export:
                    diff = compare_export_runs(
                        state.paths,
                        base_export_id=str(export_options[base_export]["export_id"]),
                        compare_export_id=str(export_options[compare_export]["export_id"]),
                    )
                    diff_cols = st.columns([1, 1, 1], gap="small")
                    diff_cols[0].metric("Added", diff["added_count"])
                    diff_cols[1].metric("Removed", diff["removed_count"])
                    diff_cols[2].metric("Changed", diff["changed_count"])
                    st.dataframe(diff["action_delta"], use_container_width=True, hide_index=True)
                    st.dataframe(diff["added_examples"], use_container_width=True, hide_index=True)
                    st.dataframe(diff["removed_examples"], use_container_width=True, hide_index=True)
                    st.dataframe(diff["changed_examples"], use_container_width=True, hide_index=True)

        clip_preview_rows = export_clip_preview_rows(state.paths, state.annotations)
        if clip_preview_rows:
            st.subheader("Exported Clips")
            st.dataframe(clip_preview_rows, use_container_width=True, hide_index=True)
            playable_clips = {
                (
                    f"{row['time_ms']}ms | {row['player']} | "
                    f"{row['action']} | {row['annotation_id']}"
                ): row
                for row in clip_preview_rows
                if row["exists"]
            }
            if playable_clips:
                selected_clip = st.selectbox("Preview clip", list(playable_clips))
                selected_clip_row = playable_clips[selected_clip]
                clip_ref = str(selected_clip_row["clip_ref"])
                st.video(str(safe_join(state.paths.root, *clip_ref.split("/"))))
                with st.form("clip_preview_correction"):
                    current_action = str(selected_clip_row["action"])
                    current_action_index = (
                        ACTION_LABELS.index(current_action)
                        if current_action in ACTION_LABELS
                        else 0
                    )
                    corrected_action = st.selectbox(
                        "Correct action",
                        ACTION_LABELS,
                        index=current_action_index,
                    )
                    correction_submitted = st.form_submit_button("Update action")
                if correction_submitted:
                    correct_annotation_action(
                        state.paths,
                        str(selected_clip_row["annotation_id"]),
                        corrected_action,
                    )
                    rebuild_metrics(
                        state.paths,
                        annotations=load_annotations(state.paths),
                        coverage=load_coverage(state.paths),
                        annotation_events=load_annotation_events(state.paths),
                    )
                    rebuild_project_summary(state.paths)
                    st.info("Action updated. Re-export the dataset to refresh training_examples.jsonl.")
                    st.rerun()
            else:
                st.warning("Export manifest references clips, but no clip files were found.")


if __name__ == "__main__":
    run()
