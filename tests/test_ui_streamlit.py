from pickleball_ai.annotations import load_annotations
from pickleball_ai.coverage import load_coverage
from pickleball_ai.schema import (
    CoverageState,
    HitCandidate,
    Project,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    SourceType,
    TargetRef,
    Video,
)
from pickleball_ai.storage import create_project_layout, write_jsonl
from pickleball_ai.ui_streamlit import (
    add_annotation_from_queue,
    add_manual_annotation,
    annotation_option,
    annotation_rows,
    clear_manual_annotation_form_state,
    correct_coverage,
    coverage_option,
    coverage_rows,
    delete_annotation,
    delete_coverage,
    discover_projects,
    export_precheck_warnings,
    find_duplicate_annotations,
    load_hit_candidate_for_queue_item,
    load_workspace,
    manual_annotation_form_key,
    mark_coverage,
    queue_item_defaults,
    queue_rows,
    update_queue_item_status,
)
from pickleball_ai.events import hit_candidates_path
from pickleball_ai.queue import load_review_queue, write_review_queue


def test_discover_projects_lists_project_directories(tmp_path):
    create_project_layout(tmp_path, Project(project_id="project-b"))
    create_project_layout(tmp_path, Project(project_id="project-a"))
    (tmp_path / "not-a-project").mkdir()

    assert discover_projects(tmp_path) == ["project-a", "project-b"]


def test_load_workspace_uses_defaults_when_optional_state_is_missing(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=1000,
    )
    create_project_layout(tmp_path, project, video)

    state = load_workspace(tmp_path, "project-1")

    assert state.project == project
    assert state.video == video
    assert [player.player_id for player in state.players] == ["A", "B", "C", "D"]
    assert state.annotations == []
    assert state.coverage == []
    assert state.queue_items == []


def test_queue_rows_format_target_ref():
    rows = queue_rows(
        [
            ReviewQueueItem(
                reason=QueueReason.HIT_CANDIDATE,
                target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
                status=QueueStatus.OPEN,
                priority=100,
            )
        ]
    )

    assert rows == [
        {
            "status": "open",
            "reason": "hit_candidate",
            "target": "hit_candidate:candidate-1",
            "priority": 100,
        }
    ]


def test_add_manual_annotation_writes_event_and_materialized_annotation(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
        window_radius_ms=200,
    )

    annotations = load_annotations(paths)
    assert annotations == [annotation]
    assert annotation_rows(annotations)[0]["annotation_id"] == annotation.annotation_id
    assert annotation_rows(annotations)[0]["window"] == "800-1200"


def test_find_duplicate_annotations_matches_time_player_and_action(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    first = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
    )
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1601,
        player_id="A",
        action="drive",
    )
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="B",
        action="drive",
    )

    duplicates = find_duplicate_annotations(
        load_annotations(paths),
        event_time_ms=1200,
        player_id="A",
        action="drive",
    )

    assert duplicates == [first]


def test_clear_manual_annotation_form_state_removes_only_annotation_widgets():
    state = {
        manual_annotation_form_key("manual", "event_time_ms"): 1000,
        manual_annotation_form_key("manual", "action"): "drive",
        manual_annotation_form_key("queue-1", "allow_duplicate"): True,
        "coverage_update:start_ms": 0,
        "unrelated": "kept",
    }

    clear_manual_annotation_form_state(state)

    assert state == {
        "coverage_update:start_ms": 0,
        "unrelated": "kept",
    }


def test_queue_item_defaults_load_hit_candidate_time(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    candidate = HitCandidate(
        candidate_id="candidate-1",
        video_id="video-1",
        timestamp_ms=1234,
        time_window={"start_ms": 1000, "end_ms": 1500},
        confidence=0.8,
        source_job_id="job-1",
        pose_frame_index=12,
    )
    write_jsonl(hit_candidates_path(paths, "job-1"), [candidate])
    item = ReviewQueueItem(
        reason=QueueReason.HIT_CANDIDATE,
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        status=QueueStatus.OPEN,
        priority=100,
        created_from_job_id="job-1",
    )

    assert load_hit_candidate_for_queue_item(paths, item) == candidate
    assert queue_item_defaults(paths, item) == {
        "event_time_ms": 1234,
        "action": "unknown",
    }


def test_queue_item_defaults_parse_coverage_gap():
    item = ReviewQueueItem(
        reason=QueueReason.COVERAGE_GAP,
        target_ref=TargetRef(type="coverage_span", id="1000-2000"),
        status=QueueStatus.OPEN,
        priority=60,
    )

    assert queue_item_defaults(None, item) == {
        "coverage_start_ms": 1000,
        "coverage_end_ms": 2000,
    }


def test_add_annotation_from_queue_accepts_queue_item(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    item = ReviewQueueItem(
        queue_item_id="queue-1",
        reason=QueueReason.HIT_CANDIDATE,
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        status=QueueStatus.OPEN,
        priority=100,
    )
    write_review_queue(paths, [item])

    annotation = add_annotation_from_queue(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
        queue_item_id="queue-1",
    )

    assert load_annotations(paths) == [annotation]
    assert load_review_queue(paths)[0].status == QueueStatus.ACCEPTED


def test_update_queue_item_status_rejects_missing_item(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    try:
        update_queue_item_status(paths, "missing", QueueStatus.ACCEPTED)
    except ValueError as exc:
        assert "queue item not found" in str(exc)
    else:
        raise AssertionError("missing queue item should raise")


def test_delete_annotation_writes_deleted_event_and_removes_materialized_annotation(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
    )

    deleted = delete_annotation(paths, annotation.annotation_id)

    assert deleted == annotation
    assert load_annotations(paths) == []


def test_annotation_option_includes_human_context_and_id(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
    )

    assert annotation_option(annotation) == f"1000ms | A | drive | {annotation.annotation_id}"


def test_mark_coverage_writes_event_and_materialized_coverage(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    event = mark_coverage(
        paths,
        start_ms=0,
        end_ms=1000,
        state=CoverageState.REVIEWED,
    )

    coverage = load_coverage(paths)
    assert coverage[0].source_event_id == event.coverage_event_id
    assert coverage_rows(coverage) == [
        {
            "start_ms": 0,
            "end_ms": 1000,
            "state": "reviewed",
            "duration_ms": 1000,
        }
    ]


def test_correct_coverage_appends_replacement_for_selected_span(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    mark_coverage(paths, start_ms=0, end_ms=1000, state=CoverageState.REVIEWED)
    span = load_coverage(paths)[0]

    event = correct_coverage(paths, span, state=CoverageState.NEEDS_RECHECK)

    coverage = load_coverage(paths)
    assert coverage[0].state == CoverageState.NEEDS_RECHECK
    assert coverage[0].source_event_id == event.coverage_event_id
    assert coverage_option(coverage[0]) == (
        f"0-1000ms | needs_recheck | {event.coverage_event_id}"
    )


def test_delete_coverage_clears_selected_span_to_unreviewed(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    mark_coverage(paths, start_ms=0, end_ms=1000, state=CoverageState.REVIEWED)
    span = load_coverage(paths)[0]

    event = delete_coverage(paths, span)

    coverage = load_coverage(paths)
    assert coverage[0].state == CoverageState.UNREVIEWED
    assert coverage[0].reason == "streamlit_delete_coverage"
    assert coverage[0].source_event_id == event.coverage_event_id


def test_export_precheck_warns_about_dataset_quality_issues(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="drive",
    )
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1200,
        player_id="A",
        action="drive",
    )
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=2000,
        player_id="B",
        action="unknown",
    )
    mark_coverage(paths, start_ms=0, end_ms=1000, state=CoverageState.UNREVIEWED)
    item = ReviewQueueItem(
        reason=QueueReason.COVERAGE_GAP,
        target_ref=TargetRef(type="coverage_span", id="0-1000"),
        status=QueueStatus.OPEN,
        priority=60,
    )

    warnings = export_precheck_warnings(
        annotations=load_annotations(paths),
        coverage=load_coverage(paths),
        queue_items=[item],
    )

    assert warnings == [
        "1 coverage gap(s), 1000ms unreviewed.",
        "1 annotation(s) still use unknown action.",
        "1 possible duplicate annotation pair(s) within 300ms.",
        "1 unresolved review queue item(s).",
    ]


def test_streamlit_entrypoint_exposes_export_button():
    source = __import__("pickleball_ai.ui_streamlit", fromlist=["run"])

    assert "Export Dataset" in source.run.__code__.co_consts
    assert "Extract clips with ffmpeg" in source.run.__code__.co_consts
    assert "Clip extraction failed: " in source.run.__code__.co_consts
