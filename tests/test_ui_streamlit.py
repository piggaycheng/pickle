from pickleball_ai.annotations import load_annotations
from pickleball_ai.annotations import load_annotation_events
from pickleball_ai.coverage import load_coverage
from pickleball_ai.schema import (
    CoverageState,
    ExportManifest,
    HitCandidate,
    Project,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    SourceType,
    TargetRef,
    TrainingExample,
    Video,
)
from pickleball_ai.storage import create_project_layout, write_json, write_jsonl
from pickleball_ai.ui_streamlit import (
    REQUIRED_ACTION_PLACEHOLDER,
    add_annotation_from_queue,
    add_manual_annotation,
    annotation_option,
    annotation_rows,
    clear_manual_annotation_form_state,
    correct_annotation_action,
    correct_coverage,
    coverage_option,
    coverage_rows,
    delete_annotation,
    delete_coverage,
    discover_projects,
    editable_coverage_spans,
    export_clip_preview_rows,
    export_precheck_warnings,
    find_duplicate_annotations,
    load_hit_candidate_for_queue_item,
    load_workspace,
    manual_annotation_form_key,
    mark_coverage,
    queue_item_defaults,
    queue_rows,
    training_readiness_report,
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


def test_add_annotation_from_queue_requires_explicit_action(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    item = ReviewQueueItem(
        queue_item_id="queue-1",
        reason=QueueReason.HIT_CANDIDATE,
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        status=QueueStatus.OPEN,
        priority=100,
    )
    write_review_queue(paths, [item])

    try:
        add_annotation_from_queue(
            paths,
            video_id="video-1",
            event_time_ms=1000,
            player_id="A",
            action=REQUIRED_ACTION_PLACEHOLDER,
            queue_item_id="queue-1",
        )
    except ValueError as exc:
        assert "action must be selected" in str(exc)
    else:
        raise AssertionError("placeholder action should raise")

    assert load_annotations(paths) == []
    assert load_review_queue(paths)[0].status == QueueStatus.OPEN


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


def test_correct_annotation_action_updates_materialized_annotation(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="unknown",
    )

    updated = correct_annotation_action(paths, annotation.annotation_id, "drive")

    annotations = load_annotations(paths)
    events = load_annotation_events(paths)
    assert updated.action == "drive"
    assert annotations[0].action == "drive"
    assert events[-1].before["action"] == "unknown"
    assert events[-1].after == {"action": "drive"}
    assert events[-1].reason == "streamlit_clip_preview_action_correction"


def test_correct_annotation_action_rejects_placeholder(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="unknown",
    )

    try:
        correct_annotation_action(paths, annotation.annotation_id, REQUIRED_ACTION_PLACEHOLDER)
    except ValueError as exc:
        assert "action must be selected" in str(exc)
    else:
        raise AssertionError("placeholder action should raise")

    assert load_annotations(paths)[0].action == "unknown"


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


def test_editable_coverage_spans_excludes_unreviewed_segments(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    mark_coverage(paths, start_ms=0, end_ms=1000, state=CoverageState.REVIEWED)
    delete_coverage(paths, load_coverage(paths)[0])

    assert editable_coverage_spans(load_coverage(paths)) == []


def test_export_clip_preview_rows_returns_empty_without_manifest(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    assert export_clip_preview_rows(paths) == []


def test_export_clip_preview_rows_lists_extracted_clips(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    manifest = ExportManifest(
        project_id="project-1",
        video_id="video-1",
        timeline_ref="exports/timeline.csv",
        training_examples_ref="exports/training_examples.jsonl",
        clips_dir_ref="exports/clips",
        clip_extraction_job_id="job-1",
        clip_count=1,
        annotation_count=1,
        training_example_count=1,
    )
    example = TrainingExample(
        example_id="example-1",
        annotation_id="ann-1",
        video_id="video-1",
        video_ref="videos/source.mp4",
        clip_ref="exports/clips/ann-1.mp4",
        player_id="A",
        action="drive",
        event_time_ms=1000,
        clip_start_ms=800,
        clip_end_ms=1200,
        timing_confidence="exact",
        source_tool="pickle",
    )
    write_json(paths.root / "exports/export_manifest.json", manifest)
    write_jsonl(paths.root / "exports/training_examples.jsonl", [example])
    (paths.root / "exports/clips").mkdir(parents=True)
    (paths.root / "exports/clips/ann-1.mp4").write_bytes(b"clip")

    assert export_clip_preview_rows(paths) == [
        {
            "annotation_id": "ann-1",
            "time_ms": 1000,
            "player": "A",
            "action": "drive",
            "clip_ref": "exports/clips/ann-1.mp4",
            "exists": True,
        }
    ]


def test_export_clip_preview_rows_prefers_current_annotation_action(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotation = add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="slice",
    )
    manifest = ExportManifest(
        project_id="project-1",
        video_id="video-1",
        timeline_ref="exports/timeline.csv",
        training_examples_ref="exports/training_examples.jsonl",
        clips_dir_ref="exports/clips",
        clip_extraction_job_id="job-1",
        clip_count=1,
        annotation_count=1,
        training_example_count=1,
    )
    example = TrainingExample(
        example_id="example-1",
        annotation_id=annotation.annotation_id,
        video_id="video-1",
        video_ref="videos/source.mp4",
        clip_ref=f"exports/clips/{annotation.annotation_id}.mp4",
        player_id="A",
        action="drive",
        event_time_ms=1000,
        clip_start_ms=800,
        clip_end_ms=1200,
        timing_confidence="exact",
        source_tool="pickle",
    )
    write_json(paths.root / "exports/export_manifest.json", manifest)
    write_jsonl(paths.root / "exports/training_examples.jsonl", [example])
    (paths.root / "exports/clips").mkdir(parents=True)
    (paths.root / f"exports/clips/{annotation.annotation_id}.mp4").write_bytes(b"clip")

    rows = export_clip_preview_rows(paths, load_annotations(paths))

    assert rows[0]["action"] == "slice"


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


def test_training_readiness_report_flags_training_blockers(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1000,
        player_id="A",
        action="unknown",
    )
    add_manual_annotation(
        paths,
        video_id="video-1",
        event_time_ms=1200,
        player_id="A",
        action="unknown",
    )
    mark_coverage(paths, start_ms=0, end_ms=1000, state=CoverageState.UNREVIEWED)
    item = ReviewQueueItem(
        reason=QueueReason.COVERAGE_GAP,
        target_ref=TargetRef(type="coverage_span", id="0-1000"),
        status=QueueStatus.OPEN,
        priority=60,
    )

    report = training_readiness_report(
        paths,
        annotations=load_annotations(paths),
        coverage=load_coverage(paths),
        queue_items=[item],
        minimum_examples_per_action=2,
    )

    assert report["readiness"] == "Not ready"
    assert report["export_status"] == "missing"
    assert report["unknown_count"] == 2
    assert report["duplicate_pair_count"] == 1
    assert report["coverage_gap_count"] == 1
    assert report["open_queue_count"] == 1
    assert report["blockers"] == [
        "No latest export found.",
        "2 trusted annotation(s) still use unknown action.",
        "1 possible duplicate annotation pair(s) within 300ms.",
        "1 coverage gap(s), 1000ms unreviewed.",
        "1 unresolved review queue item(s).",
    ]


def test_training_readiness_report_ready_when_two_actions_have_clips(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    annotations = [
        add_manual_annotation(
            paths,
            video_id="video-1",
            event_time_ms=1000,
            player_id="A",
            action="drive",
        ),
        add_manual_annotation(
            paths,
            video_id="video-1",
            event_time_ms=2000,
            player_id="A",
            action="drive",
        ),
        add_manual_annotation(
            paths,
            video_id="video-1",
            event_time_ms=3000,
            player_id="B",
            action="slice",
        ),
        add_manual_annotation(
            paths,
            video_id="video-1",
            event_time_ms=4000,
            player_id="B",
            action="slice",
        ),
    ]
    examples = [
        TrainingExample(
            example_id=f"example-{annotation.annotation_id}",
            annotation_id=annotation.annotation_id,
            video_id=annotation.video_id,
            video_ref="videos/source.mp4",
            clip_ref=f"exports/latest/clips/{annotation.annotation_id}.mp4",
            player_id=annotation.player_id,
            action=annotation.action,
            event_time_ms=annotation.event_time_ms,
            clip_start_ms=annotation.clip_start_ms,
            clip_end_ms=annotation.clip_end_ms,
            timing_confidence=annotation.timing_confidence,
            source_tool=annotation.source_tool,
        )
        for annotation in annotations
    ]
    manifest = ExportManifest(
        project_id="project-1",
        video_id="video-1",
        timeline_ref="exports/latest/timeline.csv",
        training_examples_ref="exports/latest/training_examples.jsonl",
        clips_dir_ref="exports/latest/clips",
        clip_count=4,
        annotation_count=4,
        training_example_count=4,
    )
    write_json(paths.root / "exports/latest/export_manifest.json", manifest)
    write_jsonl(paths.root / "exports/latest/training_examples.jsonl", examples)
    (paths.root / "exports/latest/clips").mkdir(parents=True)
    for annotation in annotations:
        (paths.root / f"exports/latest/clips/{annotation.annotation_id}.mp4").write_bytes(b"clip")

    report = training_readiness_report(
        paths,
        annotations=load_annotations(paths),
        coverage=[],
        queue_items=[],
        minimum_examples_per_action=2,
    )

    assert report["readiness"] == "Ready for baseline"
    assert report["export_status"] == "present"
    assert report["ready_action_count"] == 2
    assert report["missing_clip_count"] == 0
    assert report["blockers"] == []
    assert report["improvements"] == []
    rows_by_action = {str(row["action"]): row for row in report["action_rows"]}
    assert rows_by_action["drive"]["status"] == "ready"
    assert rows_by_action["drive"]["trusted_annotations"] == 2
    assert rows_by_action["slice"]["status"] == "ready"


def test_streamlit_entrypoint_exposes_export_button():
    source = __import__("pickleball_ai.ui_streamlit", fromlist=["run"])

    assert "Export Dataset" in source.run.__code__.co_consts
    assert "Extract clips with ffmpeg" in source.run.__code__.co_consts
    assert "Export is stale. Re-export to refresh training_examples.jsonl." in source.run.__code__.co_consts
    assert "Clear export outputs" in source.run.__code__.co_consts
    assert "Export History" in source.run.__code__.co_consts
    assert "View export run" in source.run.__code__.co_consts
    assert "Run examples" in source.run.__code__.co_consts
    assert "Run clips" in source.run.__code__.co_consts
    assert "Missing clips" in source.run.__code__.co_consts
    assert "Export Diff" in source.run.__code__.co_consts
    assert "Base export run" in source.run.__code__.co_consts
    assert "Compare export run" in source.run.__code__.co_consts
    assert "Added" in source.run.__code__.co_consts
    assert "Removed" in source.run.__code__.co_consts
    assert "Changed" in source.run.__code__.co_consts
    assert "Promote to latest" in source.run.__code__.co_consts
    assert "Clip extraction failed: " in source.run.__code__.co_consts
    assert "Clear to unreviewed" in source.run.__code__.co_consts
    assert "Training Readiness" in source.run.__code__.co_consts
    assert "Ready for a baseline training run." in source.run.__code__.co_consts
