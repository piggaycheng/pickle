from pickleball_ai.annotations import load_annotations
from pickleball_ai.coverage import load_coverage
from pickleball_ai.schema import (
    CoverageState,
    Project,
    QueueReason,
    QueueStatus,
    ReviewQueueItem,
    SourceType,
    TargetRef,
    Video,
)
from pickleball_ai.storage import create_project_layout
from pickleball_ai.ui_streamlit import (
    add_manual_annotation,
    annotation_option,
    annotation_rows,
    coverage_rows,
    delete_annotation,
    discover_projects,
    load_workspace,
    mark_coverage,
    queue_rows,
)


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


def test_streamlit_entrypoint_exposes_export_button():
    source = __import__("pickleball_ai.ui_streamlit", fromlist=["run"])

    assert "Export Dataset" in source.run.__code__.co_consts
