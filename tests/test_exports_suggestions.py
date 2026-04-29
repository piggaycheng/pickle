import csv

from pickleball_ai.exports import (
    EXPORT_MANIFEST_REF,
    TIMELINE_REF,
    TRAINING_EXAMPLES_REF,
    build_timeline_rows,
    build_training_examples,
    export_dataset,
)
from pickleball_ai.schema import (
    Annotation,
    ExportManifest,
    ModelSuggestion,
    Project,
    SourceType,
    TargetRef,
    TimeWindow,
    TimingConfidence,
    TrainingExample,
    Video,
)
from pickleball_ai.storage import create_project_layout, read_json, read_jsonl
from pickleball_ai.suggestions import (
    append_model_suggestion,
    evaluate_suggestions,
    load_model_suggestions,
    suggestions_path,
)


def make_annotation(
    annotation_id: str,
    *,
    source: str = "manual",
    action: str = "drive",
    player_id: str = "A",
    target_ref: str | None = None,
) -> Annotation:
    external_refs = {"target_ref": target_ref} if target_ref is not None else {}
    return Annotation(
        annotation_id=annotation_id,
        video_id="video-1",
        event_time_ms=1000,
        time_window=TimeWindow(start_ms=900, end_ms=1100),
        timing_confidence=TimingConfidence.EXACT,
        player_id=player_id,
        action=action,
        confidence=1,
        source=source,
        source_tool="pickle",
        external_refs=external_refs,
        clip_start_ms=800,
        clip_end_ms=1200,
        pose_landmarks_ref="artifacts/jobs/pose-job/pose.jsonl",
    )


def test_timeline_export_rows_exclude_raw_model_suggestions():
    rows = build_timeline_rows(
        [
            make_annotation("ann-1", source="manual"),
            make_annotation("ann-2", source="model_suggestion"),
            make_annotation("ann-3", source="corrected_model_suggestion"),
        ]
    )

    assert [row.annotation_id for row in rows] == ["ann-1", "ann-3"]


def test_training_examples_include_structured_clip_metadata():
    examples = build_training_examples(
        [make_annotation("ann-1", action="slice")],
        video_ref="videos/source.mp4",
    )

    assert len(examples) == 1
    assert examples[0].annotation_id == "ann-1"
    assert examples[0].action == "slice"
    assert examples[0].video_ref == "videos/source.mp4"
    assert examples[0].clip_start_ms == 800
    assert examples[0].pose_landmarks_ref == "artifacts/jobs/pose-job/pose.jsonl"


def test_export_dataset_writes_timeline_training_examples_and_manifest(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)

    manifest = export_dataset(
        paths,
        annotations=[
            make_annotation("ann-1", source="manual"),
            make_annotation("ann-2", source="model_suggestion"),
        ],
    )

    assert manifest.timeline_ref == TIMELINE_REF
    assert manifest.training_examples_ref == TRAINING_EXAMPLES_REF
    assert manifest.annotation_count == 2
    assert manifest.training_example_count == 1
    assert read_json(paths.root / EXPORT_MANIFEST_REF, ExportManifest) == manifest
    assert read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0].annotation_id == "ann-1"
    with (paths.root / TIMELINE_REF).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["annotation_id"] for row in rows] == ["ann-1"]


def test_model_suggestions_round_trip_separately_from_annotations(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    suggestion = ModelSuggestion(
        suggestion_id="suggestion-1",
        job_id="job-1",
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        player_id="A",
        action="drive",
        confidence=0.8,
    )

    append_model_suggestion(paths, suggestion)

    assert suggestions_path(paths, "job-1").exists()
    assert load_model_suggestions(paths, "job-1") == [suggestion]


def test_evaluate_suggestions_compares_against_accepted_annotation_refs():
    suggestion = ModelSuggestion(
        suggestion_id="suggestion-1",
        job_id="job-1",
        target_ref=TargetRef(type="hit_candidate", id="candidate-1"),
        player_id="A",
        action="drive",
        confidence=0.8,
    )
    miss = ModelSuggestion(
        suggestion_id="suggestion-2",
        job_id="job-1",
        target_ref=TargetRef(type="hit_candidate", id="candidate-2"),
        player_id="B",
        action="slice",
        confidence=0.7,
    )
    annotation = make_annotation(
        "ann-1",
        action="drive",
        player_id="A",
        target_ref="hit_candidate:candidate-1",
    )

    result = evaluate_suggestions([suggestion, miss], [annotation])

    assert result == {
        "suggestion_count": 2,
        "matched_annotation_count": 1,
        "action_match_count": 1,
        "player_match_count": 1,
        "unmatched_suggestion_count": 1,
    }

