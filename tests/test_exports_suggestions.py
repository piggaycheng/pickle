import csv
import subprocess

from pickleball_ai.clips import ClipExtractionError
from pickleball_ai.exports import (
    EXPORT_MANIFEST_REF,
    TIMELINE_REF,
    TRAINING_EXAMPLES_REF,
    build_timeline_rows,
    build_training_examples,
    clear_export_outputs,
    export_dataset,
    export_manifest_ref,
    export_run_dir_ref,
    export_run_rows,
    export_staleness_warnings,
    export_training_examples_ref,
    promote_export_run_to_latest,
)
from pickleball_ai.jobs import load_job_history
from pickleball_ai.schema import (
    Annotation,
    ExportManifest,
    JobStatus,
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
    assert examples[0].clip_ref is None
    assert examples[0].clip_start_ms == 800
    assert examples[0].pose_landmarks_ref == "artifacts/jobs/pose-job/pose.jsonl"


def test_training_examples_include_extracted_clip_ref_when_available():
    examples = build_training_examples(
        [make_annotation("ann-1", action="slice")],
        video_ref="videos/source.mp4",
        clip_refs={"ann-1": "exports/clips/ann-1.mp4"},
    )

    assert examples[0].video_ref == "videos/source.mp4"
    assert examples[0].clip_ref == "exports/clips/ann-1.mp4"


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
    assert manifest.clips_dir_ref is None
    assert manifest.clip_extraction_job_id is None
    assert manifest.clip_count == 0
    assert manifest.annotation_count == 2
    assert manifest.training_example_count == 1
    assert read_json(paths.root / EXPORT_MANIFEST_REF, ExportManifest) == manifest
    assert read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0].annotation_id == "ann-1"
    with (paths.root / TIMELINE_REF).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["annotation_id"] for row in rows] == ["ann-1"]


def test_export_dataset_extracts_clips_before_writing_training_examples(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    (paths.root / "videos" / "source.mp4").write_bytes(b"fake video")

    def fake_run(command, **kwargs):
        with open(command[-1], "wb") as handle:
            handle.write(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    manifest = export_dataset(
        paths,
        annotations=[make_annotation("ann-1", source="manual")],
        extract_clips=True,
        run_command=fake_run,
    )

    assert manifest.clip_count == 1
    assert manifest.clips_dir_ref == "exports/latest/clips"
    assert manifest.clip_extraction_job_id is not None
    example = read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0]
    assert example.video_ref == "videos/source.mp4"
    assert example.clip_ref == "exports/latest/clips/ann-1.mp4"
    assert (paths.root / "exports/latest/clips/ann-1.mp4").read_bytes() == b"clip"
    run_dir_ref = export_run_dir_ref(manifest.export_id)
    run_manifest = read_json(paths.root / export_manifest_ref(run_dir_ref), ExportManifest)
    run_example = read_jsonl(paths.root / export_training_examples_ref(run_dir_ref), TrainingExample)[0]
    assert run_manifest.clips_dir_ref == f"{run_dir_ref}/clips"
    assert run_example.clip_ref == f"{run_dir_ref}/clips/ann-1.mp4"
    assert (paths.root / f"{run_dir_ref}/clips/ann-1.mp4").read_bytes() == b"clip"


def test_export_dataset_records_failed_clip_job_and_does_not_write_export_files(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    (paths.root / "videos" / "source.mp4").write_bytes(b"fake video")

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad clip range")

    try:
        export_dataset(
            paths,
            annotations=[make_annotation("ann-1", source="manual")],
            extract_clips=True,
            run_command=failed_run,
        )
    except ClipExtractionError as exc:
        assert "bad clip range" in str(exc)
    else:
        raise AssertionError("failed clip extraction should abort export")

    assert load_job_history(paths.processing_jobs_jsonl)[-1].status == JobStatus.FAILED
    assert not (paths.root / EXPORT_MANIFEST_REF).exists()
    assert not (paths.root / TRAINING_EXAMPLES_REF).exists()
    assert not (paths.root / TIMELINE_REF).exists()


def test_clear_export_outputs_removes_export_files_and_clips_only(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    (paths.root / "videos" / "source.mp4").write_bytes(b"fake video")

    def fake_run(command, **kwargs):
        with open(command[-1], "wb") as handle:
            handle.write(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    export_dataset(
        paths,
        annotations=[make_annotation("ann-1", source="manual")],
        extract_clips=True,
        run_command=fake_run,
    )
    artifact_file = paths.job_dir("pose-job") / "pose.jsonl"
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_text("{}\n", encoding="utf-8")

    removed_refs = clear_export_outputs(paths)

    assert removed_refs == [
        TIMELINE_REF,
        TRAINING_EXAMPLES_REF,
        EXPORT_MANIFEST_REF,
        "exports/latest/clips",
    ]
    assert not (paths.root / TIMELINE_REF).exists()
    assert not (paths.root / TRAINING_EXAMPLES_REF).exists()
    assert not (paths.root / EXPORT_MANIFEST_REF).exists()
    assert not (paths.root / "exports/latest/clips").exists()
    assert any((paths.root / "exports/runs").iterdir())
    assert artifact_file.read_text(encoding="utf-8") == "{}\n"
    assert paths.processing_jobs_jsonl.exists()


def test_clear_export_outputs_noops_without_existing_exports(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    assert clear_export_outputs(paths) == []


def test_export_run_rows_return_empty_without_runs(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    assert export_run_rows(paths) == []


def test_export_run_rows_list_runs_newest_first(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)

    first = export_dataset(paths, annotations=[make_annotation("ann-1", source="manual")])
    second = export_dataset(
        paths,
        annotations=[
            make_annotation("ann-1", source="manual"),
            make_annotation("ann-2", source="manual"),
        ],
    )

    rows = export_run_rows(paths)

    assert [row["export_id"] for row in rows] == [second.export_id, first.export_id]
    assert rows[0]["training_examples"] == 2
    assert rows[0]["clips"] == 0
    assert rows[0]["annotations"] == 2
    assert rows[0]["manifest_ref"] == f"exports/runs/{second.export_id}/export_manifest.json"
    assert rows[1]["training_examples"] == 1


def test_promote_export_run_to_latest_replaces_latest_outputs(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    first = export_dataset(paths, annotations=[make_annotation("ann-1", source="manual", action="drive")])
    second = export_dataset(paths, annotations=[make_annotation("ann-2", source="manual", action="slice")])

    promoted = promote_export_run_to_latest(paths, first.export_id)

    assert promoted.export_id == first.export_id
    assert promoted.generated_at == first.generated_at
    assert read_json(paths.root / EXPORT_MANIFEST_REF, ExportManifest) == promoted
    assert read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0].annotation_id == "ann-1"
    assert read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0].action == "drive"
    assert (paths.root / export_manifest_ref(export_run_dir_ref(second.export_id))).exists()


def test_promote_export_run_to_latest_rewrites_clip_refs(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    (paths.root / "videos" / "source.mp4").write_bytes(b"fake video")

    def fake_run(command, **kwargs):
        with open(command[-1], "wb") as handle:
            handle.write(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    first = export_dataset(
        paths,
        annotations=[make_annotation("ann-1", source="manual", action="drive")],
        extract_clips=True,
        run_command=fake_run,
    )
    export_dataset(paths, annotations=[make_annotation("ann-2", source="manual", action="slice")])

    promoted = promote_export_run_to_latest(paths, first.export_id)

    latest_example = read_jsonl(paths.root / TRAINING_EXAMPLES_REF, TrainingExample)[0]
    assert promoted.clips_dir_ref == "exports/latest/clips"
    assert latest_example.clip_ref == "exports/latest/clips/ann-1.mp4"
    assert (paths.root / "exports/latest/clips/ann-1.mp4").read_bytes() == b"clip"
    assert (paths.root / f"exports/runs/{first.export_id}/clips/ann-1.mp4").read_bytes() == b"clip"


def test_promote_export_run_to_latest_rejects_missing_run(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    try:
        promote_export_run_to_latest(paths, "missing")
    except FileNotFoundError as exc:
        assert "export run manifest not found" in str(exc)
    else:
        raise AssertionError("missing export run should raise")


def test_export_staleness_warnings_return_empty_without_manifest(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    assert export_staleness_warnings(paths, [make_annotation("ann-1")]) == []


def test_export_staleness_warnings_return_empty_when_current_annotations_match(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    annotations = [make_annotation("ann-1", source="manual")]
    export_dataset(paths, annotations=annotations)

    assert export_staleness_warnings(paths, annotations) == []


def test_export_staleness_warnings_detect_changed_annotation_fields(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    export_dataset(paths, annotations=[make_annotation("ann-1", source="manual", action="drive")])

    warnings = export_staleness_warnings(
        paths,
        [make_annotation("ann-1", source="manual", action="slice")],
    )

    assert warnings == [
        "Export is stale: 1 exported example(s) differ from current annotations.",
    ]


def test_export_staleness_warnings_detect_added_and_removed_annotations(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    export_dataset(paths, annotations=[make_annotation("ann-1", source="manual")])

    warnings = export_staleness_warnings(
        paths,
        [make_annotation("ann-2", source="manual")],
    )

    assert warnings == [
        "Export is stale: 1 trusted annotation(s) are not exported.",
        "Export is stale: 1 exported example(s) no longer have trusted annotations.",
    ]


def test_export_staleness_warnings_detect_missing_training_examples(tmp_path):
    project = Project(project_id="project-1", video_id="video-1")
    video = Video(
        video_id="video-1",
        source_type=SourceType.LOCAL,
        local_path="videos/source.mp4",
        fps=30,
        duration_ms=2000,
    )
    paths = create_project_layout(tmp_path, project, video)
    export_dataset(paths, annotations=[make_annotation("ann-1", source="manual")])
    (paths.root / TRAINING_EXAMPLES_REF).unlink()

    assert export_staleness_warnings(paths, [make_annotation("ann-1", source="manual")]) == [
        "Export manifest exists, but exports/latest/training_examples.jsonl is missing.",
    ]


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
