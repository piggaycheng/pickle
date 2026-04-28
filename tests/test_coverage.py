from pickleball_ai.coverage import (
    annotation_is_allowed,
    append_coverage_event,
    coverage_path,
    project_coverage,
    rebuild_coverage,
)
from pickleball_ai.schema import (
    Annotation,
    CoverageEvent,
    CoverageState,
    Project,
    TimeWindow,
    TimingConfidence,
)
from pickleball_ai.storage import create_project_layout, read_jsonl


def make_annotation(start_ms: int = 100, end_ms: int = 200) -> Annotation:
    return Annotation(
        annotation_id="ann-1",
        video_id="video-1",
        event_time_ms=(start_ms + end_ms) // 2,
        time_window=TimeWindow(start_ms=start_ms, end_ms=end_ms),
        timing_confidence=TimingConfidence.EXACT,
        player_id="A",
        action="drive",
        confidence=1,
        source="manual",
        clip_start_ms=start_ms,
        clip_end_ms=end_ms,
    )


def test_coverage_projection_latest_event_wins_for_overlap():
    reviewed = CoverageEvent(
        coverage_event_id="cov-1",
        start_ms=0,
        end_ms=1000,
        state=CoverageState.REVIEWED,
        reason="first pass",
    )
    skipped = CoverageEvent(
        coverage_event_id="cov-2",
        start_ms=200,
        end_ms=400,
        state=CoverageState.SKIPPED_NON_GAME,
        reason="replay",
    )

    projected = project_coverage([reviewed, skipped])

    assert [(span.start_ms, span.end_ms, span.state) for span in projected] == [
        (0, 200, CoverageState.REVIEWED),
        (200, 400, CoverageState.SKIPPED_NON_GAME),
        (400, 1000, CoverageState.REVIEWED),
    ]


def test_unreviewed_range_does_not_allow_non_draft_annotation():
    coverage = project_coverage(
        [
            CoverageEvent(
                start_ms=0,
                end_ms=1000,
                state=CoverageState.UNREVIEWED,
            )
        ]
    )

    assert not annotation_is_allowed(make_annotation(100, 200), coverage)
    assert annotation_is_allowed(make_annotation(100, 200), coverage, draft=True)


def test_reviewed_and_needs_recheck_allow_annotations():
    reviewed = project_coverage(
        [
            CoverageEvent(start_ms=0, end_ms=500, state=CoverageState.REVIEWED),
            CoverageEvent(start_ms=500, end_ms=1000, state=CoverageState.NEEDS_RECHECK),
        ]
    )

    assert annotation_is_allowed(make_annotation(100, 200), reviewed)
    assert annotation_is_allowed(make_annotation(600, 700), reviewed)


def test_annotation_must_fit_inside_one_allowed_span():
    coverage = project_coverage(
        [
            CoverageEvent(start_ms=0, end_ms=200, state=CoverageState.REVIEWED),
            CoverageEvent(start_ms=200, end_ms=400, state=CoverageState.SKIPPED_NON_GAME),
        ]
    )

    assert not annotation_is_allowed(make_annotation(100, 300), coverage)


def test_rebuild_coverage_writes_materialized_state(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    append_coverage_event(paths, CoverageEvent(start_ms=0, end_ms=1000, state=CoverageState.REVIEWED))
    append_coverage_event(paths, CoverageEvent(start_ms=300, end_ms=600, state=CoverageState.NEEDS_RECHECK))

    rebuilt = rebuild_coverage(paths)

    assert [(span.start_ms, span.end_ms, span.state) for span in rebuilt] == [
        (0, 300, CoverageState.REVIEWED),
        (300, 600, CoverageState.NEEDS_RECHECK),
        (600, 1000, CoverageState.REVIEWED),
    ]
    assert read_jsonl(coverage_path(paths), type(rebuilt[0])) == rebuilt

