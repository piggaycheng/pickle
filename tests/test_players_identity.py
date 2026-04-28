import pytest
from pydantic import ValidationError

from pickleball_ai.identity import (
    IdentityValidationError,
    append_identity_session,
    append_track,
    identity_sessions_path,
    load_identity_sessions,
    load_tracks,
    resolve_identity_at,
    tracks_path,
    validate_identity_players,
    validate_identity_tracks,
)
from pickleball_ai.players import (
    UnknownPlayerError,
    default_players,
    load_players,
    players_path,
    validate_annotation_player,
    write_players,
)
from pickleball_ai.schema import (
    Annotation,
    IdentityConfirmedBy,
    IdentitySession,
    Player,
    PlayerRoster,
    Project,
    TimeWindow,
    TimingConfidence,
    Track,
)
from pickleball_ai.storage import create_project_layout


def make_annotation(player_id: str = "A") -> Annotation:
    return Annotation(
        annotation_id="ann-1",
        video_id="video-1",
        event_time_ms=100,
        time_window=TimeWindow(start_ms=90, end_ms=110),
        timing_confidence=TimingConfidence.EXACT,
        player_id=player_id,
        action="drive",
        confidence=1,
        source="manual",
        clip_start_ms=50,
        clip_end_ms=150,
    )


def make_session(
    *,
    identity_session_id: str = "session-1",
    player_id: str = "A",
    track_id: str = "track-1",
    segment_id: str | None = None,
    start_ms: int = 0,
    end_ms: int = 1000,
    invalidated_by_cut: bool = False,
) -> IdentitySession:
    return IdentitySession(
        identity_session_id=identity_session_id,
        player_id=player_id,
        track_id=track_id,
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=1,
        confirmed_by=IdentityConfirmedBy.USER,
        invalidated_by_cut=invalidated_by_cut,
    )


def test_default_players_creates_a_to_d():
    players = default_players()

    assert [player.player_id for player in players] == ["A", "B", "C", "D"]
    assert players[0].display_name == "Player A"


def test_player_roster_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        PlayerRoster(players=[Player(player_id="A", display_name="One"), Player(player_id="A", display_name="Two")])


def test_players_round_trip(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))

    roster = write_players(paths, default_players(2))

    assert players_path(paths).exists()
    assert load_players(paths) == roster.players


def test_annotation_player_must_exist():
    players = default_players(2)

    validate_annotation_player(make_annotation("A"), players)
    with pytest.raises(UnknownPlayerError):
        validate_annotation_player(make_annotation("D"), players)


def test_identity_resolution_latest_session_wins_for_overlap():
    first = make_session(identity_session_id="session-1", player_id="A", start_ms=0, end_ms=1000)
    second = make_session(identity_session_id="session-2", player_id="B", start_ms=500, end_ms=900)

    resolved = resolve_identity_at([first, second], track_id="track-1", timestamp_ms=600)

    assert resolved == second


def test_identity_resolution_ignores_cut_invalidated_session():
    invalidated = make_session(invalidated_by_cut=True)

    assert resolve_identity_at([invalidated], track_id="track-1", timestamp_ms=100) is None
    assert resolve_identity_at([invalidated], track_id="track-1", timestamp_ms=100, include_invalidated=True) == invalidated


def test_identity_resolution_returns_none_when_missing():
    session = make_session(track_id="track-1", start_ms=0, end_ms=1000)

    assert resolve_identity_at([session], track_id="track-2", timestamp_ms=100) is None
    assert resolve_identity_at([session], track_id="track-1", timestamp_ms=1000) is None


def test_identity_session_rejects_invalid_range():
    with pytest.raises(ValidationError):
        make_session(start_ms=1000, end_ms=1000)


def test_identity_validates_referenced_players_and_tracks():
    players = default_players(1)
    tracks = [Track(track_id="track-1", start_ms=0, end_ms=1000, confidence=0.8)]

    validate_identity_players([make_session(player_id="A")], players)
    validate_identity_tracks([make_session(track_id="track-1")], tracks)

    with pytest.raises(IdentityValidationError):
        validate_identity_players([make_session(player_id="B")], players)
    with pytest.raises(IdentityValidationError):
        validate_identity_tracks([make_session(track_id="track-2")], tracks)


def test_identity_and_track_round_trip(tmp_path):
    paths = create_project_layout(tmp_path, Project(project_id="project-1", video_id="video-1"))
    track = Track(track_id="track-1", segment_id="seg-1", start_ms=0, end_ms=1000, confidence=0.8)
    session = make_session(segment_id="seg-1")

    append_track(paths, track)
    append_identity_session(paths, session)

    assert tracks_path(paths).exists()
    assert identity_sessions_path(paths).exists()
    assert load_tracks(paths) == [track]
    assert load_identity_sessions(paths) == [session]
