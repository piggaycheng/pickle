from __future__ import annotations

from collections.abc import Iterable

from .schema import IdentitySession, Player, Track
from .storage import ProjectPaths, append_jsonl, read_jsonl


class IdentityValidationError(ValueError):
    pass


def identity_sessions_path(paths: ProjectPaths):
    return paths.state_dir / "identity_sessions.jsonl"


def tracks_path(paths: ProjectPaths):
    return paths.state_dir / "tracks.jsonl"


def append_identity_session(paths: ProjectPaths, session: IdentitySession) -> None:
    append_jsonl(identity_sessions_path(paths), session)


def load_identity_sessions(paths: ProjectPaths) -> list[IdentitySession]:
    return read_jsonl(identity_sessions_path(paths), IdentitySession)


def append_track(paths: ProjectPaths, track: Track) -> None:
    append_jsonl(tracks_path(paths), track)


def load_tracks(paths: ProjectPaths) -> list[Track]:
    return read_jsonl(tracks_path(paths), Track)


def resolve_identity_at(
    sessions: Iterable[IdentitySession],
    *,
    track_id: str,
    timestamp_ms: int,
    include_invalidated: bool = False,
) -> IdentitySession | None:
    matches = [
        session
        for session in sessions
        if session.track_id == track_id
        and session.start_ms <= timestamp_ms < session.end_ms
        and (include_invalidated or not session.invalidated_by_cut)
    ]
    if not matches:
        return None
    return matches[-1]


def validate_identity_players(sessions: Iterable[IdentitySession], players: Iterable[Player]) -> None:
    player_ids = {player.player_id for player in players}
    unknown = sorted({session.player_id for session in sessions if session.player_id not in player_ids})
    if unknown:
        raise IdentityValidationError(f"identity sessions reference unknown players: {', '.join(unknown)}")


def validate_identity_tracks(sessions: Iterable[IdentitySession], tracks: Iterable[Track]) -> None:
    track_ids = {track.track_id for track in tracks}
    unknown = sorted({session.track_id for session in sessions if session.track_id not in track_ids})
    if unknown:
        raise IdentityValidationError(f"identity sessions reference unknown tracks: {', '.join(unknown)}")

