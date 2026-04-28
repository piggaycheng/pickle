from __future__ import annotations

from collections.abc import Iterable

from .schema import Annotation, Player, PlayerRoster
from .storage import ProjectPaths, read_json, write_json


class UnknownPlayerError(ValueError):
    pass


def players_path(paths: ProjectPaths):
    return paths.state_dir / "players.json"


def default_players(count: int = 4) -> list[Player]:
    if count < 1:
        raise ValueError("count must be at least 1")
    return [
        Player(player_id=chr(ord("A") + index), display_name=f"Player {chr(ord('A') + index)}")
        for index in range(count)
    ]


def write_players(paths: ProjectPaths, players: Iterable[Player]) -> PlayerRoster:
    roster = PlayerRoster(players=list(players))
    write_json(players_path(paths), roster)
    return roster


def load_players(paths: ProjectPaths) -> list[Player]:
    return read_json(players_path(paths), PlayerRoster).players


def player_ids(players: Iterable[Player]) -> set[str]:
    return {player.player_id for player in players}


def validate_annotation_player(annotation: Annotation, players: Iterable[Player]) -> None:
    if annotation.player_id not in player_ids(players):
        raise UnknownPlayerError(f"unknown player_id for annotation: {annotation.player_id}")

