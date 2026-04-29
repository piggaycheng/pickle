from __future__ import annotations

from collections.abc import Iterable

from .schema import Annotation, ModelSuggestion
from .storage import ProjectPaths, append_jsonl, read_jsonl


def suggestions_path(paths: ProjectPaths, job_id: str):
    return paths.job_dir(job_id) / "suggestions.jsonl"


def append_model_suggestion(paths: ProjectPaths, suggestion: ModelSuggestion) -> None:
    append_jsonl(suggestions_path(paths, suggestion.job_id), suggestion)


def load_model_suggestions(paths: ProjectPaths, job_id: str) -> list[ModelSuggestion]:
    return read_jsonl(suggestions_path(paths, job_id), ModelSuggestion)


def evaluate_suggestions(
    suggestions: Iterable[ModelSuggestion],
    annotations: Iterable[Annotation],
) -> dict[str, int]:
    total = 0
    matched = 0
    action_matches = 0
    player_matches = 0
    for suggestion in suggestions:
        total += 1
        target_key = f"{suggestion.target_ref.type}:{suggestion.target_ref.id}"
        matches = [
            annotation
            for annotation in annotations
            if annotation.external_refs.get("target_ref") == target_key
        ]
        if not matches:
            continue
        matched += 1
        if any(annotation.action == suggestion.action for annotation in matches):
            action_matches += 1
        if any(annotation.player_id == suggestion.player_id for annotation in matches):
            player_matches += 1
    return {
        "suggestion_count": total,
        "matched_annotation_count": matched,
        "action_match_count": action_matches,
        "player_match_count": player_matches,
        "unmatched_suggestion_count": total - matched,
    }
