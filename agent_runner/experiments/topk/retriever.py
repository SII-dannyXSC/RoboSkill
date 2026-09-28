#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "be", "with", "by", "at", "from", "as", "this", "that", "it", "put", "place", "pick", "up"}


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    output: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if len(token) >= 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) >= 2 and token not in STOPWORDS:
            output.append(token)
    return output


@dataclass(frozen=True)
class Result:
    experience_id: str
    score: float
    recall: float
    jaccard: float
    keyword_bonus: float
    overlap: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tasks(tasks: dict[str, dict[str, Any]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(tasks, list):
        return tasks
    return [{"experience_id": str(task_id), "task_id": int(task_id), **task} for task_id, task in tasks.items()]


def score(query: dict[str, Any], candidate: dict[str, Any]) -> Result:
    query_text = f"{query.get('description', '')} {' '.join(query.get('keywords', []))}"
    document_text = f"{candidate.get('description', '')} {' '.join(candidate.get('keywords', []))}"
    q = set(tokenize(query_text))
    d = set(tokenize(document_text))
    overlap = q & d
    recall = len(overlap) / max(1, len(q))
    jaccard = len(overlap) / max(1, len(q | d))
    keyword_hits = sum(1 for value in candidate.get("keywords", []) if str(value).lower() in query_text.lower())
    keyword_bonus = min(0.2, keyword_hits * 0.05)
    value = min(1.0, 0.6 * recall + 0.4 * jaccard + keyword_bonus)
    return Result(
        experience_id=str(candidate["experience_id"]),
        score=round(value, 6),
        recall=round(recall, 6),
        jaccard=round(jaccard, 6),
        keyword_bonus=round(keyword_bonus, 6),
        overlap=sorted(overlap),
    )


def retrieve(query: dict[str, Any], candidates: list[dict[str, Any]], top_k: int, threshold: float) -> tuple[list[Result], list[Result]]:
    ranked = sorted((score(query, candidate) for candidate in candidates), key=lambda item: (-item.score, item.experience_id))
    selected = [item for item in ranked[:top_k] if item.score >= threshold]
    if not selected:
        selected = ranked[:1]
    return selected, ranked
