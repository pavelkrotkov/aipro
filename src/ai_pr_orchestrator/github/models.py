"""GitHub API response models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def stable_check_run_id(context: str) -> int:
    """Return a process-stable integer id derived from a status context name.

    Commit-status contexts (Statuses API) have no numeric id of their own, but
    we adapt them into ``CheckRun``s which carry one. Python's builtin
    ``hash()`` is randomized per process (``PYTHONHASHSEED``), so the same
    context would yield a different id on each runner invocation; any logic that
    ever keys on the id across events would behave non-deterministically. A
    truncated SHA-1 digest is stable across processes.
    """
    return int(hashlib.sha1(context.encode("utf-8")).hexdigest()[:15], 16)


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    head_sha: str
    head_ref: str
    base_ref: str
    author: str
    draft: bool = False
    mergeable: bool | None = None
    labels: list[str] = field(default_factory=list)
    is_fork: bool = False
    changed_files: list[str] = field(default_factory=list)
    author_association: str = ""


@dataclass(frozen=True)
class Comment:
    id: int
    body: str
    user: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CheckRun:
    id: int
    name: str
    status: str
    conclusion: str | None
    html_url: str = ""


@dataclass(frozen=True)
class ReviewComment:
    id: str
    body: str
    author: str
    path: str
    created_at: str


@dataclass(frozen=True)
class ReviewThread:
    id: str
    is_resolved: bool
    is_outdated: bool
    path: str
    comments: list[ReviewComment] = field(default_factory=list)


@dataclass(frozen=True)
class Review:
    """A submitted pull request review (the review *summary*, distinct from the
    inline review-thread comments). A reviewer can finish with zero inline
    findings by submitting a review body, so this is a first-class response
    signal alongside comments."""

    id: int
    author: str
    body: str
    state: str
    submitted_at: str
