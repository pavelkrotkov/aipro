"""In-memory fake GitHubClient for deterministic testing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_pr_orchestrator.github import models


class StaleEditError(Exception):
    """Raised when edit_comment is called with a stale updated_at timestamp."""


@dataclass
class _MutableComment:
    id: int
    issue_number: int
    body: str
    user: str
    created_at: datetime
    updated_at: datetime

    def to_model(self) -> models.Comment:
        return models.Comment(
            id=self.id,
            body=self.body,
            user=self.user,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
        )


@dataclass
class _MutableThread:
    id: str
    pr_number: int
    is_resolved: bool
    is_outdated: bool
    path: str
    comments: list[models.ReviewComment] = field(default_factory=list)

    def to_model(self) -> models.ReviewThread:
        return models.ReviewThread(
            id=self.id,
            is_resolved=self.is_resolved,
            is_outdated=self.is_outdated,
            path=self.path,
            comments=list(self.comments),
        )


@dataclass
class _MutableCheckRun:
    id: int
    ref: str
    name: str
    status: str
    conclusion: str | None
    html_url: str = ""

    def to_model(self) -> models.CheckRun:
        return models.CheckRun(
            id=self.id,
            name=self.name,
            status=self.status,
            conclusion=self.conclusion,
            html_url=self.html_url,
        )


class FakeGitHubClient:
    """In-memory implementation of the GitHubClient protocol for tests."""

    def __init__(self, *, now: datetime | None = None, page_size: int = 30) -> None:
        self._now = now or datetime.now(UTC)
        self._page_size = page_size
        self._prs: dict[int, models.PullRequest] = {}
        self._comments: dict[int, _MutableComment] = {}
        self._labels: dict[int, list[str]] = {}
        self._threads: dict[str, _MutableThread] = {}
        self._check_runs: dict[str, list[_MutableCheckRun]] = {}
        self._commit_statuses: dict[str, list[models.CheckRun]] = {}
        self._reviews: dict[int, list[models.Review]] = {}
        self._next_comment_id = 1
        self._next_check_run_id = 1
        self._next_review_comment_id = 1
        self._next_review_id = 1

    def _tick(self) -> datetime:
        """Advance internal clock by 1 second and return the new time."""
        self._now += timedelta(seconds=1)
        return self._now

    # --- Seeding helpers ---

    def seed_pr(self, pr: models.PullRequest) -> None:
        self._prs[pr.number] = pr
        self._labels[pr.number] = list(pr.labels)

    def seed_comment(
        self,
        issue_number: int,
        body: str,
        *,
        user: str = "test-user",
        comment_id: int | None = None,
        created_at: datetime | None = None,
    ) -> models.Comment:
        cid = comment_id if comment_id is not None else self._next_comment_id
        if cid >= self._next_comment_id:
            self._next_comment_id = cid + 1
        ts = created_at or self._now
        mc = _MutableComment(
            id=cid,
            issue_number=issue_number,
            body=body,
            user=user,
            created_at=ts,
            updated_at=ts,
        )
        self._comments[cid] = mc
        return mc.to_model()

    def seed_thread(
        self,
        thread_id: str,
        pr_number: int,
        *,
        path: str = "file.py",
        is_resolved: bool = False,
        is_outdated: bool = False,
        comments: list[models.ReviewComment] | None = None,
    ) -> models.ReviewThread:
        mt = _MutableThread(
            id=thread_id,
            pr_number=pr_number,
            is_resolved=is_resolved,
            is_outdated=is_outdated,
            path=path,
            comments=list(comments) if comments else [],
        )
        self._threads[thread_id] = mt
        return mt.to_model()

    def seed_check_run(
        self,
        ref: str,
        name: str,
        status: str,
        conclusion: str | None = None,
        *,
        check_run_id: int | None = None,
        html_url: str = "",
    ) -> models.CheckRun:
        crid = check_run_id if check_run_id is not None else self._next_check_run_id
        if crid >= self._next_check_run_id:
            self._next_check_run_id = crid + 1
        mcr = _MutableCheckRun(
            id=crid,
            ref=ref,
            name=name,
            status=status,
            conclusion=conclusion,
            html_url=html_url,
        )
        self._check_runs.setdefault(ref, []).append(mcr)
        return mcr.to_model()

    def seed_commit_status(
        self,
        ref: str,
        context: str,
        status: str,
        conclusion: str | None = None,
    ) -> models.CheckRun:
        """Seed a Statuses-API context already adapted to CheckRun shape, as
        ``get_commit_statuses`` returns it to the runner."""
        cr = models.CheckRun(
            id=models.stable_check_run_id(context),
            name=context,
            status=status,
            conclusion=conclusion,
        )
        self._commit_statuses.setdefault(ref, []).append(cr)
        return cr

    def seed_review(
        self,
        pr_number: int,
        *,
        author: str,
        body: str = "",
        state: str = "COMMENTED",
        submitted_at: datetime | None = None,
    ) -> models.Review:
        review = models.Review(
            id=self._next_review_id,
            author=author,
            body=body,
            state=state,
            submitted_at=(submitted_at or self._now).isoformat(),
        )
        self._next_review_id += 1
        self._reviews.setdefault(pr_number, []).append(review)
        return review

    # --- Protocol implementation ---

    def get_pr(self, number: int) -> models.PullRequest:
        if number not in self._prs:
            raise KeyError(f"PR #{number} not found in fake")
        pr = self._prs[number]
        current_labels = self._labels.get(number, [])
        if list(pr.labels) != current_labels:
            return replace(pr, labels=list(current_labels))
        return pr

    def get_pr_files(self, pr_number: int) -> list[str]:
        if pr_number not in self._prs:
            raise KeyError(f"PR #{pr_number} not found in fake")
        return list(self._prs[pr_number].changed_files)

    def get_pr_comments(self, issue_number: int) -> list[models.Comment]:
        return [
            mc.to_model()
            for mc in sorted(self._comments.values(), key=lambda c: c.id)
            if mc.issue_number == issue_number
        ]

    def post_comment(self, issue_number: int, body: str) -> models.Comment:
        cid = self._next_comment_id
        self._next_comment_id += 1
        ts = self._tick()
        mc = _MutableComment(
            id=cid,
            issue_number=issue_number,
            body=body,
            user="fake-bot",
            created_at=ts,
            updated_at=ts,
        )
        self._comments[cid] = mc
        return mc.to_model()

    def edit_comment(self, comment_id: int, body: str) -> models.Comment:
        if comment_id not in self._comments:
            raise KeyError(f"Comment #{comment_id} not found in fake")
        mc = self._comments[comment_id]
        mc.body = body
        mc.updated_at = self._tick()
        return mc.to_model()

    def edit_comment_optimistic(
        self, comment_id: int, body: str, *, expected_updated_at: str
    ) -> models.Comment:
        if comment_id not in self._comments:
            raise KeyError(f"Comment #{comment_id} not found in fake")
        mc = self._comments[comment_id]
        if mc.updated_at.isoformat() != expected_updated_at:
            raise StaleEditError(
                f"Comment #{comment_id} updated_at mismatch: "
                f"expected {expected_updated_at}, got {mc.updated_at.isoformat()}"
            )
        mc.body = body
        mc.updated_at = self._tick()
        return mc.to_model()

    def add_label(self, issue_number: int, label: str) -> list[dict[str, Any]]:
        labels = self._labels.setdefault(issue_number, [])
        if label not in labels:
            labels.append(label)
        return [{"name": lbl} for lbl in labels]

    def remove_label(self, issue_number: int, label: str) -> None:
        labels = self._labels.get(issue_number, [])
        if label in labels:
            labels.remove(label)

    def get_check_runs(self, ref: str) -> list[models.CheckRun]:
        return [mcr.to_model() for mcr in self._check_runs.get(ref, [])]

    def get_commit_statuses(self, ref: str) -> list[models.CheckRun]:
        return list(self._commit_statuses.get(ref, []))

    def get_pull_request_reviews(self, pr_number: int) -> list[models.Review]:
        return list(self._reviews.get(pr_number, []))

    def get_review_threads(self, pr_number: int) -> list[models.ReviewThread]:
        return [mt.to_model() for mt in self._threads.values() if mt.pr_number == pr_number]

    def reply_to_review_thread(self, thread_id: str, body: str) -> dict[str, Any] | None:
        if thread_id not in self._threads:
            raise KeyError(f"Thread {thread_id} not found in fake")
        mt = self._threads[thread_id]
        rc_id = f"RC_{self._next_review_comment_id}"
        self._next_review_comment_id += 1
        rc = models.ReviewComment(
            id=rc_id,
            body=body,
            author="fake-bot",
            path=mt.path,
            created_at=self._tick().isoformat(),
        )
        mt.comments.append(rc)
        return {"comment": {"id": rc_id}}

    def resolve_review_thread(self, thread_id: str) -> dict[str, Any] | None:
        if thread_id not in self._threads:
            raise KeyError(f"Thread {thread_id} not found in fake")
        self._threads[thread_id].is_resolved = True
        return {"thread": {"id": thread_id, "isResolved": True}}

    def set_check_runs(self, ref: str, runs: list[models.CheckRun]) -> None:
        self._check_runs[ref] = [
            _MutableCheckRun(
                id=run.id,
                ref=ref,
                name=run.name,
                status=run.status,
                conclusion=run.conclusion,
                html_url=run.html_url,
            )
            for run in runs
        ]
