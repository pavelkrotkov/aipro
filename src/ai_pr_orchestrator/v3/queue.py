"""V3 GitHub issue queue, lease/claim, and authoritative workflow state.

GitHub is the durable source of truth for what work exists and where it stands. A
work item is one GitHub issue; its queue membership is encoded in issue labels, and
its authoritative workflow state lives in *one* designated comment on that issue (the
"state comment") as a machine-readable block delimited by markers. Human-readable
discussion before/after the block is preserved.

The persisted block is a serialized :class:`~ai_pr_orchestrator.v3.domain.WorkflowState`
(run/round/phase, reviewer findings and dispositions) whose ``extras`` carry the
claim/lease attribution (host, branch, worktree, PR, lease expiry, heartbeat). Local
state is a cache only — after a full restart, active work is reconstructable purely
from GitHub labels plus these comments plus live CAO inspection.

Concurrency
-----------
Writes use optimistic concurrency through the ``expected_updated_at`` precondition of
:class:`GitHubWorkflowStateStore.save_state`:

- ``None`` = *create-only*: save only if no state comment exists yet (used for the
  initial claim), else :class:`StateConflictError`.
- a ``datetime`` = *expect-that-version*: save only if the state comment still holds
  that ``updated_at``, else :class:`StateConflictError`.

This prevents two foremen from both claiming one issue and two runs from
silently last-write-winning over each other. Stale leases are recovered *only*
through the explicit :meth:`GitHubIssueQueue.reclaim_expired` path.

Size / compaction
-----------------
The block must stay under ``GitHubQueueConfig.max_state_block_chars``. When a save
would exceed it, the oldest reviewer findings/dispositions are dropped (most recent
last, capped at ``_MAX_KEPT_FINDINGS``) before raising; if it still does not fit, a
:class:`StateBlockTooLargeError` is raised rather than silently truncating state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ..github.protocol import GitHubClient as GitHubClientProtocol
from .config import GitHubQueueConfig
from .domain import (
    VALID_PHASES,
    DomainError,
    GitHubIssueRef,
    WorkflowState,
    WorkItem,
)
from .interfaces import StateConflictError

#: Delimiters for the machine-readable block inside a state comment. Human-readable
#: text around the block is untouched.
_START_MARKER = "aipro-v3-state:start"
_END_MARKER = "aipro-v3-state:end"
_BLOCK_RE = re.compile(
    rf"<!--\s*{re.escape(_START_MARKER)}\s*-->(.*?)<!--\s*{re.escape(_END_MARKER)}\s*-->",
    re.DOTALL,
)

#: Maximum number of reviewer findings/dispositions kept in the live block after a
#: compaction pass. Older entries are dropped (they remain recoverable from GitHub
#: review threads; the block is a working summary, not the audit log).
_MAX_KEPT_FINDINGS = 20

#: Claim/lease keys stored in ``WorkflowState.extras``. ``run_id`` and ``phase`` are
#: NOT included: they are first-class ``WorkflowState`` fields already (the extras
#: field rejects values that collide with validated fields), so the claim carries only
#: the durable attribution that the state does not. ``_REQUIRED_CLAIM_KEYS`` is the
#: subset that must exist for a claim to be recognized at all; the rest are optional
#: (e.g. no PR is known until one is opened).
_REQUIRED_CLAIM_KEYS = ("host_id", "claimed_at", "lease_expires_at")
_CLAIM_KEYS = _REQUIRED_CLAIM_KEYS + ("heartbeat_at", "branch", "worktree", "pr_number")


class QueueError(RuntimeError):
    """Base error for queue/lease/state operations."""


class ClaimConflictError(StateConflictError):
    """Raised when an issue is already actively claimed (or a lease is not stale)."""


class NoActiveClaimError(QueueError):
    """Raised when a state has no claim/lease attribution to heartbeat or abandon."""


class StateBlockTooLargeError(QueueError):
    """Raised when a state block cannot be compacted under the configured bound."""


class MalformedStateError(QueueError):
    """Raised when a state comment cannot be parsed into a :class:`WorkflowState`."""


def work_item_id_of(issue: GitHubIssueRef) -> str:
    """Stable work-item id derived from the authoritative GitHub issue."""

    return issue.slug()


def issue_from_work_item_id(work_item_id: str) -> GitHubIssueRef:
    """Invert :func:`work_item_id_of` (``owner/repo#number``)."""

    slug = work_item_id.rsplit("#", 1)
    if len(slug) != 2 or "/" not in slug[0]:
        raise DomainError(f"Malformed work_item_id {work_item_id!r}, expected owner/repo#number")
    owner, _, repo = slug[0].partition("/")
    if not owner or not repo or "/" in repo:
        raise DomainError(f"Malformed work_item_id {work_item_id!r}, expected owner/repo#number")
    return GitHubIssueRef(owner=owner, repo=repo, number=int(slug[1]))


@dataclass(frozen=True)
class Claim:
    """Durable lease/claim attribution for one work item in one run."""

    run_id: str
    host_id: str
    claimed_at: datetime
    lease_expires_at: datetime
    phase: str = "claiming"
    branch: str | None = None
    worktree: str | None = None
    pr_number: int | None = None
    heartbeat_at: datetime | None = None

    def is_stale(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return now > self.lease_expires_at

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "host_id": self.host_id,
            "claimed_at": _iso(self.claimed_at),
            "lease_expires_at": _iso(self.lease_expires_at),
        }
        if self.branch is not None:
            out["branch"] = self.branch
        if self.worktree is not None:
            out["worktree"] = self.worktree
        if self.pr_number is not None:
            out["pr_number"] = self.pr_number
        if self.heartbeat_at is not None:
            out["heartbeat_at"] = _iso(self.heartbeat_at)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(
            run_id=data.get("run_id", ""),
            host_id=data["host_id"],
            claimed_at=_from_iso(data["claimed_at"]),
            lease_expires_at=_from_iso(data["lease_expires_at"]),
            phase=data.get("phase", "claiming"),
            branch=data.get("branch"),
            worktree=data.get("worktree"),
            pr_number=data.get("pr_number"),
            heartbeat_at=_from_iso(data["heartbeat_at"]) if data.get("heartbeat_at") else None,
        )


def claim_from_state(state: WorkflowState) -> Claim:
    missing = [k for k in _REQUIRED_CLAIM_KEYS if k not in state.extras]
    if missing:
        raise NoActiveClaimError(
            f"State for {state.work_item_id} has no claim/lease attribution (missing {missing})"
        )
    claim = Claim.from_dict(state.extras)
    # run_id and phase are first-class state fields, not extras.
    return replace(claim, run_id=state.run_id, phase=state.phase)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _from_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _embed_block(block: dict[str, Any]) -> str:
    payload = json.dumps(block, sort_keys=True, separators=(",", ":"))
    return f"<!-- {_START_MARKER} -->\n{payload}\n<!-- {_END_MARKER} -->"


def _extract_block(body: str) -> dict[str, Any] | None:
    m = _BLOCK_RE.search(body)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError as exc:
        raise MalformedStateError(f"State comment block is not valid JSON: {exc}") from exc


def _replace_block(body: str, block: dict[str, Any]) -> str:
    """Replace the state block in ``body``, preserving surrounding human text."""

    new_block = _embed_block(block)
    if _BLOCK_RE.search(body):
        return _BLOCK_RE.sub(lambda _m: new_block, body, count=1)
    # No block yet (shouldn't normally happen on update): append a standalone block.
    return f"{body.rstrip()}\n\n{new_block}\n"


class GitHubIssueQueue:
    """GitHub-backed queue + authoritative workflow-state store.

    Implements :class:`~ai_pr_orchestrator.v3.interfaces.GitHubWorkflowStateStore`
    (``load_work_item`` / ``load_state`` / ``save_state``) plus the queue operations:
    claim, heartbeat, phase transitions, list-ready, and stale-lease reconciliation.
    Set ``dry_run=True`` to compute transitions without mutating GitHub.
    """

    def __init__(
        self,
        client: GitHubClientProtocol,
        owner: str,
        repo: str,
        config: GitHubQueueConfig | None = None,
        *,
        host_id: str = "",
        dry_run: bool = False,
    ) -> None:
        self._client = client
        self._cfg = config or GitHubQueueConfig()
        self._owner = owner
        self._repo = repo
        self._host_id = host_id
        self._dry_run = dry_run

    # --- GitHubWorkflowStateStore protocol ---------------------------------

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem:
        return WorkItem(
            id=work_item_id_of(issue), issue=issue, labels=self._client.get_labels(issue.number)
        )

    def load_state(self, work_item_id: str) -> WorkflowState | None:
        issue = issue_from_work_item_id(work_item_id)
        comment = self._find_state_comment(issue.number)
        if comment is None:
            return None
        block = _extract_block(comment.body)
        if block is None:
            return None
        return WorkflowState.from_dict(block)

    def save_state(self, state: WorkflowState, expected_updated_at: datetime | None) -> None:
        """Persist ``state`` with an optimistic-concurrency precondition."""

        issue = issue_from_work_item_id(state.work_item_id)
        block = _compacted(state, self._cfg.max_state_block_chars)
        comment = self._find_state_comment(issue.number)

        if expected_updated_at is None:
            if comment is not None:
                raise StateConflictError(
                    f"State already exists for {issue.slug()}; refusing create-only save"
                )
            if not self._dry_run:
                self._client.post_comment(issue.number, _embed_block(block))
            return

        if comment is None:
            raise StateConflictError(
                f"No existing state for {issue.slug()}; cannot update without a precondition"
            )
        existing_block = _extract_block(comment.body)
        if existing_block is None:
            raise MalformedStateError(f"Existing state comment for {issue.slug()} has no block")
        try:
            existing_updated = WorkflowState.from_dict(existing_block).updated_at
        except DomainError as exc:
            raise MalformedStateError(f"Malformed state for {issue.slug()}: {exc}") from exc
        if existing_updated != expected_updated_at:
            raise StateConflictError(
                f"Optimistic concurrency conflict for {issue.slug()}: "
                f"expected updated_at {expected_updated_at.isoformat()!r}, "
                f"found {existing_updated.isoformat()!r}"
            )
        if not self._dry_run:
            # GitHub's issue-comment PATCH has no If-Match, so this read-then-PATCH
            # is best-effort at the transport level; the version check above catches
            # the common lost-update race and callers are expected to retry.
            new_body = _replace_block(comment.body, block)
            self._client.edit_comment(comment.id, new_body)

    # --- Queue operations --------------------------------------------------

    def list_ready(self) -> list[GitHubIssueRef]:
        numbers = self._client.list_issues_by_label(self._cfg.enabled_label)
        return [GitHubIssueRef(owner=self._owner, repo=self._repo, number=n) for n in numbers]

    def claim(
        self,
        issue: GitHubIssueRef,
        run_id: str,
        *,
        branch: str | None = None,
        worktree: str | None = None,
        pr_number: int | None = None,
        now: datetime | None = None,
    ) -> WorkflowState:
        """Claim ``issue`` for ``run_id`` (create-only). Raises :class:`ClaimConflictError`
        if the issue already carries a state comment."""

        now = now or datetime.now(UTC)
        claim = Claim(
            run_id=run_id,
            host_id=self._host_id,
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=self._cfg.lease_seconds),
            phase="claiming",
            branch=branch,
            worktree=worktree,
            pr_number=pr_number,
        )
        state = WorkflowState(
            work_item_id=work_item_id_of(issue),
            run_id=run_id,
            phase="claiming",
            updated_at=now,
            extras=claim.to_dict(),
        )
        try:
            self.save_state(state, expected_updated_at=None)
        except StateConflictError as exc:
            raise ClaimConflictError(str(exc)) from exc
        self._apply_phase_labels(issue, "claiming")
        return state

    def heartbeat(self, state: WorkflowState, *, now: datetime | None = None) -> WorkflowState:
        """Extend the lease and refresh the heartbeat for ``state``'s claim."""

        claim = claim_from_state(state)
        now = now or datetime.now(UTC)
        new_claim = replace(
            claim,
            lease_expires_at=now + timedelta(seconds=self._cfg.lease_seconds),
            heartbeat_at=now,
        )
        new_state = replace(state, updated_at=now, extras=new_claim.to_dict())
        self.save_state(new_state, expected_updated_at=state.updated_at)
        return new_state

    def transition(
        self,
        issue: GitHubIssueRef,
        state: WorkflowState,
        phase: str,
        *,
        terminal_reason: str | None = None,
        round_id: str | None = None,
    ) -> WorkflowState:
        """Advance a claimed work item to ``phase`` and update its queue label."""

        if phase not in VALID_PHASES:
            raise DomainError(f"Invalid phase {phase!r}")
        new_state = state.transition(
            cast(Any, phase), round_id=round_id, terminal_reason=terminal_reason
        )
        self.save_state(new_state, expected_updated_at=state.updated_at)
        self._apply_phase_labels(issue, phase)
        return new_state

    def complete(
        self, issue: GitHubIssueRef, state: WorkflowState, *, reason: str = "done"
    ) -> WorkflowState:
        return self.transition(issue, state, "done", terminal_reason=reason)

    def fail(self, issue: GitHubIssueRef, state: WorkflowState, *, reason: str) -> WorkflowState:
        return self.transition(issue, state, "failed", terminal_reason=reason)

    def mark_review(self, issue: GitHubIssueRef, state: WorkflowState) -> WorkflowState:
        return self.transition(issue, state, "reviewing")

    def mark_needs_human(
        self, issue: GitHubIssueRef, state: WorkflowState, *, reason: str
    ) -> WorkflowState:
        return self.transition(issue, state, "escalated", terminal_reason=reason)

    def is_claim_stale(self, state: WorkflowState, now: datetime | None = None) -> bool:
        try:
            return claim_from_state(state).is_stale(now or datetime.now(UTC))
        except NoActiveClaimError:
            return True

    def reclaim_expired(
        self,
        issue: GitHubIssueRef,
        stale_state: WorkflowState,
        run_id: str,
        *,
        branch: str | None = None,
        worktree: str | None = None,
        pr_number: int | None = None,
        now: datetime | None = None,
    ) -> WorkflowState:
        """Claim ``issue`` again after its previous lease expired (explicit recovery).

        Fails with :class:`ClaimConflictError` if the existing lease is still valid — a
        live claim can only be recovered through reconciliation, never blindly.
        """

        now = now or datetime.now(UTC)
        if not self.is_claim_stale(stale_state, now):
            raise ClaimConflictError(
                f"Claim for {stale_state.work_item_id} is still active; "
                "cannot reclaim until its lease expires"
            )
        claim = Claim(
            run_id=run_id,
            host_id=self._host_id,
            claimed_at=now,
            lease_expires_at=now + timedelta(seconds=self._cfg.lease_seconds),
            phase="claiming",
            branch=branch,
            worktree=worktree,
            pr_number=pr_number,
        )
        new_state = WorkflowState(
            work_item_id=stale_state.work_item_id,
            run_id=run_id,
            phase="claiming",
            updated_at=now,
            extras=claim.to_dict(),
        )
        self.save_state(new_state, expected_updated_at=stale_state.updated_at)
        self._apply_phase_labels(issue, "claiming")
        return new_state

    # --- Private helpers ---------------------------------------------------

    def _find_state_comment(self, issue_number: int):
        for comment in self._client.get_pr_comments(issue_number):
            if _BLOCK_RE.search(comment.body):
                return comment
        return None

    def _apply_phase_labels(self, issue: GitHubIssueRef, phase: str) -> None:
        target = self._phase_label(phase)
        lifecycle = (
            self._cfg.enabled_label,
            self._cfg.active_label,
            self._cfg.review_label,
            self._cfg.needs_human_label,
            self._cfg.done_label,
            self._cfg.error_label,
        )
        current = set(self._client.get_labels(issue.number))
        for label in lifecycle:
            if target == label:
                if label not in current and not self._dry_run:
                    self._client.add_label(issue.number, label)
            elif label in current and not self._dry_run:
                self._client.remove_label(issue.number, label)

    def _phase_label(self, phase: str) -> str | None:
        mapping = {
            "queued": self._cfg.enabled_label,
            "reviewing": self._cfg.review_label,
            "escalated": self._cfg.needs_human_label,
            "done": self._cfg.done_label,
            "failed": self._cfg.error_label,
        }
        if phase in mapping:
            return mapping[phase]
        # Any other active phase claims the work: active label.
        return self._cfg.active_label


def _compacted(state: WorkflowState, max_chars: int) -> dict[str, Any]:
    """Return ``state``'s serialized block, compacting history if it is too large.

    Compaction drops the *oldest* findings/dispositions first (most recent last),
    one at a time, until the block fits or nothing is left to drop. The block is a
    working summary, not the audit log; full history remains on GitHub review
    threads. If even the emptied state exceeds ``max_chars``, raise rather than
    silently truncating authoritative state.
    """

    block = state.to_dict()
    if len(_embed_block(block)) <= max_chars:
        return block

    findings = list(state.findings)
    dispositions = list(state.dispositions)
    while True:
        # Prefer dropping the oldest of whichever history list is longer.
        if len(findings) >= len(dispositions) and findings:
            findings = findings[1:]
        elif dispositions:
            dispositions = dispositions[1:]
        elif not findings:
            break
        trimmed = replace(
            state,
            findings=findings,
            dispositions=dispositions,
        )
        block = trimmed.to_dict()
        if len(_embed_block(block)) <= max_chars:
            return block
    raise StateBlockTooLargeError(
        f"State block for {state.work_item_id} cannot fit within {max_chars} chars "
        "even after compaction; refusing to truncate authoritative state"
    )
