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

These guarantees are enforced *after* the write as well as before:

- **Create-only arbitration.** GitHub has no uniqueness constraint on issue comments,
  so two concurrent claimants can both observe "no state comment" and both POST. After
  posting, the queue re-scans the issue; if several state comments appeared, the
  earliest-created one wins and every other claim's comment is deleted — all but that
  one claimant raise :class:`ClaimConflictError`, so at most one foreman proceeds.
- **Post-write verification.** GitHub's comment ``PATCH`` has no If-Match, so a
  read-then-PATCH cannot be atomic. After editing, the queue re-reads the comment; if
  a concurrent writer clobbered our version in the window, the re-read detects it and
  raises :class:`StateConflictError` rather than reporting success.

Stale leases are recovered *only* through the explicit
:meth:`GitHubIssueQueue.reclaim_expired` path, which refuses terminal phases and
expired-but-not-yet-stale claims. A live claim can never be revived by a heartbeat
after its lease expires (heartbeats on expired leases are rejected), so a paused
owner cannot outrace reconciliation.

Size / compaction
-----------------
The block must stay under ``GitHubQueueConfig.max_state_block_chars``. When a save
would exceed it, only findings that carry a ``thread_id`` (recoverable from a GitHub
review thread) are dropped, oldest first; *unthreaded* findings live only in this
state and are never silently discarded. If even that compaction cannot fit, a
:class:`StateBlockTooLargeError` is raised rather than truncating authoritative state.

Markers
-------
The block delimiters are derived from :attr:`GitHubQueueConfig.state_comment_marker`
(``<marker>:start`` / ``<marker>:end``), so a deployment can migrate or coexist under a
distinct marker. A comment that contains a bare marker with no well-formed block is
treated as malformed/conflicting state, never as "no state".
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

#: Terminal phases that must never be reopened by lease reconciliation.
_TERMINAL_PHASES = ("done", "failed", "escalated")

#: Claim/lease keys stored in ``WorkflowState.extras``. ``run_id`` and ``phase`` are
#: NOT included: they are first-class ``WorkflowState`` fields already (the extras
#: field rejects values that collide with validated fields), so the claim carries only
#: the durable attribution that the state does not. ``_REQUIRED_CLAIM_KEYS`` is the
#: subset that must exist for a claim to be recognized at all; the rest are optional
#: (e.g. no PR is known until one is opened).
_REQUIRED_CLAIM_KEYS = ("host_id", "claimed_at", "lease_expires_at")


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


class LabelSyncError(QueueError):
    """Raised when phase state committed but its GitHub label transition failed.

    The authoritative state is already persisted; recover by calling
    :meth:`GitHubIssueQueue.repair_labels` after the transient error clears.
    """


def work_item_id_of(issue: GitHubIssueRef) -> str:
    """Stable work-item id derived from the authoritative GitHub issue."""

    return issue.slug()


def _utc(dt: datetime) -> datetime:
    """Return ``dt`` made timezone-aware (UTC). Naive input is assumed UTC."""

    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return _utc(dt).isoformat()


def _from_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


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
        now = _utc(now or datetime.now(UTC))
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


class GitHubIssueQueue:
    """GitHub-backed queue + authoritative workflow-state store.

    Implements :class:`~ai_pr_orchestrator.v3.interfaces.GitHubWorkflowStateStore`
    (``load_work_item`` / ``load_state`` / ``save_state``) plus the queue operations:
    claim, heartbeat, phase transitions, list-ready, stale-lease reconciliation, and
    idempotent label repair. Set ``dry_run=True`` to compute transitions without
    mutating GitHub.
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
        # Derive block delimiters from the configured marker so migrations can use a
        # distinct marker without hard-coded collisions.
        marker = self._cfg.state_comment_marker
        self._start = f"{marker}:start"
        self._end = f"{marker}:end"
        self._block_re = re.compile(
            rf"<!--\s*{re.escape(self._start)}\s*-->(.*?)<!--\s*{re.escape(self._end)}\s*-->",
            re.DOTALL,
        )
        self._any_marker_re = re.compile(
            rf"<!--\s*(?:{re.escape(self._start)}|{re.escape(self._end)})\s*-->",
        )
        self._lifecycle_labels = (
            self._cfg.enabled_label,
            self._cfg.active_label,
            self._cfg.review_label,
            self._cfg.needs_human_label,
            self._cfg.done_label,
            self._cfg.error_label,
        )

    # --- Block (de)serialization ------------------------------------------

    def _embed_block(self, block: dict[str, Any]) -> str:
        payload = json.dumps(block, sort_keys=True, separators=(",", ":"))
        return f"<!-- {self._start} -->\n{payload}\n<!-- {self._end} -->"

    def _extract_block(self, body: str) -> dict[str, Any] | None:
        m = self._block_re.search(body)
        if not m:
            return None
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError as exc:
            raise MalformedStateError(f"State comment block is not valid JSON: {exc}") from exc

    def _replace_block(self, body: str, block: dict[str, Any]) -> str:
        """Replace the state block in ``body``, preserving surrounding human text."""

        new_block = self._embed_block(block)
        if self._block_re.search(body):
            return self._block_re.sub(lambda _m: new_block, body, count=1)
        return f"{body.rstrip()}\n\n{new_block}\n"

    def _has_any_marker(self, body: str) -> bool:
        return bool(self._any_marker_re.search(body))

    # --- GitHubWorkflowStateStore protocol ---------------------------------

    def load_work_item(self, issue: GitHubIssueRef) -> WorkItem:
        return WorkItem(
            id=work_item_id_of(issue), issue=issue, labels=self._client.get_labels(issue.number)
        )

    def load_state(self, work_item_id: str) -> WorkflowState | None:
        issue = self._resolve_issue(work_item_id)
        comment = self._find_state_comment(issue.number)
        if comment is None:
            return None
        block = self._extract_block(comment.body)
        if block is None:
            return None
        return WorkflowState.from_dict(block)

    def save_state(self, state: WorkflowState, expected_updated_at: datetime | None) -> None:
        """Persist ``state`` with an optimistic-concurrency precondition."""

        issue = self._resolve_issue(state.work_item_id)
        block = self._compacted(state)
        comment = self._find_state_comment(issue.number)

        if expected_updated_at is None:
            self._save_create_only(issue, comment, block)
            return
        self._save_update(issue, comment, state, block, expected_updated_at)

    def _save_create_only(
        self,
        issue: GitHubIssueRef,
        comment,
        block: dict[str, Any],
    ) -> None:
        # A bare/lone marker is malformed authoritative state, not "unclaimed": refuse
        # to create a second authoritative comment for an issue that ever had one.
        if any(self._has_any_marker(c.body) for c in self._client.get_pr_comments(issue.number)):
            raise ClaimConflictError(
                f"Issue {issue.slug()} has malformed or existing state markers; "
                "refusing a create-only claim"
            )
        if comment is not None:
            raise ClaimConflictError(
                f"State already exists for {issue.slug()}; refusing create-only claim"
            )
        if self._dry_run:
            return
        created = self._client.post_comment(issue.number, self._embed_block(block))
        # Post-write arbitration: at most one claim comment may remain authoritative.
        self._settle_after_create(issue, created)

    def _save_update(
        self,
        issue: GitHubIssueRef,
        comment,
        state: WorkflowState,
        block: dict[str, Any],
        expected_updated_at: datetime | None,
    ) -> None:
        if comment is None:
            raise StateConflictError(
                f"No existing state for {issue.slug()}; cannot update without a precondition"
            )
        existing_block = self._extract_block(comment.body)
        if existing_block is None:
            raise MalformedStateError(f"Existing state comment for {issue.slug()} has no block")
        try:
            existing_updated = WorkflowState.from_dict(existing_block).updated_at
        except DomainError as exc:
            raise MalformedStateError(f"Malformed state for {issue.slug()}: {exc}") from exc
        if expected_updated_at is None or existing_updated != expected_updated_at:
            raise StateConflictError(
                f"Optimistic concurrency conflict for {issue.slug()}: "
                f"existing updated_at {existing_updated.isoformat()!r} no longer "
                f"matches the expected precondition"
            )
        if self._dry_run:
            return
        new_body = self._replace_block(comment.body, block)
        edited = self._client.edit_comment(comment.id, new_body)
        # Post-write verification: re-read what now owns the comment. If a concurrent
        # writer clobbered our write in the read-edit window, surface a conflict rather
        # than reporting success (GitHub PATCH has no If-Match).
        verify = self._client.get_comment(edited.id) if edited.id else None
        if verify is not None:
            verify_block = self._extract_block(verify.body)
            if verify_block is not None:
                try:
                    verify_updated = WorkflowState.from_dict(verify_block).updated_at
                except DomainError:
                    verify_updated = None
                if verify_updated != state.updated_at:
                    raise StateConflictError(
                        f"Post-write conflict for {issue.slug()}: comment #{edited.id} "
                        f"no longer holds version {state.updated_at.isoformat()!r}"
                    )

    def _settle_after_create(self, issue: GitHubIssueRef, created) -> None:
        """Resolve any duplicate state comments created by a concurrent claim.

        The earliest-created state comment is authoritative; every later duplicate is
        deleted. If our own comment is not the authoritative one, we lose the claim.
        """

        state_comments = [
            c for c in self._client.get_pr_comments(issue.number) if self._block_re.search(c.body)
        ]
        created_id = getattr(created, "id", None)
        if not state_comments:
            # Our own state comment vanished between POST and this rescan (a concurrent
            # claimant's arbitration deleted it). We cannot hold the claim.
            raise ClaimConflictError(
                f"State comment for {issue.slug()} disappeared after posting; "
                "a concurrent claimant's arbitration superseded this claim"
            )
        authoritative = min(state_comments, key=lambda c: getattr(c, "id", 0))
        for dup in state_comments:
            if dup.id != authoritative.id and not self._dry_run:
                self._client.delete_comment(dup.id)
        if created is not None and created_id != authoritative.id:
            raise ClaimConflictError(
                f"Concurrent claim won for {issue.slug()}; this foreman did not "
                "become the authoritative claimant"
            )

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
        """Claim ``issue`` for ``run_id``.

        Raises :class:`ClaimConflictError` if the issue is already actively claimed.
        An issue that was *requeued* (phase ``queued``) has no current owner and is
        claimable through the normal path — the existing state comment is overwritten
        with the new claim via a compare-and-swap on its ``updated_at``.
        """

        now = _utc(now or datetime.now(UTC))
        existing = self.load_state(work_item_id_of(issue))

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
            if existing is None:
                self.save_state(state, expected_updated_at=None)
            elif existing.phase == "queued":
                # Preserve workflow history AND non-claim extras across a fresh claim
                # of requeued work (mixed-version rollouts must not lose fields).
                extras = {k: v for k, v in existing.extras.items() if k not in _REQUIRED_CLAIM_KEYS}
                extras.update(claim.to_dict())
                state = replace(
                    state,
                    round_id=existing.round_id,
                    findings=list(existing.findings),
                    dispositions=list(existing.dispositions),
                    extras=extras,
                )
                self.save_state(state, expected_updated_at=existing.updated_at)
            else:
                raise ClaimConflictError(
                    f"Issue {issue.slug()} is in phase {existing.phase!r} and cannot be claimed"
                )
        except StateConflictError as exc:
            raise ClaimConflictError(str(exc)) from exc
        try:
            self._apply_phase_labels(issue, "claiming")
        except Exception as exc:
            raise LabelSyncError(
                f"Claim for {issue.slug()} committed but label update failed: {exc}"
            ) from exc
        return state

    def heartbeat(self, state: WorkflowState, *, now: datetime | None = None) -> WorkflowState:
        """Extend the lease and refresh the heartbeat for ``state``'s claim.

        Raises :class:`ClaimConflictError` if the lease has already expired — an
        expired lease is recovered only through :meth:`reclaim_expired`, never by a
        paused owner reviving it. Non-claim ``extras`` are preserved.
        """

        claim = claim_from_state(state)
        now = _utc(now or datetime.now(UTC))
        if self._host_id and claim.host_id != self._host_id:
            raise ClaimConflictError(
                f"Lease for {state.work_item_id} is held by host {claim.host_id!r}, "
                f"not {self._host_id!r}; a non-owner cannot heartbeat it"
            )
        if claim.is_stale(now):
            raise ClaimConflictError(
                f"Lease for {state.work_item_id} expired at "
                f"{claim.lease_expires_at.isoformat()}; recover via reclaim_expired"
            )
        new_claim = replace(
            claim,
            lease_expires_at=now + timedelta(seconds=self._cfg.lease_seconds),
            heartbeat_at=now,
        )
        # Merge refreshed claim keys into the existing extras so forward-compatible /
        # adapter-specific keys survive a routine heartbeat.
        extras = dict(state.extras)
        extras.update(new_claim.to_dict())
        new_state = replace(state, updated_at=now, extras=extras)
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
        """Advance a claimed work item to ``phase`` and update its queue label.

        The state is committed first (authoritative). If the label transition then
        fails, :class:`LabelSyncError` is raised but the phase is already persisted;
        call :meth:`repair_labels` to recover without re-running the whole transition.
        """

        if phase not in VALID_PHASES:
            raise DomainError(f"Invalid phase {phase!r}")
        self._require_same_identity(issue, state)
        new_state = state.transition(
            cast(Any, phase), round_id=round_id, terminal_reason=terminal_reason
        )
        self.save_state(new_state, expected_updated_at=state.updated_at)
        try:
            self._apply_phase_labels(issue, phase)
        except Exception as exc:  # state committed; labels need repair, not a full retry
            raise LabelSyncError(
                f"State for {issue.slug()} committed to {phase!r} but label update failed: {exc}"
            ) from exc
        return new_state

    def repair_labels(self, issue: GitHubIssueRef, state: WorkflowState | None = None) -> None:
        """Idempotently repair the issue's lifecycle label to match its *current* phase.

        Reads the authoritative state from GitHub (no optimistic precondition), so it
        can fix a label left behind by a partial failure without needing the caller's
        now-stale handle.
        """

        current = state or self.load_state(work_item_id_of(issue))
        if current is None:
            return
        self._require_same_identity(issue, current)
        self._apply_phase_labels(issue, current.phase)

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
            return claim_from_state(state).is_stale(_utc(now or datetime.now(UTC)))
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

        Fails with :class:`ClaimConflictError` if the existing lease is still valid, and
        refuses to reclaim terminal phases (``done``/``failed``/``escalated``) — completed
        or escalated work is never reopened by reconciliation. Workflow history
        (``round_id``, findings, dispositions, non-claim extras) is preserved across the
        reclamation.
        """

        now = _utc(now or datetime.now(UTC))
        self._require_same_identity(issue, stale_state)
        if stale_state.phase in _TERMINAL_PHASES:
            raise NoActiveClaimError(
                f"State for {stale_state.work_item_id} is terminal "
                f"({stale_state.phase!r}) and cannot be reclaimed"
            )
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
        # Preserve authoritative workflow data; only attribution/lease is replaced.
        extras = {k: v for k, v in stale_state.extras.items() if k not in _REQUIRED_CLAIM_KEYS}
        extras.update(claim.to_dict())
        new_state = WorkflowState(
            work_item_id=stale_state.work_item_id,
            run_id=run_id,
            phase="claiming",
            round_id=stale_state.round_id,
            updated_at=now,
            findings=list(stale_state.findings),
            dispositions=list(stale_state.dispositions),
            extras=extras,
        )
        self.save_state(new_state, expected_updated_at=stale_state.updated_at)
        try:
            self._apply_phase_labels(issue, "claiming")
        except Exception as exc:
            raise LabelSyncError(
                f"Reclaim for {issue.slug()} committed but label update failed: {exc}"
            ) from exc
        return new_state

    # --- Private helpers ---------------------------------------------------

    def _require_same_identity(self, issue: GitHubIssueRef, state: WorkflowState) -> None:
        """Reject split-identity calls where ``issue`` and ``state`` refer to different
        work items — committing state to one issue while mutating another's labels."""

        if work_item_id_of(issue) != state.work_item_id:
            raise DomainError(
                f"Issue {issue.slug()} does not match the state's work item {state.work_item_id!r}"
            )

    def _resolve_issue(self, work_item_id: str) -> GitHubIssueRef:
        """Parse ``work_item_id`` and refuse identities bound to another repository.

        Without this, two repos' ``owner/repo#42`` would alias to a single issue number
        on this client, silently reading/mutating the wrong repo.
        """

        slug = work_item_id.rsplit("#", 1)
        if len(slug) != 2 or "/" not in slug[0]:
            raise DomainError(
                f"Malformed work_item_id {work_item_id!r}, expected owner/repo#number"
            )
        owner, _, repo = slug[0].partition("/")
        if not owner or not repo or "/" in repo:
            raise DomainError(
                f"Malformed work_item_id {work_item_id!r}, expected owner/repo#number"
            )
        if owner != self._owner or repo != self._repo:
            raise DomainError(
                f"work_item_id {work_item_id!r} is bound to {owner}/{repo} but this "
                f"queue is configured for {self._owner}/{self._repo}"
            )
        return GitHubIssueRef(owner=self._owner, repo=self._repo, number=int(slug[1]))

    def _find_state_comment(self, issue_number: int):
        for comment in self._client.get_pr_comments(issue_number):
            if self._block_re.search(comment.body):
                return comment
        return None

    def _apply_phase_labels(self, issue: GitHubIssueRef, phase: str) -> None:
        target = self._phase_label(phase)
        current = set(self._client.get_labels(issue.number))
        for label in self._lifecycle_labels:
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
        return self._cfg.active_label

    def _compacted(self, state: WorkflowState) -> dict[str, Any]:
        """Serialize ``state``'s block, compacting recoverable history if too large.

        Only findings that carry a ``thread_id`` (recoverable from a GitHub review
        thread) are dropped, oldest first. *Unthreaded* findings exist only in this
        state and are never silently discarded. If even that compaction cannot fit,
        raise rather than truncate authoritative state.
        """

        max_chars = self._cfg.max_state_block_chars
        block = state.to_dict()
        if len(self._embed_block(block)) <= max_chars:
            return block

        threaded = [f for f in state.findings if f.thread_id is not None]
        unthreaded = [f for f in state.findings if f.thread_id is None]
        # Note: dispositions reference findings; dropping threaded findings leaves their
        # dispositions orphaned in the working summary, so we only drop from `threaded`.
        while threaded:
            threaded = threaded[1:]  # drop oldest threaded finding
            trimmed = replace(
                state,
                findings=unthreaded + threaded,
                dispositions=list(state.dispositions),
            )
            block = trimmed.to_dict()
            if len(self._embed_block(block)) <= max_chars:
                return block
        raise StateBlockTooLargeError(
            f"State block for {state.work_item_id} cannot fit within {max_chars} chars "
            "even after keeping all unthreaded findings; refusing to truncate "
            "authoritative state"
        )
