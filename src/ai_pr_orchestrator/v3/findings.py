"""Structured finding processing: deduplication, conflicts, dispositions.

Pure deterministic policy over :mod:`ai_pr_orchestrator.v3.domain` finding
types (issue #50). No I/O, no LLM calls: exact duplicate structured findings
are identified by a normalized key alone.

Every operation is stable under sorting so results are reproducible across
retries of the same reviewer/head/claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import UTC
from typing import Any

from .domain import (
    ArchivedFinding,
    DispositionAction,
    DomainError,
    Evidence,
    FindingDisposition,
    FindingProvenance,
    FindingStatus,
    LaneName,
    ReviewerFinding,
)

#: Higher is more urgent.
SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "minor": 1,
    "major": 2,
    "blocker": 3,
}

#: Disposition action → the finding status it settles into.
#: ``rebut`` is deliberately absent: a coder's rebuttal keeps the finding
#: open until a later reviewer round confirms (accept) or rejects (fix).
#: See PR #73 review thread 5 / issue #87 for the rebuttal-path design.
ACTION_TO_STATUS: dict[DispositionAction, FindingStatus] = {
    "fix": "accepted",
    "reject_wont_fix": "rejected",
    "already_addressed": "rejected",
    "reply_deferred": "deferred",
    "escalate_human": "deferred",
    "accept": "accepted",
}

_WS = re.compile(r"\s+")

#: Reserved join/field separator. Legal inside ``head_sha``/``path``/claim
#: values, so every component fed to a key builder is checked and rejected if
#: it contains the separator — otherwise a crafted payload could make two
#: distinct findings collide on the same key. (finding round-1 fix #8)
_SEP = "\x1f"


def normalize_claim(text: str) -> str:
    """Normalize claim text for comparison: collapse whitespace, lowercase."""
    return _WS.sub(" ", text).strip().lower()


def _claim_of(finding: ReviewerFinding) -> str:
    """The machine-comparable claim, falling back to the body when the claim
    is missing or blank. A blank claim must never become a universal dedup key
    (finding round-1 fix #14)."""
    claim = finding.claim
    if claim is not None and claim.strip():
        return claim
    return finding.body


def _reject_separator(component: str, context: str) -> None:
    if _SEP in component:
        raise DomainError(f"{context} must not contain the reserved separator character")


def _canonical_line_end(line: int | None, line_end: int | None) -> str:
    """Canonical encoding of the line range in a dedup/id key: the schema
    defines ``line_end=None`` as a single-line finding, so ``(line=10,
    line_end=None)`` and ``(line=10, line_end=10)`` must produce the same
    key (finding round-2 fix #7)."""
    if line_end is None or line_end == line:
        return ""
    return str(line_end)


def compute_finding_id(
    *,
    head_sha: str,
    lane: LaneName,
    claim: str,
    path: str | None = None,
    line: int | None = None,
    line_end: int | None = None,
) -> str:
    """Deterministic finding id, stable across retries of the same
    reviewer/head/claim after normalization.

    The id deliberately excludes timestamps, run ids, and round ids: the
    same reviewer reporting the same normalized claim against the same
    head SHA yields the same id.
    """
    _reject_separator(head_sha, "head_sha")
    _reject_separator(lane, "lane")
    _reject_separator(claim, "claim")
    if path is not None:
        _reject_separator(path, "path")
    raw = _SEP.join(
        (
            head_sha.strip().lower(),
            lane.strip().lower(),
            normalize_claim(claim),
            path or "",
            str(line or ""),
            _canonical_line_end(line, line_end),
        )
    )
    return "finding-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def finding_dedup_key(finding: ReviewerFinding) -> str:
    """Semantic/positional dedup key: same head, same normalized claim, same
    location. Findings sharing a key are exact duplicates regardless of which
    reviewer lane produced them."""
    _reject_separator(finding.head_sha or "", "head_sha")
    _reject_separator(_claim_of(finding), "claim")
    if finding.path is not None:
        _reject_separator(finding.path, "path")
    raw = _SEP.join(
        (
            (finding.head_sha or "").strip().lower(),
            normalize_claim(_claim_of(finding)),
            finding.path or "",
            str(finding.line or ""),
            _canonical_line_end(finding.line, finding.line_end),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _severity_of(finding: ReviewerFinding) -> int:
    return SEVERITY_RANK[finding.severity]


def _aware_created_at(finding: ReviewerFinding) -> Any:
    """Normalize a possibly-naive ``created_at`` for comparison.

    Naive datetimes (as constructed directly by callers) would otherwise
    raise ``TypeError`` when compared against aware ones during dedup, which
    would abort the whole pass (finding round-1 fix #17).
    """
    dt = finding.created_at
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _canonical_among(group: list[ReviewerFinding]) -> ReviewerFinding:
    """Pick the canonical representative: highest severity, then earliest
    created_at, then lexicographically smallest id — all deterministic."""
    return sorted(group, key=lambda f: (-_severity_of(f), _aware_created_at(f), f.id))[0]


def _first_present(group: list[ReviewerFinding], attr: str) -> Any:
    for f in sorted(group, key=lambda f: f.id):
        value = getattr(f, attr)
        if value is not None:
            return value
    return None


@dataclass
class QuarantinedFinding:
    """A finding that was set aside rather than applied, with the reason."""

    finding: ReviewerFinding
    reason: str


@dataclass
class FindingRegistry:
    """In-memory registry that validates, deduplicates, and settles findings.

    ``register`` quarantines findings whose ``head_sha`` differs from the
    registry's current head SHA; quarantined findings are never applied and
    never silently dropped. ``deduplicate`` merges exact duplicates while
    preserving every origin's provenance and all evidence. ``detect_conflicts``
    groups contradictory findings over the same code region for explicit
    adjudication instead of collapsing them.
    """

    current_head_sha: str | None = None
    findings: list[ReviewerFinding] = field(default_factory=list)
    quarantined: list[QuarantinedFinding] = field(default_factory=list)
    archived: list[ArchivedFinding] = field(default_factory=list)
    #: When True, a finding with no ``head_sha`` at all is quarantined rather
    #: than silently treated as matching the current head. This is the strict
    #: default; set to False as the explicit legacy-compatibility path for
    #: pre-#50 payloads that predate ``head_sha`` (finding round-1 fix #2).
    quarantine_unknown_head_sha: bool = True
    #: When True, a finding that lives on a GitHub review thread must carry a
    #: coder ``reply_body`` before its disposition settles it (finding
    #: round-1 fix #12).
    require_coder_reply_before_resolve: bool = True

    def __post_init__(self) -> None:
        # Hydrating from persisted state (FindingRegistry(findings=...))
        # bypasses ``register``, so the same admission checks must run here:
        # duplicate ids (a legacy state with two lanes reusing an id would
        # corrupt apply_disposition and detect_conflicts' by-id map — round-2
        # fix #1) and stale/unknown head_sha (persisted findings from a
        # previous PR head must be quarantined, not left active — round-3
        # fix #1).
        seen: set[str] = set()
        admitted: list[ReviewerFinding] = []
        for finding in self.findings:
            if finding.id in seen:
                raise DomainError(f"finding id {finding.id!r} is already registered")
            seen.add(finding.id)
            stored = self._admit(finding)
            if stored is not None:
                admitted.append(stored)
        object.__setattr__(self, "findings", admitted)

    def _admit(self, finding: ReviewerFinding) -> ReviewerFinding | None:
        """Shared admission path for register() and hydration.

        Checks head-SHA quarantine and seeds provenance. Returns the stored
        finding, or ``None`` when the finding was quarantined. Caller is
        responsible for duplicate-id detection (``register`` against live
        state, ``__post_init__`` via the pre-pass ``seen`` set).
        """
        if self.current_head_sha:
            given = finding.head_sha and finding.head_sha.strip().lower()
            current = self.current_head_sha.strip().lower()
            if not given:
                if self.quarantine_unknown_head_sha:
                    self.quarantined.append(
                        QuarantinedFinding(
                            finding=finding,
                            reason=(
                                f"finding has no head_sha to verify against current "
                                f"head {self.current_head_sha!r}"
                            ),
                        )
                    )
                    return None
                # Explicit legacy-compatibility path: pre-#50 findings carry
                # no head_sha; admit them rather than silently treating unknown
                # as current, because the operator opted in.
            elif given != current:
                self.quarantined.append(
                    QuarantinedFinding(
                        finding=finding,
                        reason=(
                            f"finding head_sha {finding.head_sha!r} does not match current "
                            f"head {self.current_head_sha!r}"
                        ),
                    )
                )
                return None
        if not finding.sources:
            finding.sources = [
                FindingProvenance(
                    lane=finding.lane,
                    finding_id=finding.id,
                    run_id=finding.run_id,
                    round_id=finding.round_id,
                    thread_id=finding.thread_id,
                )
            ]
        return finding

    def register(self, finding: ReviewerFinding) -> ReviewerFinding | None:
        """Validate and admit one finding; return the stored finding, or
        ``None`` when it was quarantined."""
        if any(existing.id == finding.id for existing in self.findings):
            raise DomainError(f"finding id {finding.id!r} is already registered")
        stored = self._admit(finding)
        if stored is None:
            return None
        self.findings.append(stored)
        return stored

    def deduplicate(self) -> list[ReviewerFinding]:
        """Merge exact duplicates (same dedup key) in place and return the
        surviving findings.

        The canonical representative carries the highest severity (ties
        broken by earliest ``created_at`` then id); merged-in duplicates
        contribute their evidence and provenance, never their deletion.
        """
        # Hydrated registries (FindingRegistry(findings=...)) bypass
        # ``register``, so seed missing provenance here too (finding round-1
        # fix #10).
        for finding in self.findings:
            if not finding.sources:
                finding.sources = [
                    FindingProvenance(
                        lane=finding.lane,
                        finding_id=finding.id,
                        run_id=finding.run_id,
                        round_id=finding.round_id,
                        thread_id=finding.thread_id,
                    )
                ]
        groups: dict[str, list[ReviewerFinding]] = {}
        for finding in self.findings:
            groups.setdefault(finding_dedup_key(finding), []).append(finding)
        merged: list[ReviewerFinding] = []
        for key in sorted(groups):
            group = groups[key]
            if len(group) == 1:
                merged.append(group[0])
                continue
            statuses = {f.status for f in group}
            if len(statuses) > 1:
                raise DomainError(
                    f"cannot deduplicate findings across lifecycle states {sorted(statuses)}"
                )
            canonical = _canonical_among(group)
            evidence: list[Evidence] = []
            seen_evidence: set[str] = set()
            sources: list[FindingProvenance] = []
            seen_sources: set[str] = set()
            for f in sorted(group, key=lambda f: f.id):
                for item in f.evidence:
                    # Canonical serialization: evidence dicts merge an
                    # ``extras`` bucket, so plain repr() is insertion-order
                    # dependent and identical evidence with differently
                    # ordered keys would both survive (finding round-2 fix #8).
                    marker = json.dumps(item.to_dict(), sort_keys=True, default=repr)
                    if marker not in seen_evidence:
                        seen_evidence.add(marker)
                        evidence.append(item)
                for src in f.sources:
                    if src.finding_id not in seen_sources:
                        seen_sources.add(src.finding_id)
                        sources.append(src)
            merged.append(
                replace(
                    canonical,
                    evidence=evidence,
                    sources=sources,
                    confidence=_max_confidence(group),
                    falsification=_first_present(group, "falsification"),
                    reproduction_command=_first_present(group, "reproduction_command"),
                    suggested_fix=_first_present(group, "suggested_fix"),
                )
            )
        self.findings = merged
        return merged

    def detect_conflicts(self) -> dict[str, list[str]]:
        """Group obviously contradictory findings: same path, overlapping
        line ranges, incompatible claims, and at least two distinct lanes.

        Conflicting pairs that share a finding are merged into connected
        COMPONENTS so every finding in the web carries one consistent
        ``conflict_group_id`` (finding round-1 fix #4), and findings from the
        same lane never conflict with themselves (finding round-1 fix #5).

        Returns a mapping of conflict group id to finding ids and stamps
        ``conflict_group_id`` on each participating finding. Conflicting
        findings are never merged or dropped — they stay distinct so a
        human/policy can adjudicate.
        """
        by_id = {f.id: f for f in self.findings}
        by_path: dict[str, list[ReviewerFinding]] = {}
        for finding in self.findings:
            if finding.path and finding.line is not None:
                by_path.setdefault(finding.path, []).append(finding)
        # Union-find over conflict edges to group findings into connected
        # components regardless of the order pairs are considered.
        parent: dict[str, str] = {}
        edges: list[tuple[str, str]] = []

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        for path in sorted(by_path):
            bucket = sorted(by_path[path], key=lambda f: f.id)
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    if a.lane == b.lane:
                        continue
                    if not _ranges_overlap(a, b):
                        continue
                    if normalize_claim(_claim_of(a)) == normalize_claim(_claim_of(b)):
                        continue
                    edges.append((a.id, b.id))
                    union(a.id, b.id)
        components: dict[str, set[str]] = {}
        for a, b in edges:
            components.setdefault(find(a), set()).update((a, b))
        result: dict[str, list[str]] = {}
        for finding in self.findings:
            finding.conflict_group_id = None
        for root in sorted(components):
            members = sorted(components[root])
            # Length-prefixed member encoding: finding ids are arbitrary
            # non-empty strings and may contain the ``_SEP`` character, so a
            # plain join could make distinct member sets hash identically
            # (['a','b\x1fc'] vs ['a\x1fb','c']); the length prefix keeps each
            # encoded member unambiguous (finding round-2 fix #2).
            encoded = ",".join(f"{len(m)}:{m}" for m in members)
            group_id = "conflict-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
            for member in members:
                by_id[member].conflict_group_id = group_id
            result[group_id] = members
        return result

    def apply_disposition(
        self,
        finding_id: str,
        action: DispositionAction,
        *,
        rationale: str,
        decided_by: LaneName,
        thread_id: str | None = None,
        reply_body: str | None = None,
    ) -> tuple[ReviewerFinding, FindingDisposition]:
        """Settle one finding via an explicit disposition.

        Returns the updated finding and the disposition record. Only
        open findings can be settled; a finding already at a terminal
        status raises :class:`DomainError`.
        """
        index = next((i for i, f in enumerate(self.findings) if f.id == finding_id), None)
        if index is None:
            raise DomainError(f"no finding with id {finding_id!r} is registered")
        finding = self.findings[index]
        if finding.status not in ("open", "deferred"):
            raise DomainError(
                f"finding {finding_id!r} is already {finding.status!r} and cannot be re-settled"
            )
        # The persisted disposition must keep the thread identity: default the
        # caller's ``thread_id`` from the finding and reject a mismatch rather
        # than silently dropping the finding's thread (finding round-2 fix #4).
        if thread_id is not None and finding.thread_id and thread_id != finding.thread_id:
            raise DomainError(
                f"finding {finding_id!r} lives on thread {finding.thread_id!r}; "
                f"disposition supplied mismatched thread_id {thread_id!r}"
            )
        effective_thread_id = thread_id if thread_id is not None else finding.thread_id
        if (
            self.require_coder_reply_before_resolve
            and effective_thread_id
            and not (reply_body or "").strip()
        ):
            raise DomainError(
                f"finding {finding_id!r} lives on a review thread; reply_body is "
                "required before resolving it"
            )
        disposition = FindingDisposition(
            finding_id=finding_id,
            action=action,
            rationale=rationale,
            decided_by=decided_by,
            thread_id=effective_thread_id,
            reply_body=reply_body,
        )
        # ``rebut`` keeps the finding open: a coder's push-back must be
        # reviewable by an independent round, not settled to a terminal
        # status. ``ACTION_TO_STATUS`` deliberately omits ``rebut`` so
        # the lookup below raises if a future caller forgets the
        # special case here (PR #73 review thread 5 / issue #87).
        if action == "rebut":
            new_status: FindingStatus = "open"
        else:
            new_status = ACTION_TO_STATUS[action]
        updated = replace(
            finding,
            status=new_status,
            status_reason=f"{action}: {rationale}",
        )
        self.findings[index] = updated
        return updated, disposition

    def compact(self) -> tuple[list[ReviewerFinding], list[ArchivedFinding]]:
        """Move settled findings into the compact archive.

        Returns (active findings, newly archived records). Archive records
        keep identity, outcome, summary, and provenance — durable state stays
        bounded without losing the audit trail. ``deferred`` findings are NOT
        archived: a deferred finding must stay active (with its evidence) so a
        follow-up disposition can still resolve it later (finding round-1 fix
        #15).
        """
        active: list[ReviewerFinding] = []
        for finding in self.findings:
            if finding.status in ("accepted", "rejected", "archived"):
                self.archived.append(ArchivedFinding.from_finding(finding))
            else:
                active.append(finding)
        self.findings = active
        return active, list(self.archived)


def _max_confidence(group: list[ReviewerFinding]) -> float | None:
    values = [f.confidence for f in group if f.confidence is not None]
    return max(values) if values else None


def _ranges_overlap(a: ReviewerFinding, b: ReviewerFinding) -> bool:
    start_a = a.line or 0
    end_a = a.line_end or start_a
    start_b = b.line or 0
    end_b = b.line_end or start_b
    return start_a <= end_b and start_b <= end_a


#: GitHub markdown special characters escaped in reviewer-controlled text.
#: ``@`` is included so a reviewer body cannot turn its text into a user
#: mention (finding round-2 fix #6).
_MD_ESCAPE = re.compile(r"([\\`*_{}\[\]<>#+\-.!|@])")


def _md_escape(text: str | None) -> str:
    # Normalize line breaks to spaces first: a raw newline would let the
    # reviewer body escape the bullet it is rendered into (finding round-2
    # fix #6). ``@`` is backslash-escaped, which GitHub renders literally
    # without triggering a mention notification.
    collapsed = (text or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return _MD_ESCAPE.sub(r"\\\1", collapsed)


def _md_escape_code(text: str | None) -> str:
    """Escape only the characters that break out of an inline code span
    (backtick/backslash); inline code does not render the other Markdown
    special characters, so a path like ``src/app.py`` stays readable."""
    return (text or "").replace("\\", "\\\\").replace("`", "\\`")


def render_findings_summary(findings: list[ReviewerFinding]) -> str:
    """Render a concise human-readable GitHub markdown summary.

    Open and terminal findings are listed grouped by severity (highest
    first); conflicting findings are flagged, never resolved silently.
    Reviewer-controlled text (bodies, paths, reasons) is Markdown-escaped
    so a finding cannot inject formatting into the posted summary (finding
    round-1 fix #18).
    """
    lines: list[str] = ["## Reviewer findings", ""]
    ordered = sorted(findings, key=lambda f: (-_severity_of(f), f.path or "", f.line or 0, f.id))
    if not ordered:
        lines.append("No reviewer findings.")
        return "\n".join(lines)
    for finding in ordered:
        location = ""
        if finding.path:
            location = f" (`{_md_escape_code(finding.path)}:{finding.line or '?'}`)"
        lines.append(f"- **{finding.severity}** — {_md_escape(finding.body)}{location}")
        if finding.conflict_group_id:
            lines.append(
                f"  - ⚠️ conflicts with another reviewer's finding "
                f"(group `{_md_escape_code(finding.conflict_group_id)}`); needs adjudication"
            )
        if finding.status != "open":
            lines.append(
                f"  - disposition: **{finding.status}** — {_md_escape(finding.status_reason)}"
            )
    return "\n".join(lines)


__all__ = [
    "ACTION_TO_STATUS",
    "SEVERITY_RANK",
    "ArchivedFinding",
    "FindingRegistry",
    "QuarantinedFinding",
    "compute_finding_id",
    "finding_dedup_key",
    "normalize_claim",
    "render_findings_summary",
]
