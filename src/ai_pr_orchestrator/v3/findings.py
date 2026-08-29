"""Structured finding processing: deduplication, conflicts, dispositions.

Pure deterministic policy over :mod:`ai_pr_orchestrator.v3.domain` finding
types (issue #50). No I/O, no LLM calls: exact duplicate structured findings
are identified by a normalized key alone.

Every operation is stable under sorting so results are reproducible across
retries of the same reviewer/head/claim.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any

from .domain import (
    DispositionAction,
    DomainError,
    Evidence,
    FindingDisposition,
    FindingProvenance,
    FindingStatus,
    LaneName,
    ReviewerFinding,
    Severity,
)

#: Higher is more urgent.
SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "minor": 1,
    "major": 2,
    "blocker": 3,
}

#: Disposition action → the finding status it settles into.
ACTION_TO_STATUS: dict[DispositionAction, FindingStatus] = {
    "fix": "accepted",
    "reject_wont_fix": "rejected",
    "already_addressed": "rejected",
    "reply_deferred": "deferred",
    "escalate_human": "deferred",
}

_WS = re.compile(r"\s+")


def normalize_claim(text: str) -> str:
    """Normalize claim text for comparison: collapse whitespace, lowercase."""
    return _WS.sub(" ", text).strip().lower()


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
    raw = "\x1f".join(
        (
            head_sha.strip().lower(),
            lane.strip().lower(),
            normalize_claim(claim),
            path or "",
            str(line or ""),
            str(line_end or ""),
        )
    )
    return "finding-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def finding_dedup_key(finding: ReviewerFinding) -> str:
    """Semantic/positional dedup key: same head, same normalized claim, same
    location. Findings sharing a key are exact duplicates regardless of which
    reviewer lane produced them."""
    claim = finding.claim if finding.claim is not None else finding.body
    raw = "\x1f".join(
        (
            (finding.head_sha or "").strip().lower(),
            normalize_claim(claim),
            finding.path or "",
            str(finding.line or ""),
            str(finding.line_end or ""),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _severity_of(finding: ReviewerFinding) -> int:
    return SEVERITY_RANK[finding.severity]


def _canonical_among(group: list[ReviewerFinding]) -> ReviewerFinding:
    """Pick the canonical representative: highest severity, then earliest
    created_at, then lexicographically smallest id — all deterministic."""
    return sorted(group, key=lambda f: (-_severity_of(f), f.created_at, f.id))[0]


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
class ArchivedFinding:
    """Compact record kept after a terminal disposition, so durable state
    stays bounded once findings settle. Full evidence may be dropped; the
    identity, outcome, and reviewer provenance survive."""

    finding_id: str
    lane: LaneName
    severity: Severity
    status: FindingStatus
    status_reason: str
    summary: str
    sources: list[FindingProvenance] = field(default_factory=list)

    @classmethod
    def from_finding(cls, finding: ReviewerFinding) -> ArchivedFinding:
        if finding.status_reason is None:
            raise DomainError(
                f"finding {finding.id} has no status_reason; cannot archive without a reason"
            )
        sources = finding.sources or [
            FindingProvenance(
                lane=finding.lane,
                finding_id=finding.id,
                run_id=finding.run_id,
                round_id=finding.round_id,
                thread_id=finding.thread_id,
            )
        ]
        return cls(
            finding_id=finding.id,
            lane=finding.lane,
            severity=finding.severity,
            status=finding.status,
            status_reason=finding.status_reason,
            summary=finding.claim if finding.claim is not None else finding.body,
            sources=list(sources),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "lane": self.lane,
            "severity": self.severity,
            "status": self.status,
            "status_reason": self.status_reason,
            "summary": self.summary,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchivedFinding:
        data = dict(data)
        data["sources"] = [
            s if isinstance(s, FindingProvenance) else FindingProvenance.from_dict(s)
            for s in data.get("sources", [])
        ]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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

    def register(self, finding: ReviewerFinding) -> ReviewerFinding | None:
        """Validate and admit one finding; return the stored finding, or
        ``None`` when it was quarantined."""
        if (
            self.current_head_sha
            and finding.head_sha
            and finding.head_sha.strip().lower() != self.current_head_sha.strip().lower()
        ):
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
        self.findings.append(finding)
        return finding

    def deduplicate(self) -> list[ReviewerFinding]:
        """Merge exact duplicates (same dedup key) in place and return the
        surviving findings.

        The canonical representative carries the highest severity (ties
        broken by earliest ``created_at`` then id); merged-in duplicates
        contribute their evidence and provenance, never their deletion.
        """
        groups: dict[str, list[ReviewerFinding]] = {}
        for finding in self.findings:
            groups.setdefault(finding_dedup_key(finding), []).append(finding)
        merged: list[ReviewerFinding] = []
        for key in sorted(groups):
            group = groups[key]
            if len(group) == 1:
                merged.append(group[0])
                continue
            canonical = _canonical_among(group)
            evidence: list[Evidence] = []
            seen_evidence: set[str] = set()
            sources: list[FindingProvenance] = []
            seen_sources: set[str] = set()
            for f in sorted(group, key=lambda f: f.id):
                for item in f.evidence:
                    marker = repr(item.to_dict())
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
        line ranges, incompatible claims (different normalized claims and
        the group spans more than one lane).

        Returns a mapping of conflict group id to finding ids and stamps
        ``conflict_group_id`` on each participating finding. Conflicting
        findings are never merged or dropped — they stay distinct so a
        human/policy can adjudicate.
        """
        by_path: dict[str, list[ReviewerFinding]] = {}
        for finding in self.findings:
            if finding.path and finding.line is not None:
                by_path.setdefault(finding.path, []).append(finding)
        groups: dict[str, list[ReviewerFinding]] = {}
        for path in sorted(by_path):
            bucket = sorted(by_path[path], key=lambda f: f.id)
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    if not _ranges_overlap(a, b):
                        continue
                    claim_a = a.claim if a.claim is not None else a.body
                    claim_b = b.claim if b.claim is not None else b.body
                    if normalize_claim(claim_a) == normalize_claim(claim_b):
                        continue
                    if a.lane == b.lane and len({a.id, b.id}) == 1:
                        continue
                    pair = tuple(sorted((a.id, b.id)))
                    group_key = (
                        "conflict-"
                        + hashlib.sha256("\x1f".join((path, *pair)).encode("utf-8")).hexdigest()[
                            :12
                        ]
                    )
                    groups.setdefault(group_key, [])
                    if a.id not in [f.id for f in groups[group_key]]:
                        groups[group_key].append(a)
                    if b.id not in [f.id for f in groups[group_key]]:
                        groups[group_key].append(b)
        result: dict[str, list[str]] = {}
        for group_id in sorted(groups):
            members = sorted(groups[group_id], key=lambda f: f.id)
            for finding in members:
                finding.conflict_group_id = group_id
            result[group_id] = [f.id for f in members]
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
        if finding.status != "open":
            raise DomainError(
                f"finding {finding_id!r} is already {finding.status!r} and cannot be re-settled"
            )
        disposition = FindingDisposition(
            finding_id=finding_id,
            action=action,
            rationale=rationale,
            decided_by=decided_by,
            thread_id=thread_id,
            reply_body=reply_body,
        )
        updated = replace(
            finding,
            status=ACTION_TO_STATUS[action],
            status_reason=f"{action}: {rationale}",
        )
        self.findings[index] = updated
        return updated, disposition

    def compact(self) -> tuple[list[ReviewerFinding], list[ArchivedFinding]]:
        """Move findings at a terminal status into the compact archive.

        Returns (active findings, newly archived records). Archived records
        keep identity, outcome, summary, and provenance — durable state
        stays bounded without losing the audit trail.
        """
        active: list[ReviewerFinding] = []
        for finding in self.findings:
            if finding.status in ("accepted", "rejected", "deferred", "archived"):
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


def render_findings_summary(findings: list[ReviewerFinding]) -> str:
    """Render a concise human-readable GitHub markdown summary.

    Open and terminal findings are listed grouped by severity (highest
    first); conflicting findings are flagged, never resolved silently.
    """
    lines: list[str] = ["## Reviewer findings", ""]
    ordered = sorted(findings, key=lambda f: (-_severity_of(f), f.path or "", f.line or 0, f.id))
    if not ordered:
        lines.append("No reviewer findings.")
        return "\n".join(lines)
    for finding in ordered:
        location = ""
        if finding.path:
            location = f" (`{finding.path}:{finding.line or '?'}`)"
        lines.append(f"- **{finding.severity}** — {finding.body}{location}")
        if finding.conflict_group_id:
            lines.append(
                f"  - ⚠️ conflicts with another reviewer's finding "
                f"(group `{finding.conflict_group_id}`); needs adjudication"
            )
        if finding.status != "open":
            lines.append(f"  - disposition: **{finding.status}** — {finding.status_reason}")
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
