"""Tests for the structured finding schema and processing (issue #50)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_pr_orchestrator.v3.domain import (
    DomainError,
    Evidence,
    FindingProvenance,
    ReviewerFinding,
)
from ai_pr_orchestrator.v3.findings import (
    ArchivedFinding,
    FindingRegistry,
    compute_finding_id,
    finding_dedup_key,
    normalize_claim,
    render_findings_summary,
)

HEAD = "abc1234def5678"


def make_finding(**overrides: object) -> ReviewerFinding:
    defaults: dict = {
        "id": "f1",
        "lane": "review-a",
        "body": "Possible null deref",
        "severity": "major",
        "run_id": "run-1",
        "round_id": "round-1",
        "created_at": datetime(2026, 8, 1, tzinfo=UTC),
        "path": "src/app.py",
        "line": 10,
        "head_sha": HEAD,
    }
    defaults.update(overrides)
    return ReviewerFinding(**defaults)


class TestFindingValidation:
    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(DomainError, match="severity"):
            make_finding(severity="critical")

    def test_confidence_bounds(self) -> None:
        make_finding(confidence=0.0)
        make_finding(confidence=1.0)
        with pytest.raises(DomainError, match="confidence"):
            make_finding(confidence=1.5)
        with pytest.raises(DomainError, match="confidence"):
            make_finding(confidence=-0.1)

    def test_malformed_path_rejected(self) -> None:
        for bad in ("/abs/path.py", "../escape.py", "a/../b.py", "src//x.py", ""):
            with pytest.raises(DomainError):
                make_finding(path=bad)

    def test_malformed_lines_rejected(self) -> None:
        with pytest.raises(DomainError, match="line"):
            make_finding(line=0)
        with pytest.raises(DomainError, match="line_end"):
            make_finding(line=5, line_end=4)

    def test_stale_head_sha_quarantined_by_registry(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        stale = make_finding(head_sha="0000000")
        assert registry.register(stale) is None
        assert registry.findings == []
        assert len(registry.quarantined) == 1
        assert "head_sha" in registry.quarantined[0].reason

    def test_matching_head_sha_accepted(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD.upper())
        assert registry.register(make_finding()) is not None
        assert len(registry.findings) == 1

    def test_terminal_status_requires_reason(self) -> None:
        with pytest.raises(DomainError, match="status_reason"):
            make_finding(status="rejected")
        with pytest.raises(DomainError, match="status_reason"):
            make_finding(status="open", status_reason="premature")

    def test_missing_evidence_items_rejected(self) -> None:
        with pytest.raises(DomainError):
            make_finding(evidence=[Evidence(kind="file")])  # nothing to point at

    def test_evidence_kind_validated(self) -> None:
        with pytest.raises(DomainError, match="evidence kind"):
            Evidence(**{"kind": "vibes", "text": "trust me"})  # ty: ignore[invalid-argument-type]

    def test_unknown_fields_preserved_as_extras(self) -> None:
        payload = make_finding().to_dict()
        payload["brand_new_field"] = {"nested": True}
        finding = ReviewerFinding.from_dict(payload)
        assert finding.extras == {"brand_new_field": {"nested": True}}
        assert ReviewerFinding.from_dict(finding.to_dict()) == finding
        assert finding.to_dict()["brand_new_field"] == {"nested": True}


class TestRoundTrip:
    def test_full_round_trip(self) -> None:
        finding = make_finding(
            confidence=0.8,
            claim="null deref when payload is None",
            evidence=[
                Evidence(
                    kind="file",
                    path="src/app.py",
                    line_start=10,
                    line_end=12,
                    snippet="x = p.value",
                ),
                Evidence(kind="thread", thread_id="PRRT_1"),
            ],
            falsification="run test_null_payload",
            reproduction_command="pytest tests/test_app.py::test_null_payload",
            suggested_fix="guard against None",
            conflict_group_id="conflict-abc",
            sources=[
                FindingProvenance(
                    lane="review-a", finding_id="f1", run_id="run-1", round_id="round-1"
                )
            ],
        )
        assert ReviewerFinding.from_dict(finding.to_dict()) == finding

    def test_old_payload_shape_round_trips(self) -> None:
        """Payloads written by the pre-#50 schema (issue #41 fields only)
        must load unchanged — the #43 queue serializes this shape."""
        old_payload = {
            "id": "f9",
            "lane": "review-b",
            "body": "Unclear error message",
            "severity": "minor",
            "run_id": "run-1",
            "round_id": "round-2",
            "created_at": "2026-08-01T00:00:00+00:00",
            "path": "src/cli.py",
            "line": 42,
            "thread_id": "PRRT_9",
        }
        finding = ReviewerFinding.from_dict(old_payload)
        assert finding.id == "f9"
        assert finding.lane == "review-b"
        assert finding.severity == "minor"
        assert finding.thread_id == "PRRT_9"
        assert finding.status == "open"
        assert finding.evidence == []
        assert finding.sources == []
        assert finding.confidence is None
        assert finding.head_sha is None
        # New fields default; serialization adds only known keys.
        data = finding.to_dict()
        assert "confidence" in data and data["confidence"] is None
        assert "extras" not in data


class TestDeterministicIds:
    @staticmethod
    def _fid(
        head_sha: str = HEAD,
        lane: str = "review-a",
        claim: str = "Null  dereference\nwhen payload is None",
        path: str | None = "src/app.py",
        line: int | None = 10,
        line_end: int | None = 12,
    ) -> str:
        return compute_finding_id(
            head_sha=head_sha, lane=lane, claim=claim, path=path, line=line, line_end=line_end
        )

    def test_stable_across_retries(self) -> None:
        assert self._fid() == self._fid()
        assert self._fid() == self._fid(claim="null Dereference when payload IS None")

    def test_varies_with_inputs(self) -> None:
        base = self._fid()
        assert base != self._fid(line=11)
        assert base != self._fid(head_sha="ffffff")
        assert base != self._fid(lane="review-b")

    def test_dedup_key_ignores_reviewer_and_timestamps(self) -> None:
        a = make_finding(id="a", lane="review-a", claim="null deref")
        b = make_finding(
            id="b",
            lane="review-b",
            body="NULL DEREF",
            claim="null deref",
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
        assert finding_dedup_key(a) == finding_dedup_key(b)


class TestDeduplication:
    def test_duplicates_merged_with_provenance_and_evidence(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        a = make_finding(
            id="a",
            lane="review-a",
            claim="null deref",
            severity="major",
            evidence=[Evidence(kind="file", path="src/app.py", line_start=10, text="deref")],
        )
        b = make_finding(
            id="b",
            lane="review-b",
            claim="null deref",
            severity="minor",
            evidence=[
                Evidence(kind="thread", thread_id="PRRT_2"),
                Evidence(kind="file", path="src/app.py", line_start=10, text="deref"),
            ],
            suggested_fix="add a None check",
        )
        registry.register(a)
        registry.register(b)
        merged = registry.deduplicate()
        assert len(merged) == 1
        m = merged[0]
        assert m.severity == "major"  # highest severity wins
        assert {s.lane for s in m.sources} == {"review-a", "review-b"}
        assert {s.finding_id for s in m.sources} == {"a", "b"}
        assert len(m.evidence) == 2  # evidence union, exact pair deduped
        assert m.suggested_fix == "add a None check"

    def test_distinct_findings_not_merged(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", claim="null deref", line=10))
        registry.register(make_finding(id="b", claim="race condition", line=10))
        assert len(registry.deduplicate()) == 2

    def test_no_llm_needed_for_exact_duplicates(self) -> None:
        """Exact duplicates are identified purely by the normalized key."""
        a = make_finding(claim="Off-by-one in loop")
        b = make_finding(lane="review-b", claim="  off-by-one   in LOOP ")
        assert finding_dedup_key(a) == finding_dedup_key(b)


class TestConflicts:
    def test_conflicting_findings_grouped_not_collapsed(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(
            make_finding(
                id="a", lane="review-a", claim="this is a bug", body="This is a bug", line=10
            )
        )
        registry.register(
            make_finding(
                id="b",
                lane="review-b",
                claim="this is intended",
                body="This is intended behavior",
                line=10,
            )
        )
        groups = registry.detect_conflicts()
        assert len(groups) == 1
        (group_id,) = groups
        assert sorted(groups[group_id]) == ["a", "b"]
        by_id = {f.id: f for f in registry.findings}
        assert by_id["a"].conflict_group_id == group_id
        assert by_id["b"].conflict_group_id == group_id
        assert by_id["a"].body != by_id["b"].body  # both preserved verbatim

    def test_agreement_not_flagged_and_disjoint_lines_not_flagged(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", lane="review-a", claim="bug", line=10))
        registry.register(make_finding(id="b", lane="review-b", claim="bug", line=10))
        registry.register(make_finding(id="c", lane="review-c", claim="bug", line=200))
        assert registry.detect_conflicts() == {}

    def test_conflict_ids_deterministic(self) -> None:
        def build() -> dict:
            registry = FindingRegistry(current_head_sha=HEAD)
            registry.register(make_finding(id="a", lane="review-a", claim="is a bug", line=10))
            registry.register(
                make_finding(id="b", lane="review-b", claim="works as intended", line=10)
            )
            return registry.detect_conflicts()

        assert build() == build()


class TestDispositions:
    def test_accept_reject_defer_with_reason(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        registry.register(make_finding(id="f2", claim="y"))
        registry.register(make_finding(id="f3", claim="z"))
        updated, disposition = registry.apply_disposition(
            "f1", "fix", rationale="real bug", decided_by="foreman", thread_id="PRRT_1"
        )
        assert updated.status == "accepted"
        assert "real bug" in (updated.status_reason or "")
        assert disposition.action == "fix"
        assert disposition.finding_id == "f1"
        assert disposition.decided_by == "foreman"

        registry.apply_disposition(
            "f2", "reject_wont_fix", rationale="by design", decided_by="foreman"
        )
        registry.apply_disposition(
            "f3", "reply_deferred", rationale="next round", decided_by="foreman"
        )
        statuses = {f.id: f.status for f in registry.findings}
        assert statuses == {"f1": "accepted", "f2": "rejected", "f3": "deferred"}

    def test_double_disposition_rejected(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1"))
        registry.apply_disposition("f1", "fix", rationale="ok", decided_by="foreman")
        with pytest.raises(DomainError, match="already"):
            registry.apply_disposition(
                "f1", "reject_wont_fix", rationale="no", decided_by="foreman"
            )

    def test_unknown_finding_rejected(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        with pytest.raises(DomainError, match="no finding"):
            registry.apply_disposition("nope", "fix", rationale="r", decided_by="foreman")


class TestCompactionAndSummary:
    def test_terminal_findings_compact_to_bounded_archive(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        registry.register(make_finding(id="f2", claim="y"))
        registry.apply_disposition("f1", "fix", rationale="real", decided_by="foreman")
        active, archived = registry.compact()
        assert [f.id for f in active] == ["f2"]
        assert len(archived) == 1
        record = archived[0]
        assert record.finding_id == "f1"
        assert record.status == "accepted"
        assert record.lane == "review-a"
        assert record.sources and record.sources[0].finding_id == "f1"
        # Compaction is idempotent.
        assert registry.compact() == ([f for f in active], archived)

    def test_archived_record_round_trip(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1"))
        registry.apply_disposition(
            "f1", "reject_wont_fix", rationale="wontfix", decided_by="foreman"
        )
        _, archived = registry.compact()
        assert ArchivedFinding.from_dict(archived[0].to_dict()) == archived[0]

    def test_render_summary(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", lane="review-a", claim="bug", line=10))
        registry.register(make_finding(id="b", lane="review-b", claim="intended", line=10))
        registry.detect_conflicts()
        registry.register(make_finding(id="c", claim="nit", line=99, severity="minor"))
        text = render_findings_summary(registry.findings)
        assert text.startswith("## Reviewer findings")
        assert "blocker" not in text
        assert "conflicts with another reviewer's finding" in text
        assert "`src/app.py:10`" in text
        # A settled finding renders its disposition.
        registry.apply_disposition("c", "fix", rationale="trivial", decided_by="foreman")
        text2 = render_findings_summary(registry.findings)
        assert "disposition: **accepted**" in text2

    def test_render_summary_empty(self) -> None:
        assert "No reviewer findings" in render_findings_summary([])


def test_normalize_claim() -> None:
    assert normalize_claim("  A  B\tC\n") == "a b c"
