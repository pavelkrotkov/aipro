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
        # New #50 extension fields default; they are omitted on write so
        # legacy payloads stay compact (finding round-1 fix #7).
        data = finding.to_dict()
        assert "confidence" not in data
        assert "head_sha" not in data
        assert "claim" not in data
        assert "evidence" not in data
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
            "f1",
            "fix",
            rationale="real bug",
            decided_by="foreman",
            thread_id="PRRT_1",
            reply_body="fixed in 1234abc",
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


class TestRound1Fixes:
    """Regression tests for codex review round-1 findings on the #50 schema."""

    # --- #1: legacy path validation tolerated on deserialization -----------
    def test_legacy_permissive_path_loads_leniently(self) -> None:
        legacy = make_finding().to_dict()
        legacy["path"] = "/abs/legacy.py"  # strict #50 rules reject absolutes
        finding = ReviewerFinding.from_dict(legacy)
        assert finding.path == "/abs/legacy.py"
        # ...but strict construction still rejects it.
        with pytest.raises(DomainError):
            make_finding(path="/abs/legacy.py")

    # --- #2: missing head_sha quarantined, explicit legacy compat ----------
    def test_unknown_head_sha_quarantined_by_default(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        assert registry.register(make_finding(head_sha=None)) is None
        assert registry.findings == []
        assert len(registry.quarantined) == 1
        assert "no head_sha" in registry.quarantined[0].reason

    def test_unknown_head_sha_legacy_compat_opt_in(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD, quarantine_unknown_head_sha=False)
        assert registry.register(make_finding(head_sha=None)) is not None
        assert len(registry.findings) == 1

    # --- #3: reject dedup across lifecycle states --------------------------
    def test_dedup_across_lifecycle_states_rejected(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(
            make_finding(id="a", claim="null deref", status="accepted", status_reason="saw it")
        )
        registry.register(make_finding(id="b", claim="null deref"))
        with pytest.raises(DomainError, match="across lifecycle states"):
            registry.deduplicate()

    # --- #4: conflicts merge into connected components ---------------------
    def test_conflict_connected_component_single_group(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", lane="review-a", claim="is a bug", line=10))
        registry.register(make_finding(id="b", lane="review-b", claim="works", line=10))
        registry.register(make_finding(id="c", lane="review-c", claim="flaky", line=10))
        groups = registry.detect_conflicts()
        assert len(groups) == 1
        (group_id,) = groups
        assert sorted(groups[group_id]) == ["a", "b", "c"]
        # Every member carries one consistent group id.
        for finding in registry.findings:
            assert finding.conflict_group_id == group_id

    # --- #5: same-lane findings never conflict -----------------------------
    def test_same_lane_findings_not_conflicted(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", lane="review-a", claim="is a bug", line=10))
        registry.register(make_finding(id="b", lane="review-a", claim="intended", line=10))
        assert registry.detect_conflicts() == {}

    # --- #6: evidence kind-specific validation -----------------------------
    def test_evidence_file_requires_path(self) -> None:
        with pytest.raises(DomainError, match="code path"):
            Evidence(kind="file", text="deref")

    def test_evidence_snippet_requires_path_and_text(self) -> None:
        with pytest.raises(DomainError, match="snippet text"):
            Evidence(kind="snippet", path="src/app.py")
        Evidence(kind="snippet", path="src/app.py", snippet="x = 1")

    def test_evidence_command_log_require_text(self) -> None:
        with pytest.raises(DomainError, match="non-blank text"):
            Evidence(kind="command")
        with pytest.raises(DomainError, match="non-blank text"):
            Evidence(kind="log", text="   ")
        Evidence(kind="command", text="pytest -q")
        Evidence(kind="log", text="output")

    # --- #8: reserved separator rejected in key components -----------------
    def test_separator_injection_rejected(self) -> None:
        with pytest.raises(DomainError, match="separator"):
            compute_finding_id(head_sha=HEAD, lane="review-a", claim="a\x1fb")
        with pytest.raises(DomainError, match="separator"):
            compute_finding_id(head_sha=HEAD, lane="review-a", claim="x", path="src/\x1fapp")
        with pytest.raises(DomainError, match="separator"):
            finding_dedup_key(make_finding(claim="a\x1fb"))

    # --- #9: duplicate finding ids rejected at registration ----------------
    def test_duplicate_finding_id_rejected(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        with pytest.raises(DomainError, match="already registered"):
            registry.register(make_finding(id="f1", claim="different"))

    # --- #10: hydrated registries seed provenance in deduplicate -----------
    def test_hydrated_registry_seeds_sources(self) -> None:
        registry = FindingRegistry(findings=[make_finding(id="f1", claim="x", sources=[])])
        merged = registry.deduplicate()
        assert merged[0].sources and merged[0].sources[0].finding_id == "f1"

    # --- #11 (rebutted): conflicts are advisory, never auto-resolved -------
    def test_conflicts_are_advisory_not_auto_resolved(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        a = make_finding(id="a", lane="review-a", claim="is a bug", line=10)
        b = make_finding(id="b", lane="review-b", claim="intended", line=10)
        registry.register(a)
        registry.register(b)
        registry.detect_conflicts()
        # Both findings remain present, open, and fully intact for adjudication.
        assert len(registry.findings) == 2
        assert all(f.status == "open" for f in registry.findings)
        assert {f.id for f in registry.findings} == {"a", "b"}

    # --- #12: require coder reply before resolve ---------------------------
    def test_threaded_disposition_requires_reply(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", thread_id="PRRT_1"))
        with pytest.raises(DomainError, match="reply_body"):
            registry.apply_disposition("f1", "fix", rationale="r", decided_by="foreman")
        updated, _ = registry.apply_disposition(
            "f1", "fix", rationale="r", decided_by="foreman", reply_body="done"
        )
        assert updated.status == "accepted"

    def test_reply_policy_off_when_disabled(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD, require_coder_reply_before_resolve=False)
        registry.register(make_finding(id="f1", claim="x", thread_id="PRRT_1"))
        updated, _ = registry.apply_disposition("f1", "fix", rationale="r", decided_by="foreman")
        assert updated.status == "accepted"

    # --- #13: evidence unknown nested fields preserved in extras -----------
    def test_evidence_unknown_fields_preserved(self) -> None:
        payload = Evidence(kind="file", path="src/app.py").to_dict()
        payload["future_meta"] = {"k": 1}
        evidence = Evidence.from_dict(payload)
        assert evidence.extras == {"future_meta": {"k": 1}}
        assert evidence.to_dict()["future_meta"] == {"k": 1}

    # --- #14: blank claim falls back to body for dedup ---------------------
    def test_blank_claim_not_universal_dedup_key(self) -> None:
        a = make_finding(id="a", claim="   ", body="foo")
        b = make_finding(id="b", claim="", body="bar")
        assert finding_dedup_key(a) != finding_dedup_key(b)
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(a)
        registry.register(b)
        assert len(registry.deduplicate()) == 2

    # --- #15: deferred findings stay active and re-settlable ---------------
    def test_deferred_finding_not_archived_and_re_settleable(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        registry.apply_disposition(
            "f1", "reply_deferred", rationale="next round", decided_by="foreman"
        )
        active, archived = registry.compact()
        assert archived == []
        assert [f.id for f in active] == ["f1"]
        updated, _ = registry.apply_disposition(
            "f1", "fix", rationale="now fixed", decided_by="foreman"
        )
        assert updated.status == "accepted"

    # --- #16: line_end requires line / line_start --------------------------
    def test_line_end_requires_line(self) -> None:
        with pytest.raises(DomainError, match="line_end requires line"):
            make_finding(line=None, line_end=12)
        with pytest.raises(DomainError, match="line_end requires line_start"):
            Evidence(kind="file", path="src/app.py", line_end=12)

    # --- #17: naive/aware created_at compare in dedup ----------------------
    def test_dedup_handles_mixed_naive_aware_created_at(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", claim="null deref", created_at=datetime(2026, 8, 1)))
        registry.register(
            make_finding(
                id="b",
                claim="null deref",
                created_at=datetime(2026, 8, 2, tzinfo=UTC),
            )
        )
        merged = registry.deduplicate()
        assert len(merged) == 1

    # --- #18: reviewer markdown escaped in summary -------------------------
    def test_render_summary_escapes_reviewer_markdown(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", body="a *b* [c](http://x) `d`"))
        text = render_findings_summary(registry.findings)
        assert "*b*" not in text
        assert "[c]" not in text
        assert "\\*b\\*" in text
        assert "\\[c\\]" in text

    def test_render_summary_escapes_path_backtick(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", path="src/`evil`.py", body="ok"))
        text = render_findings_summary(registry.findings)
        assert "src/`evil`.py" not in text  # naked backtick would break the code span
        assert "src/\\`evil\\`.py" in text


# --- Review round 2 ---------------------------------------------------------


class TestReviewRound2:
    def _conflict_registry(self) -> FindingRegistry:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="a", lane="review-a", claim="is a bug", line=10))
        registry.register(make_finding(id="b", lane="review-b", claim="intended", line=10))
        return registry

    # --- round-2 fix #1 -----------------------------------------------------
    def test_hydrated_registry_rejects_duplicate_ids(self) -> None:
        with pytest.raises(DomainError, match="already registered"):
            FindingRegistry(
                findings=[
                    make_finding(id="dup", lane="review-a", claim="x"),
                    make_finding(id="dup", lane="review-b", claim="y"),
                ]
            )

    # --- round-2 fix #2 -----------------------------------------------------
    def test_conflict_group_ids_differ_for_separator_ambiguous_members(self) -> None:
        registry = self._conflict_registry()
        registry.findings[0].id = "a"  # explicit; ids are arbitrary strings
        # Two member sets whose plain-separator join collides must not share a
        # group id: ['a', 'b\x1fc'] vs ['a\x1fb', 'c'].
        registry.findings[0].id = "a"
        registry.findings[1].id = "b\x1fc"
        groups_1 = registry.detect_conflicts()
        registry = self._conflict_registry()
        registry.findings[0].id = "a\x1fb"
        registry.findings[1].id = "c"
        groups_2 = registry.detect_conflicts()
        assert set(groups_1) != set(groups_2)

    # --- round-2 fix #3 -----------------------------------------------------
    def test_provenance_unknown_fields_survive_round_trip(self) -> None:
        payload = FindingProvenance(
            lane="review-a", finding_id="f1", run_id="run-1", round_id="round-1"
        ).to_dict()
        payload["future_meta"] = {"priority": 3}
        provenance = FindingProvenance.from_dict(payload)
        assert provenance.extras == {"future_meta": {"priority": 3}}
        reloaded = FindingProvenance.from_dict(provenance.to_dict())
        assert reloaded.extras == {"future_meta": {"priority": 3}}
        assert reloaded.to_dict()["future_meta"] == {"priority": 3}

    # --- round-2 fix #4 -----------------------------------------------------
    def test_disposition_thread_id_defaults_from_finding(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", thread_id="PRRT_1"))
        _, disposition = registry.apply_disposition(
            "f1", "fix", rationale="r", decided_by="foreman", reply_body="done"
        )
        assert disposition.thread_id == "PRRT_1"

    def test_disposition_rejects_mismatched_thread_id(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", thread_id="PRRT_1"))
        with pytest.raises(DomainError, match="mismatched thread_id"):
            registry.apply_disposition(
                "f1",
                "fix",
                rationale="r",
                decided_by="foreman",
                thread_id="PRRT_other",
                reply_body="done",
            )

    # --- round-2 fix #5 -----------------------------------------------------
    def test_blank_claim_archive_summary_falls_back_to_body(self) -> None:
        archived = ArchivedFinding.from_finding(
            make_finding(
                id="f1",
                claim="   ",
                body="real body text",
                status="rejected",
                status_reason="reject_wont_fix: stale",
            )
        )
        assert archived.summary == "real body text"

    # --- round-2 fix #6 -----------------------------------------------------
    def test_render_summary_neutralizes_mentions_and_newlines(self) -> None:
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x", body="text\n\n@alice ping"))
        text = render_findings_summary(registry.findings)
        # The reviewer body is flattened onto the single bullet line and its
        # mention is neutralized.
        assert "- **major** — text  \\@alice ping (`src/app.py:10`)" in text
        assert "@alice" not in text.replace("\\@alice", "")

    # --- round-2 fix #7 -----------------------------------------------------
    def test_single_line_canonicalization_in_keys(self) -> None:
        assert compute_finding_id(
            head_sha=HEAD, lane="review-a", claim="x", path="src/app.py", line=10
        ) == compute_finding_id(
            head_sha=HEAD, lane="review-a", claim="x", path="src/app.py", line=10, line_end=10
        )
        assert finding_dedup_key(make_finding(id="a", claim="x", line=10)) == finding_dedup_key(
            make_finding(id="b", claim="x", line=10, line_end=10)
        )
        # A genuinely multi-line finding must still differ.
        assert finding_dedup_key(
            make_finding(id="a", claim="x", line=10, line_end=20)
        ) != finding_dedup_key(make_finding(id="b", claim="x", line=10, line_end=10))

    # --- round-2 fix #8 -----------------------------------------------------
    def test_evidence_dedup_is_order_insensitive(self) -> None:
        e1 = Evidence(kind="file", path="src/app.py", extras={"a": 1, "b": 2})
        e2 = Evidence(kind="file", path="src/app.py", extras={"b": 2, "a": 1})
        a = make_finding(id="a", claim="x", evidence=[e1])
        b = make_finding(id="b", claim="x", evidence=[e2])
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(a)
        registry.register(b)
        merged = registry.deduplicate()
        assert len(merged) == 1
        assert len(merged[0].evidence) == 1

    # --- round-2 fix #9 -----------------------------------------------------
    def test_archive_survives_workflow_state_round_trip(self) -> None:
        from ai_pr_orchestrator.v3.domain import WorkflowState

        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        registry.apply_disposition("f1", "fix", rationale="r", decided_by="foreman")
        _, archived = registry.compact()
        state = WorkflowState(
            work_item_id="wi-1", run_id="run-1", phase="reviewing", archived=list(archived)
        )
        reloaded = WorkflowState.from_dict(state.to_dict())
        assert reloaded.archived == archived
        assert reloaded.archived[0].finding_id == "f1"
        assert reloaded.archived[0].sources and reloaded.archived[0].sources[0].finding_id == "f1"

    # --- round-2 fix #10 ----------------------------------------------------
    def test_deferred_is_not_terminal_but_still_requires_reason(self) -> None:
        from ai_pr_orchestrator.v3.domain import (
            FINDING_STATUSES_REQUIRING_REASON,
            TERMINAL_FINDING_STATUSES,
        )

        assert "deferred" in FINDING_STATUSES_REQUIRING_REASON
        assert "deferred" not in TERMINAL_FINDING_STATUSES
        # Deferred findings are never archived by compact (round-1 fix #15).
        registry = FindingRegistry(current_head_sha=HEAD)
        registry.register(make_finding(id="f1", claim="x"))
        registry.apply_disposition("f1", "reply_deferred", rationale="r", decided_by="foreman")
        active, archived = registry.compact()
        assert [f.id for f in active] == ["f1"]
        assert archived == []
