"""Tests for the issue-content / PR-create GitHub client extensions (issue #55)."""

from __future__ import annotations

from ai_pr_orchestrator.github.fake import FakeGitHubClient
from ai_pr_orchestrator.github.models import PullRequest


def _pr(number: int, *, state: str = "open") -> PullRequest:
    return PullRequest(
        number=number,
        title=f"PR {number}",
        body="",
        state=state,
        head_sha="sha",
        head_ref=f"branch-{number}",
        base_ref="main",
        author="someone",
    )


def test_get_issue_body_round_trip():
    fake = FakeGitHubClient()
    fake.seed_issue(7, labels=["v3-work"], body="Do the thing")
    assert fake.get_issue_body(7) == "Do the thing"


def test_get_issue_body_missing_issue_is_none():
    fake = FakeGitHubClient()
    assert fake.get_issue_body(404) is None


def test_create_pr_assigns_numbers_and_lists():
    fake = FakeGitHubClient()
    pr = fake.create_pr("feat: x", "body", head="feat/x", base="main")
    assert pr.number == 1
    assert pr.head_ref == "feat/x" and pr.base_ref == "main"
    assert pr.state == "open"
    fake.seed_pr(_pr(50, state="closed"))
    assert [p.number for p in fake.list_open_prs()] == [1]
    second = fake.create_pr("feat: y", "", head="feat/y", base="main")
    assert second.number == 51


def test_fake_has_protocol_methods():
    fake = FakeGitHubClient()
    for name in ("get_issue_body", "create_pr", "list_open_prs"):
        assert callable(getattr(fake, name))
