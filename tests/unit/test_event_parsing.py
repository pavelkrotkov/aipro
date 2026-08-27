"""Tests for GitHub Actions event payload parsing."""

from __future__ import annotations

from ai_pr_orchestrator.runner import parse_event


def test_parse_pull_request_event() -> None:
    event = {"pull_request": {"number": 42, "head": {"sha": "abc"}}}
    parsed = parse_event(event, event_name="pull_request")
    assert parsed.event_type == "pull_request"
    assert parsed.pr_number == 42
    assert parsed.head_sha == "abc"


def test_parse_issue_comment_event_on_pr() -> None:
    event = {
        "issue": {"number": 7, "pull_request": {"url": "https://x"}},
        "comment": {"body": "hi"},
    }
    parsed = parse_event(event, event_name="issue_comment")
    assert parsed.event_type == "issue_comment"
    assert parsed.pr_number == 7


def test_parse_pull_request_review_event() -> None:
    event = {
        "pull_request": {"number": 9, "head": {"sha": "sha9"}},
        "review": {"id": 1},
    }
    parsed = parse_event(event, event_name="pull_request_review")
    assert parsed.event_type == "pull_request_review"
    assert parsed.pr_number == 9
    assert parsed.head_sha == "sha9"


def test_parse_pull_request_review_comment_event() -> None:
    # The webhook carries the full pull_request object, so an inline review
    # comment resolves to the PR the same way a review submission does.
    event = {
        "pull_request": {"number": 11, "head": {"sha": "sha11"}},
        "comment": {"id": 7},
    }
    parsed = parse_event(event, event_name="pull_request_review_comment")
    assert parsed.event_type == "pull_request_review_comment"
    assert parsed.pr_number == 11
    assert parsed.head_sha == "sha11"


def test_parse_check_run_event() -> None:
    event = {
        "check_run": {
            "head_sha": "sha-cr",
            "pull_requests": [{"number": 11}],
        }
    }
    parsed = parse_event(event, event_name="check_run")
    assert parsed.event_type == "check_run"
    assert parsed.pr_number == 11
    assert parsed.head_sha == "sha-cr"


def test_parse_check_suite_event() -> None:
    event = {
        "check_suite": {
            "head_sha": "sha-cs",
            "pull_requests": [{"number": 12}],
        }
    }
    parsed = parse_event(event, event_name="check_suite")
    assert parsed.event_type == "check_suite"
    assert parsed.pr_number == 12
    assert parsed.head_sha == "sha-cs"


def test_parse_status_event() -> None:
    event = {"sha": "sha-status", "state": "success"}
    parsed = parse_event(event, event_name="status")
    assert parsed.event_type == "status"
    assert parsed.head_sha == "sha-status"
    assert parsed.pr_number is None


def test_parse_workflow_dispatch_event() -> None:
    event = {"inputs": {"pr": "33"}}
    parsed = parse_event(event, event_name="workflow_dispatch")
    assert parsed.event_type == "workflow_dispatch"
    assert parsed.pr_number == 33


def test_parse_unknown_event_returns_empty() -> None:
    event = {"foo": "bar"}
    parsed = parse_event(event)
    assert parsed.pr_number is None
    assert parsed.head_sha is None


def test_infer_review_comment_payload_without_hint() -> None:
    # A pull_request_review_comment payload (a ``comment`` key alongside the
    # full ``pull_request`` object) must infer as pull_request_review_comment,
    # not the coarser ``pull_request`` — otherwise the payload-first helper
    # rejects the correct ambient GITHUB_EVENT_NAME hint as contradictory.
    event = {
        "pull_request": {"number": 12, "head": {"sha": "sha12"}},
        "comment": {"id": 3},
    }
    parsed = parse_event(event)
    assert parsed.event_type == "pull_request_review_comment"
    assert parsed.pr_number == 12
    assert parsed.head_sha == "sha12"

    # With the matching ambient hint, the hint is honored (not rejected).
    hinted = parse_event(event, event_name="pull_request_review_comment")
    assert hinted.event_type == "pull_request_review_comment"


def test_infer_event_type_without_hint() -> None:
    pr_event = {"pull_request": {"number": 5}}
    cr_event = {"check_run": {"head_sha": "s", "pull_requests": [{"number": 6}]}}
    assert parse_event(pr_event).pr_number == 5
    assert parse_event(cr_event).pr_number == 6
    assert parse_event(cr_event).head_sha == "s"


def test_issue_comment_without_pr_link_returns_no_pr_number() -> None:
    # issue_comment events on plain issues (not PRs) should not carry a pr_number.
    event = {"issue": {"number": 99}, "comment": {"body": "hi"}}
    parsed = parse_event(event, event_name="issue_comment")
    assert parsed.pr_number is None


# --- Defensive parsing: malformed nested payloads must not raise ---


def test_parse_pull_request_with_non_dict_nested_objects() -> None:
    # ``pull_request``/``head`` arriving as non-dicts must coerce to empty
    # rather than raising AttributeError.
    event = {"pull_request": "oops", "head": ["nope"]}
    parsed = parse_event(event, event_name="pull_request")
    assert parsed.event_type == "pull_request"
    assert parsed.pr_number is None
    assert parsed.head_sha is None


def test_parse_check_run_with_non_list_pull_requests() -> None:
    event = {"check_run": {"head_sha": "deadbeef", "pull_requests": "not-a-list"}}
    parsed = parse_event(event, event_name="check_run")
    assert parsed.head_sha == "deadbeef"
    assert parsed.pr_number is None


def test_parse_workflow_dispatch_with_non_dict_inputs_does_not_raise() -> None:
    event = {"inputs": "not-a-dict"}
    parsed = parse_event(event, event_name="workflow_dispatch")
    assert parsed.pr_number is None


def test_infer_event_name_with_non_dict_inputs_does_not_raise() -> None:
    # Inference must not crash on a non-dict ``inputs``/``issue``.
    parsed = parse_event({"inputs": "x", "issue": 5})
    assert parsed.event_type == "unknown"
