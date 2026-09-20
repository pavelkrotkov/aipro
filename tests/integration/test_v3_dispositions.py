"""Real CAO HTTP -> parser -> foreman -> serialized GitHub state decisions."""

import json
from dataclasses import replace

import pytest

from ai_pr_orchestrator.v3.cao import session_name_for
from ai_pr_orchestrator.v3.domain import GitHubIssueRef, WorkflowState
from ai_pr_orchestrator.v3.findings import FindingRegistry
from ai_pr_orchestrator.v3.foreman import ForemanPolicyLoop, _ForemanEscalation
from tests.integration._harness import developer_output, developer_session_name, script_protocol


@pytest.mark.parametrize("proposal", ["fix", "rebut"])
def test_explicit_independent_acceptance_retains_proposal(fake_cao, foreman_harness, proposal):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, proposal=proposal)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2
    assert outcome.review_rounds == 2
    state = queue.load_state("owner/repo#1")
    assert state.findings == []
    assert [a.finding_id for a in state.archived] == ["guard"]
    assert [d.action for d in state.dispositions] == [proposal, "accept"]
    response, acceptance = state.dispositions
    assert response.decided_by == "developer"
    assert acceptance.decided_by == "requirements-reviewer"
    assert acceptance.response_to_round_id == response.round_id == "response-1"
    assert response.rationale == "test_guard proves the premise cannot occur"
    assert len(github.list_open_prs()) == 1
    # Deserialization/object loss retains immutable decision identity and evidence.
    restored = WorkflowState.from_dict(state.to_dict())
    assert (
        ForemanPolicyLoop._merge_dispositions(restored.dispositions, state.dispositions)
        == state.dispositions
    )
    assert loop._pending_proposals(restored) == {}
    assert loop._saved_review_round(restored) == 2
    with pytest.raises(_ForemanEscalation, match="conflicting replay"):
        loop._merge_dispositions(
            restored.dispositions, [replace(response, rationale="changed evidence")]
        )


@pytest.mark.parametrize(
    "reply",
    [
        "[]",
        "clean review",
        '{"findings":[],"dispositions":[]}',
        '{"findings":[],"dispositions":[{"finding_id":"guard","action":"accept","rationale":"proof","response_to_round_id":"response-0"}]}',
        '{"findings":[],"dispositions":[{"finding_id":"foreign","action":"accept","rationale":"proof","response_to_round_id":"response-1"}]}',
        '{"findings":[],"dispositions":[{"finding_id":"guard","action":"accept","rationale":"proof","response_to_round_id":"response-1","decided_by":"developer"}]}',
    ],
)
def test_invalid_or_stale_acceptance_never_gates(fake_cao, foreman_harness, reply):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, reply=reply)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "Invalid disposition output" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert [(f.id, f.status) for f in state.findings] == [("guard", "open")]
    assert [d.action for d in state.dispositions] == ["rebut"]
    assert not state.archived
    assert not github.list_open_prs()


def test_rejected_rebuttal_returns_actual_reason_to_coder(fake_cao, foreman_harness):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, decision="fix")
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"  # Third coder response is missing, never clean.
    state = queue.load_state("owner/repo#1")
    assert [d.action for d in state.dispositions] == ["rebut", "fix"]
    assert state.findings[0].status == "open"
    coder = fake_cao._sessions[developer_session_name(loop.run_id)]
    assert "Independent reproduction confirms the coder evidence" in coder.submitted_messages[2]
    assert not github.list_open_prs()


@pytest.mark.parametrize("action", ["accept", "reply_deferred", "escalate_human"])
def test_coder_cannot_settle_own_finding(fake_cao, foreman_harness, action):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop, proposal=action)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "Invalid disposition output" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state.dispositions == []
    assert state.findings[0].status == "open"
    assert not github.list_open_prs()


def test_proposal_save_failure_prevents_next_review(fake_cao, foreman_harness, monkeypatch):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    save = queue.save_state

    def fail_proposal(state, **kwargs):
        if state.dispositions:
            raise RuntimeError("proposal persistence unavailable")
        return save(state, **kwargs)

    monkeypatch.setattr(queue, "save_state", fail_proposal)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "proposal persistence unavailable" in outcome.reason
    reviewer = fake_cao._sessions[session_name_for(loop.run_id, "requirements-reviewer")]
    assert len(reviewer.submitted_messages) == 1
    assert not github.list_open_prs()


@pytest.mark.parametrize(
    "payload",
    [
        {"dispositions": []},
        {"dispositions": [{"finding_id": "other", "action": "rebut", "rationale": "evidence"}]},
        {"dispositions": [{"finding_id": "guard", "action": "rebut", "rationale": " "}]},
        {"dispositions": [{"finding_id": "guard", "action": "rebut", "rationale": 7}]},
        {
            "dispositions": [
                {
                    "finding_id": "guard",
                    "action": "rebut",
                    "rationale": "evidence",
                    "run_id": "forged",
                }
            ]
        },
        {"dispositions": [{"finding_id": "guard", "action": "rebut", "rationale": "evidence"}] * 2},
    ],
)
def test_incomplete_or_forged_coder_response_fails_closed(fake_cao, foreman_harness, payload):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id),
        [developer_output(), developer_output(dispositions=payload["dispositions"])],
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "Invalid disposition output" in outcome.reason
    assert queue.load_state("owner/repo#1").dispositions == []
    assert not github.list_open_prs()


@pytest.mark.parametrize("origin_available", [True, False])
def test_saved_proposal_survives_foreman_and_controller_loss(
    fake_cao, foreman_harness, monkeypatch, origin_available
):
    from ai_pr_orchestrator.v3.cao import CaoSessionController
    from ai_pr_orchestrator.v3.cao_lane import CaoLaneExecutor
    from ai_pr_orchestrator.v3.domain import GitHubIssueRef

    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    save = queue.save_state

    def crash_after_proposal(state, **kwargs):
        save(state, **kwargs)
        if state.dispositions:
            raise KeyboardInterrupt("process lost after durable proposal")

    monkeypatch.setattr(queue, "save_state", crash_after_proposal)
    with pytest.raises(KeyboardInterrupt):
        loop.run_pass()
    monkeypatch.setattr(queue, "save_state", save)
    state = WorkflowState.from_dict(queue.load_state("owner/repo#1").to_dict())
    if not origin_available:
        from ai_pr_orchestrator.v3.config import ReviewPolicyConfig

        loop._cfg = replace(
            loop._cfg,
            review_policy=ReviewPolicyConfig(
                reviewer_lanes=["breaker-reviewer", "architecture-reviewer"]
            ),
        )
        fake_cao.set_output(
            session_name_for(loop.run_id, "breaker-reviewer"),
            json.dumps(
                {
                    "findings": [],
                    "dispositions": [
                        {
                            "finding_id": "guard",
                            "action": "accept",
                            "rationale": "Independent fallback reproduced the evidence",
                            "response_to_round_id": "response-1",
                        }
                    ],
                }
            ),
        )
    # A changed configured coder cannot reuse the saved proposal turn under another actor.
    worker = loop._worker_lane()
    with monkeypatch.context() as changed_config:
        changed_config.setattr(
            loop, "_worker_lane", lambda: replace(worker, lane="other-developer")
        )
        with pytest.raises(_ForemanEscalation, match="provenance"):
            loop._pending_proposals(state)
    original = loop._executor
    with CaoSessionController(original._controller._config, loop._lanes) as controller:
        executor = CaoLaneExecutor(
            controller, loop._lanes, git=original._git, catalog=original._catalog
        )
        fresh = ForemanPolicyLoop(
            queue,
            loop._broker,
            loop._lanes,
            executor,
            loop._gate,
            loop._git,
            loop._cfg,
            run_id=loop.run_id,
            worktree_root="/wt",
            committer_name="test",
            committer_email="test@invalid",
        )
        outcome = fresh._run_loop(
            GitHubIssueRef("owner", "repo", 1),
            state,
            state.extras["worktree"],
            state.extras["branch"],
            now=None,
        )
    assert outcome.final_phase == "done", outcome.reason
    assert outcome.coder_invocations == 2  # Total durable usage; no additional coder call.
    restored = queue.load_state("owner/repo#1")
    assert [d.action for d in restored.dispositions] == ["rebut", "accept"]
    assert not restored.findings
    assert len(github.list_open_prs()) == 1
    assert len(fake_cao._sessions[developer_session_name(loop.run_id)].submitted_messages) == 2


def test_acceptance_save_failure_never_gates(fake_cao, foreman_harness, monkeypatch):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    save = queue.save_state

    def fail_acceptance(state, **kwargs):
        if any(d.action == "accept" for d in state.dispositions):
            raise RuntimeError("acceptance persistence unavailable")
        return save(state, **kwargs)

    monkeypatch.setattr(queue, "save_state", fail_acceptance)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    state = queue.load_state("owner/repo#1")
    assert [d.action for d in state.dispositions] == ["rebut"]
    assert state.findings[0].status == "open"
    assert not state.archived
    assert not github.list_open_prs()


def test_mixed_proposals_preserve_partial_adjudication(fake_cao, foreman_harness):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    findings = [
        {"id": fid, "body": body, "severity": "major"}
        for fid, body in [("guard", "missing guard"), ("race", "race condition")]
    ]
    proposals = [
        {"finding_id": fid, "action": action, "rationale": "specific regression evidence"}
        for fid, action in [("guard", "fix"), ("race", "rebut")]
    ]
    decisions = [
        {
            "finding_id": fid,
            "action": action,
            "rationale": "independent result",
            "response_to_round_id": "response-1",
        }
        for fid, action in [("guard", "accept"), ("race", "fix")]
    ]
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id),
        [developer_output(), developer_output(dispositions=proposals)],
    )
    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "requirements-reviewer"),
        [json.dumps(findings), json.dumps({"findings": [], "dispositions": decisions})],
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    state = queue.load_state("owner/repo#1")
    assert [(f.id, f.status) for f in state.findings] == [("race", "open")]
    assert [f.finding_id for f in state.archived] == ["guard"]
    assert [d.action for d in state.dispositions] == ["fix", "rebut", "accept", "fix"]
    assert not github.list_open_prs()


def test_rebuttal_at_round_cap_stays_open(fake_cao, foreman_harness):
    from ai_pr_orchestrator.v3.config import ReviewPolicyConfig

    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    loop._cfg = replace(loop._cfg, review_policy=ReviewPolicyConfig(max_review_rounds=1))
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "review-round cap" in outcome.reason
    state = queue.load_state("owner/repo#1")
    assert state.findings[0].status == "open"
    assert [d.action for d in state.dispositions] == ["rebut"]
    assert not github.list_open_prs()


def test_acceptance_cas_then_object_loss_replays_without_resurrection(
    fake_cao, foreman_harness, monkeypatch
):
    from ai_pr_orchestrator.v3.domain import GitHubIssueRef
    from ai_pr_orchestrator.v3.queue import GitHubIssueQueue

    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    save = queue.save_state

    def crash_after_acceptance(state, **kwargs):
        save(state, **kwargs)
        if any(d.action == "accept" for d in state.dispositions):
            raise KeyboardInterrupt("acceptance CAS succeeded; process lost")

    monkeypatch.setattr(queue, "save_state", crash_after_acceptance)
    with pytest.raises(KeyboardInterrupt):
        loop.run_pass()
    fresh_queue = GitHubIssueQueue(
        github, "owner", "repo", loop._cfg.github_queue, host_id="host-e2e"
    )
    fresh = ForemanPolicyLoop(
        fresh_queue,
        loop._broker,
        loop._lanes,
        loop._executor,
        loop._gate,
        loop._git,
        loop._cfg,
        run_id=loop.run_id,
        worktree_root="/wt",
        committer_name="test",
        committer_email="test@invalid",
    )
    state = fresh_queue.load_state("owner/repo#1")
    assert state is not None
    assert state.findings == []
    assert [d.action for d in state.dispositions] == ["rebut", "accept"]
    assert [f.finding_id for f in state.archived] == ["guard"]
    issue = GitHubIssueRef("owner", "repo", 1)
    registry = FindingRegistry(
        findings=list(state.findings),
        archived=list(state.archived),
        quarantine_unknown_head_sha=False,
    )
    fresh._persist_round(issue, state, "review-2", registry, [state.dispositions[-1]])
    replayed = fresh_queue.load_state(issue.slug())
    assert replayed is not None
    assert replayed.dispositions == state.dispositions
    assert replayed.archived == state.archived
    assert replayed.findings == []
    version = replayed.updated_at
    with pytest.raises(_ForemanEscalation, match="conflicting replay"):
        fresh._persist_round(
            issue,
            replayed,
            "review-2",
            registry,
            [replace(state.dispositions[-1], rationale="different decision content")],
        )
    unchanged = fresh_queue.load_state(issue.slug())
    assert unchanged is not None and unchanged.updated_at == version
    assert not github.list_open_prs()  # Interrupted before the gate.


def test_conflict_group_changes_do_not_mutate_review_cas_baseline(fake_cao, foreman_harness):
    """Acceptance of one conflict member must not discard a later legitimate rejection."""
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)

    def output(lane, responses):
        name = (
            developer_session_name(loop.run_id)
            if lane == "developer"
            else session_name_for(loop.run_id, lane)
        )
        payloads = (
            [
                developer_output(
                    summary=response if isinstance(response, str) else "addressed reviewer finding",
                    dispositions=[] if isinstance(response, str) else response["dispositions"],
                )
                for response in responses
            ]
            if lane == "developer"
            else [json.dumps(response) for response in responses]
        )
        fake_cao.set_output_sequence(name, payloads)

    def finding(fid, body):
        return {"id": fid, "body": body, "severity": "major", "path": "src/guard.py", "line": 10}

    def decide(fid, action, turn="response-1"):
        return {
            "finding_id": fid,
            "action": action,
            "rationale": "Independent reproduction",
            "response_to_round_id": turn,
        }

    output(
        "requirements-reviewer",
        [
            [finding("guard", "guard is missing")],
            {"findings": [], "dispositions": [decide("guard", "accept")]},
            [],
        ],
    )
    output(
        "breaker-reviewer",
        [
            [finding("race", "guard is harmful")],
            {"findings": [], "dispositions": [decide("race", "fix")]},
            {"findings": [], "dispositions": [decide("race", "fix", "response-2")]},
        ],
    )
    output(
        "developer",
        [
            "implemented",
            {
                "dispositions": [
                    {"finding_id": fid, "action": "rebut", "rationale": "Concrete evidence"}
                    for fid in ["guard", "race"]
                ]
            },
            {
                "dispositions": [
                    {"finding_id": "race", "action": "fix", "rationale": "New regression evidence"}
                ]
            },
        ],
    )
    outcome = loop.run_pass()[0]
    state = queue.load_state("owner/repo#1")
    assert outcome.reason == "unresolved findings after 3 review rounds"
    assert [(d.finding_id, d.action) for d in state.dispositions] == [
        ("guard", "rebut"),
        ("race", "rebut"),
        ("guard", "accept"),
        ("race", "fix"),
        ("race", "fix"),
        ("race", "fix"),
    ]
    assert [f.finding_id for f in state.archived] == ["guard"]
    assert [(f.id, f.conflict_group_id) for f in state.findings] == [("race", None)]
    assert not github.list_open_prs()


@pytest.mark.parametrize(
    "repeat_id,severity",
    [("guard", "major"), ("guard-repeat", "major"), ("guard-repeat", "blocker")],
)
def test_repeated_open_finding_keeps_identity_and_evidence(
    fake_cao, foreman_harness, repeat_id, severity
):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    first = [
        {"id": fid, "body": body, "severity": "major"}
        for fid, body in [("guard", "Missing guard"), ("race", "Race condition")]
    ]
    proposal = {
        "dispositions": [
            {"finding_id": fid, "action": "rebut", "rationale": "Concrete evidence"}
            for fid in ["guard", "race"]
        ]
    }
    repeated = {
        "id": repeat_id,
        "body": "Missing guard",
        "severity": severity,
        "reproduction_command": "pytest regression_guard.py",
    }
    decisions = [
        {
            "finding_id": fid,
            "action": action,
            "rationale": "Independent evidence",
            "response_to_round_id": "response-1",
        }
        for fid, action in [("guard", "fix"), ("race", "accept")]
    ]
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id),
        [developer_output(), developer_output(dispositions=proposal["dispositions"])],
    )
    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "requirements-reviewer"),
        [json.dumps(first), json.dumps({"findings": [repeated], "dispositions": decisions})],
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    state = queue.load_state("owner/repo#1")
    assert [(f.id, f.status) for f in state.findings] == [("guard", "open")]
    assert state.findings[0].severity == severity
    assert state.findings[0].reproduction_command == "pytest regression_guard.py"
    assert {(source.finding_id, source.round_id) for source in state.findings[0].sources} == {
        ("guard", "review-1"),
        (repeat_id, "review-2"),
    }
    assert [a.finding_id for a in state.archived] == ["race"]
    assert len(state.dispositions) == 4
    assert not github.list_open_prs()


def test_saved_rejection_does_not_reset_coder_budget(fake_cao, foreman_harness, monkeypatch):
    loop, queue, gh = foreman_harness()
    script_protocol(fake_cao, loop, decision="fix")
    loop._cfg = replace(
        loop._cfg, safety=replace(loop._cfg.safety, max_coder_invocations_per_run=2)
    )
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id),
        [
            developer_output(),
            developer_output(
                dispositions=[
                    {"finding_id": "guard", "action": "rebut", "rationale": "first evidence"}
                ]
            ),
            developer_output(
                dispositions=[
                    {"finding_id": "guard", "action": "fix", "rationale": "second evidence"}
                ]
            ),
        ],
    )

    def reply(action, turn):
        return json.dumps(
            {
                "findings": [],
                "dispositions": [
                    {
                        "finding_id": "guard",
                        "action": action,
                        "rationale": "independent result",
                        "response_to_round_id": turn,
                    }
                ],
            }
        )

    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "requirements-reviewer"),
        [
            '[{"id":"guard","body":"missing guard","severity":"major"}]',
            reply("fix", "response-1"),
            reply("accept", "response-2"),
        ],
    )
    save = queue.save_state

    def crash_after_rejection(state, **kwargs):
        save(state, **kwargs)
        if state.dispositions and state.dispositions[-1].response_to_round_id:
            raise KeyboardInterrupt("lost process after rejection CAS")

    monkeypatch.setattr(queue, "save_state", crash_after_rejection)
    with pytest.raises(KeyboardInterrupt):
        loop.run_pass()
    monkeypatch.setattr(queue, "save_state", save)
    state = queue.load_state("owner/repo#1")
    fresh = ForemanPolicyLoop(
        queue,
        loop._broker,
        loop._lanes,
        loop._executor,
        loop._gate,
        loop._git,
        loop._cfg,
        run_id=loop.run_id,
        worktree_root="/wt",
        committer_name="test",
        committer_email="test@invalid",
    )
    with pytest.raises(_ForemanEscalation, match="budget"):
        fresh._run_loop(
            GitHubIssueRef("owner", "repo", 1),
            state,
            state.extras["worktree"],
            state.extras["branch"],
            now=None,
        )
    prompts = fake_cao._sessions[developer_session_name(loop.run_id)].submitted_messages
    assert len(prompts) == 2, f"actual coder calls={len(prompts)}"
    assert not gh.list_open_prs()


@pytest.mark.parametrize("after_dispatch", [False, True])
def test_unknown_initial_coder_attempt_remains_charged(
    fake_cao, foreman_harness, monkeypatch, after_dispatch
):
    from ai_pr_orchestrator.v3.foreman import _ForemanEscalation

    loop, queue, gh = foreman_harness()
    for lane in loop._lanes:
        fake_cao.set_output(session_name_for(loop.run_id, lane.lane), "[]")
    execute = loop._executor.execute

    def crash(*args, **kwargs):
        if after_dispatch:
            execute(*args, **kwargs)
        raise KeyboardInterrupt("lost process at unknown initial invocation outcome")

    monkeypatch.setattr(loop._executor, "execute", crash)
    with pytest.raises(KeyboardInterrupt):
        loop.run_pass()
    state = queue.load_state("owner/repo#1")
    assert state.findings == []
    fresh = ForemanPolicyLoop(
        queue,
        loop._broker,
        loop._lanes,
        loop._executor,
        loop._gate,
        loop._git,
        loop._cfg,
        run_id=loop.run_id,
        worktree_root="/wt",
        committer_name="test",
        committer_email="test@invalid",
    )
    monkeypatch.setattr(
        loop._executor,
        "execute",
        lambda *a, **kw: pytest.fail("restarted initial attempt exceeded cap1"),
    )
    with pytest.raises(_ForemanEscalation, match="budget"):
        fresh._run_loop(
            GitHubIssueRef("owner", "repo", 1),
            state,
            state.extras["worktree"],
            state.extras["branch"],
            now=None,
        )
    assert not gh.list_open_prs()


@pytest.mark.parametrize(
    "repeat_id,body,decision",
    [
        ("guard", "A different claim", "fix"),
        ("guard", "Alleged missing guard", "accept"),
        ("new-id", "Alleged missing guard", "accept"),
    ],
)
def test_conflicting_id_or_acceptance_repetition_fails_closed(
    fake_cao, foreman_harness, repeat_id, body, decision
):
    loop, queue, github = foreman_harness()
    script_protocol(fake_cao, loop)
    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "requirements-reviewer"),
        [
            '[{"id":"guard","body":"Alleged missing guard","severity":"major"}]',
            json.dumps(
                {
                    "findings": [{"id": repeat_id, "body": body, "severity": "major"}],
                    "dispositions": [
                        {
                            "finding_id": "guard",
                            "action": decision,
                            "rationale": "independent evidence",
                            "response_to_round_id": "response-1",
                        }
                    ],
                }
            ),
        ],
    )
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    state = queue.load_state("owner/repo#1")
    assert [(f.id, f.status) for f in state.findings] == [("guard", "open")]
    assert not state.archived
    assert [d.action for d in state.dispositions] == ["rebut"]
    assert not github.list_open_prs()


def test_coder_charge_save_failure_prevents_dispatch(fake_cao, foreman_harness, monkeypatch):
    loop, queue, github = foreman_harness()
    save = queue.save_state

    def reject_charge(state, **kwargs):
        if "coder_usage" in state.extras:
            raise RuntimeError("cannot reserve coder budget")
        return save(state, **kwargs)

    monkeypatch.setattr(queue, "save_state", reject_charge)
    outcome = loop.run_pass()[0]
    assert outcome.final_phase == "escalated"
    assert "cannot reserve coder budget" in outcome.reason
    assert not fake_cao._sessions
    assert not github.list_open_prs()
