"""External Git boundary and structured CAO model-response fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_pr_orchestrator.v3.cao import session_name_for
from ai_pr_orchestrator.v3.config import SafetyPolicyConfig, V3Config
from ai_pr_orchestrator.v3.interfaces import GateDecision

NOW = datetime.now(UTC)


@dataclass
class StaticGate:
    """In-memory :class:`CIPRGate` that returns a pre-set decision.

    Use one per scenario, configured to either pass or fail on demand.
    """

    decision: GateDecision

    def evaluate(self, issue: Any, pr: Any) -> GateDecision:
        return self.decision


@dataclass
class FakeGitOperations:
    default: str = "main"
    branches: list[str] = field(default_factory=lambda: ["main"])
    worktrees: dict[str, str] = field(default_factory=dict)
    commits: list[tuple[str, str]] = field(default_factory=list)

    def default_branch(self) -> str:
        return self.default

    def create_branch(self, branch: str, from_ref: str) -> None:
        self.branches.append(branch)

    def create_worktree(self, path: str, branch: str, *, run_id: str, work_item_id: str) -> str:
        self.worktrees[path] = branch
        return path

    def create_review_checkout(
        self, path: str, source_workdir: str, head_sha: str, *, run_id: str, work_item_id: str
    ) -> str:
        self.worktrees[path] = f"detached:{head_sha}"
        return path

    def head_sha(self, workdir: str) -> str:
        return "sha"

    def verify_checkout(self, path: str, run_id: str, work_item_id: str) -> None:
        pass

    def current_branch(self, workdir: str) -> str:
        return self.worktrees.get(workdir, "aipro-issue-1")

    def repo_instructions(self, workdir: str) -> str:
        return ""

    def write_issue_description(self, workdir: str, description: str) -> tuple[str, str]:
        raise NotImplementedError("use real GitWorktreeOps for issue input delivery tests")

    def commit(self, workdir: str, message: str, *, name: str, email: str) -> str:
        self.commits.append((workdir, message))
        return "sha"

    def commit_count(self, workdir: str, base_ref: str) -> int:
        return sum(1 for w, _ in self.commits if w == workdir)

    def patch_digest(self, workdir: str, base_ref: str) -> str:
        return "|".join(sorted(self.changed_files(workdir, base_ref)))

    def push(self, branch: str) -> None:
        return None

    def push_sha(self, head_sha: str, branch: str) -> None:
        return None

    def cleanup_worktree(self, path: str) -> None:
        self.worktrees.pop(path, None)

    def cleanup_review_checkout(self, path: str, head_sha: str) -> None:
        self.worktrees.pop(path, None)

    def changed_files(self, workdir: str, base_ref: str | None = None) -> list[str]:
        return [f"src/{workdir.strip('/').replace('/', '_') or 'x'}.py"] if base_ref else []


__all__ = [
    "FakeGitOperations",
    "StaticGate",
    "developer_output",
    "developer_session_name",
]


def developer_session_name(run_id: str, issue_number: int = 1) -> str:
    return session_name_for(run_id, "developer", f"owner/repo#{issue_number}")


def developer_output(
    *,
    summary: str = "implemented requested change",
    dispositions: list[dict[str, str]] | None = None,
    no_changes: bool = False,
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "tests": [{"command": "pytest -q", "result": "passed", "notes": ""}],
            "concerns": [],
            "no_changes": no_changes,
            "dispositions": dispositions or [],
        }
    )


def script_protocol(
    fake_cao, loop, *, proposal="rebut", decision="accept", reply=None, issue_number: int = 1
):
    loop._cfg = V3Config(
        safety=SafetyPolicyConfig(
            max_coder_invocations_per_run=3,
            max_reviewer_triggers_per_run=9,
            max_commits_per_run=3,
        )
    )
    coder = developer_output(
        summary="addressed reviewer finding",
        dispositions=[
            {
                "finding_id": "guard",
                "action": proposal,
                "rationale": "test_guard proves the premise cannot occur",
            }
        ],
    )
    reviewer = {
        "findings": [],
        "dispositions": [
            {
                "finding_id": "guard",
                "action": decision,
                "rationale": "Independent reproduction confirms the coder evidence",
                "response_to_round_id": "response-1",
            }
        ],
    }
    fake_cao.set_output_sequence(
        developer_session_name(loop.run_id, issue_number),
        [developer_output(), coder],
    )
    fake_cao.set_output_sequence(
        session_name_for(loop.run_id, "requirements-reviewer"),
        [
            '[{"id":"guard","body":"Alleged missing guard","severity":"major"}]',
            json.dumps(reviewer) if reply is None else reply,
        ],
    )
    for lane in ("breaker-reviewer", "architecture-reviewer"):
        fake_cao.set_output(session_name_for(loop.run_id, lane), "[]")
