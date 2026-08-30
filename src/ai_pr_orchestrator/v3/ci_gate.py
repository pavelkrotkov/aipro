"""V3 CI/PR gate implementation (issue #55).

Production :class:`~ai_pr_orchestrator.v3.interfaces.CIPRGate`: evaluates one
pull request's commit statuses and check-runs (via the GitHub client
protocol) against :class:`~ai_pr_orchestrator.v3.config.CIPolicyConfig`.

Policy, stated once:

- A check that has not completed, or a commit status that is still
  ``pending``, is *pending* — and a gate cannot pass with outstanding work
  (:class:`~ai_pr_orchestrator.v3.interfaces.GateDecision` enforces the
  inverse, so a bug here fails loudly rather than passing silently).
- Anything that concluded ``failure``/``timed_out``/``cancelled``/
  ``action_required`` (or a commit status in state ``error``/``failure``) is
  a failed check.
- Every name in ``CIPolicyConfig.required_checks`` must be present; a
  required check that never reported is recorded as failed by name, because
  "did not run" must be visible, not silently equivalent to green.
- With ``require_green_ci_before_merge`` and *no* checks at all, the gate
  fails with an explicit detail: absence of evidence is not green.

No vendor, model, or provider name appears in this module.
"""

from __future__ import annotations

from ..github.protocol import GitHubClient as GitHubClientProtocol
from .config import CIPolicyConfig
from .domain import GitHubIssueRef, GitHubPullRequestRef
from .interfaces import GateDecision

#: Check-run conclusions that count as failed.
_FAILED_CONCLUSIONS = frozenset(("failure", "timed_out", "cancelled", "action_required", "stale"))
#: Commit-status states that count as failed.
_FAILED_STATES = frozenset(("error", "failure"))
#: Check-run conclusion that marks completion.
_COMPLETED = "completed"


class CIPRGateImpl:
    """GitHub-backed CI gate over check-runs and commit statuses."""

    def __init__(
        self,
        client: GitHubClientProtocol,
        config: CIPolicyConfig | None = None,
    ) -> None:
        self._client = client
        self._cfg = config or CIPolicyConfig()

    def evaluate(self, issue: GitHubIssueRef, pr: GitHubPullRequestRef) -> GateDecision:
        del issue  # the gate reads the PR's head ref; the issue names the audit log
        ref = pr.head_sha
        runs = self._client.get_check_runs(ref)
        statuses = self._client.get_commit_statuses(ref)

        by_name: dict[str, str] = {}
        failed: list[str] = []
        pending: list[str] = []
        for run in runs:
            if run.status != _COMPLETED:
                pending.append(run.name)
            elif run.conclusion in _FAILED_CONCLUSIONS or run.conclusion is None:
                failed.append(run.name)
            else:
                by_name[run.name] = run.conclusion
        for status in statuses:
            if status.status != _COMPLETED:
                pending.append(status.name)
            elif status.conclusion in _FAILED_STATES:
                failed.append(status.name)
            else:
                by_name[status.name] = status.conclusion or "success"

        missing_required = [name for name in self._cfg.required_checks if name not in by_name]
        if self._cfg.required_checks:
            # A required check that has not completed is pending, not failed;
            # one that never reported at all is failed by name.
            failed.extend(
                name for name in missing_required if name not in pending and name not in failed
            )

        if not runs and not statuses:
            if missing_required:
                # Immediately after a push, required checks legitimately have not
                # reported yet. Their initial absence is *pending* (in-flight),
                # not a definitive failure — so the caller can re-poll rather than
                # conclude the run failed. Returning empty pending/failed here
                # would both lose the missing-required signal and look like a
                # green/no-op gate.
                return GateDecision(
                    passed=False,
                    pending_checks=tuple(dict.fromkeys(missing_required)),
                    failed_checks=(),
                    detail="required checks not yet reported for head " + ref,
                )
            if self._cfg.require_green_ci_before_merge:
                return GateDecision(
                    passed=False,
                    pending_checks=(),
                    failed_checks=(),
                    detail="no checks reported for head " + ref,
                )
            return GateDecision(
                passed=True,
                pending_checks=(),
                failed_checks=(),
                detail="no checks reported and policy does not require green CI",
            )

        failed = sorted(set(failed))
        pending_t = tuple(sorted(set(pending)))
        passed = not failed and not pending_t
        detail = f"required={sorted(self._cfg.required_checks)} observed={sorted(by_name)}"
        return GateDecision(
            passed=passed,
            pending_checks=pending_t,
            failed_checks=tuple(failed),
            detail=detail,
        )
