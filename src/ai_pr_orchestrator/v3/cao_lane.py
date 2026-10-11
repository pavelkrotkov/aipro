"""CAO lane execution with durable turn identity and bounded result collection."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from .cao import (
    CaoAdoptionMismatchError,
    CaoSessionController,
    CaoTransportError,
    SessionBusyError,
    SessionIdentityUncertainError,
    message_digest,
)
from .catalog import ModelCatalog
from .domain import LaneIdentity
from .interfaces import (
    GitOperations,
    LaneExecutionContext,
    LaneResult,
    ModelLease,
    SessionHandle,
    SessionSpec,
)
from .lanes import LaneRegistry


class CaoLaneExecutor:
    """A :class:`LaneExecutor` that runs every unit of work on CAO.

    Parameters
    ----------
    controller:
        A :class:`CaoSessionController`, including its work-submission and
        turn-context operations. The executor never owns its lifecycle: the
        caller is responsible for opening and closing it.
    lane_registry:
        The :class:`LaneRegistry` that owns the ``lane -> profile``
        binding. The executor looks each lane up on every call so a
        reconfigured registry is picked up without re-instantiation.
    git:
        The existing repository Git operations; completed results are populated
        from actual worktree and issue-branch changes, never agent summaries.
    catalog:
        The same resolved catalog used by the broker. Leased model refs must
        resolve to a model descriptor and an explicit Hermes provider.
    env:
        Optional session defaults, copied before model overrides are applied.
    poll_interval_seconds:
        Wall-clock sleep between ``poll_session`` calls. Defaults to a
        small value suitable for the E2E harness against an in-process
        CAO fake; a real ``cao-server`` typically uses a few seconds.
    max_poll_seconds:
        Wall-clock budget for one ``execute`` call. The CAO controller's
        own ``session_timeout_seconds`` is the authoritative cap on the
        session's life; this budget is the executor's own safety net to
        avoid an infinite poll loop if the controller's idle-settle
        never trips. Defaults to the controller timeout plus 30 seconds;
        explicit overrides must exceed the controller timeout. This cannot
        interrupt a blocked HTTP call, which has its own request timeout.
    """

    def __init__(
        self,
        controller: CaoSessionController,
        lane_registry: LaneRegistry,
        *,
        git: GitOperations,
        catalog: ModelCatalog | None = None,
        env: Mapping[str, str] | None = None,
        poll_interval_seconds: float = 0.05,
        max_poll_seconds: float | None = None,
    ) -> None:
        self._controller = controller
        self._lanes = lane_registry
        self._git = git
        self._catalog = catalog or ModelCatalog()
        self._env = dict(env or {})
        self._poll_interval = poll_interval_seconds
        timeout = controller.session_timeout_seconds
        self._max_poll = timeout + 30.0 if max_poll_seconds is None else max_poll_seconds
        if not isfinite(self._max_poll) or self._max_poll <= timeout:
            raise ValueError("max_poll_seconds must be finite and exceed session_timeout_seconds")

    def _session_env(self, lease: ModelLease | None) -> dict[str, str]:
        env = dict(self._env)
        env.update(
            GIT_CONFIG_COUNT="6",
            GIT_CONFIG_KEY_0="remote.origin.pushurl",
            GIT_CONFIG_VALUE_0="aipro-no-push://authoritative-branch",
            GIT_CONFIG_KEY_1="url.aipro-no-push://authoritative-branch.pushInsteadOf",
            GIT_CONFIG_VALUE_1="https://github.com/",
            GIT_CONFIG_KEY_2="url.aipro-no-push://authoritative-branch.pushInsteadOf",
            GIT_CONFIG_VALUE_2="git@github.com:",
            GIT_CONFIG_KEY_3="url.aipro-no-push://authoritative-branch.pushInsteadOf",
            GIT_CONFIG_VALUE_3="ssh://git@github.com/",
            GIT_CONFIG_KEY_4="credential.helper",
            GIT_CONFIG_VALUE_4="",
            GIT_CONFIG_KEY_5="credential.interactive",
            GIT_CONFIG_VALUE_5="false",
            GIT_TERMINAL_PROMPT="0",
            GIT_ASKPASS="/bin/false",
            SSH_ASKPASS="/bin/false",
            SSH_AUTH_SOCK="",
            GIT_SSH_COMMAND="ssh -oBatchMode=yes -oIdentitiesOnly=yes -oIdentityFile=/dev/null",
            GH_TOKEN="",
            GITHUB_TOKEN="",
            GH_CONFIG_DIR="/nonexistent/aipro-no-github-auth",
        )
        if lease is None:
            return env
        entry = self._catalog.get(lease.assignment.model_ref)
        if entry is None:
            raise ValueError(f"Leased model {lease.assignment.model_ref!r} is absent from catalog")
        if not entry.provider or entry.endpoint is not None:
            raise ValueError(
                f"Leased model {entry.ref!r} requires an explicit Hermes provider "
                "and no custom endpoint; this launcher cannot bind a custom endpoint"
            )
        env.update(AIPRO_MODEL=entry.descriptor, AIPRO_PROVIDER=entry.provider)
        return env

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease: ModelLease | None = None,
    ) -> LaneResult:
        """Submit a new turn once, or collect its matching durable result."""
        registered_lane = self._lanes.get(lane.lane)
        spec = SessionSpec(
            lane=registered_lane,
            run_id=context.run_id,
            workdir=workdir,
            env=self._session_env(lease),
            context=context,
            command=task_prompt,
            model_lease=lease,
        )

        handle = self._controller.start_session(spec)
        try:
            metadata = self._controller._metadata_for(handle)
            digest = message_digest(task_prompt, context)
            if metadata.input_digest == digest:
                if metadata.context != context:
                    raise SessionIdentityUncertainError(
                        handle.session_id, "matching input has different turn attribution"
                    )
            else:
                self._controller.submit_work(handle, task_prompt, context=context)
            return self.collect(handle, registered_lane, workdir, context)
        except (CaoTransportError, SessionBusyError, SessionIdentityUncertainError):
            raise
        except BaseException:
            self._controller.terminate_session(handle)
            raise

    def collect(
        self, handle: SessionHandle, lane: LaneIdentity, workdir: str, context: LaneExecutionContext
    ) -> LaneResult:
        import time

        metadata = self._controller._metadata_for(handle)
        if (
            metadata.context,
            metadata.workdir,
            metadata.lane,
            metadata.input_digest is not None,
        ) != (context, workdir, lane, True):
            raise CaoAdoptionMismatchError("Cannot collect a different attributed turn")
        deadline = time.monotonic() + self._max_poll
        while True:
            result = self._controller.poll_session(handle)
            if result is not None:
                base_ref = (
                    context.head_sha
                    if lane.role == "reviewer" and context.head_sha
                    else self._git.default_branch()
                )
                return LaneResult(
                    session=handle,
                    exit_code=result.exit_code,
                    output_summary=result.output_summary,
                    changed_files=self._git.changed_files(workdir, base_ref),
                    findings=list(result.findings),
                    dispositions=list(result.dispositions),
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"CaoLaneExecutor exceeded its {self._max_poll:.0f}s poll budget for {handle.session_id!r}"
                )
            time.sleep(self._poll_interval)

    def terminate_session(self, handle: SessionHandle) -> None:
        """Confirm remote teardown before a disposable checkout is removed."""
        self._controller.terminate_session(handle)


__all__ = ["CaoLaneExecutor"]
