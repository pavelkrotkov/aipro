"""Production :class:`LaneExecutor` bridge over :class:`CaoSessionController`.

The foreman policy loop and the lane-execution contract are defined in
:mod:`ai_pr_orchestrator.v3.interfaces`; this module is the production
implementation that satisfies the :class:`LaneExecutor` Protocol by
delegating every unit of work to a CAO session. The unit tests in
``test_v3_foreman.py`` use ``ScriptedExecutor`` so the foreman's branch
logic can be exercised without spinning up a CAO; this module is the
real-execution counterpart, used by the E2E harness and the future
production path.

Three properties are load-bearing:

- **The session's name is a pure function of (run, lane)**, so a crashed
  foreman adopting a previous process's session by name rebuilds the
  full identity from CAO's metadata, not from this module's local state.
- **Termination is idempotent and best-effort**: an exception mid-poll
  triggers ``terminate_session`` so the CAO session is not orphaned. A
  re-raise surfaces the typed lane error to the caller so the foreman
  can classify it (transient vs permanent).
- **The poll loop honors the controller's own idle-settle and timeout
  rules**: this module never invents its own timeout; it just calls
  ``poll_session`` in a loop with a small wall-clock budget and reports
  the controller's normalized lifecycle. The controller already maps
  ``observed terminal-status`` onto ``LaneResult.exit_code`` (0 on
  completed, 1 otherwise).
"""

from __future__ import annotations

from .cao import CaoSessionController
from .domain import LaneIdentity
from .interfaces import (
    LaneExecutionContext,
    LaneResult,
    SessionSpec,
)
from .lanes import LaneRegistry


class CaoLaneExecutor:
    """A :class:`LaneExecutor` that runs every unit of work on CAO.

    Parameters
    ----------
    controller:
        A :class:`CaoSessionController` (or anything that satisfies the
        :class:`~ai_pr_orchestrator.v3.interfaces.CAOSessionController`
        Protocol). The executor never owns the controller's lifecycle: the
        caller is responsible for opening and closing it.
    lane_registry:
        The :class:`LaneRegistry` that owns the ``lane -> profile``
        binding. The executor looks each lane up on every call so a
        reconfigured registry is picked up without re-instantiation.
    poll_interval_seconds:
        Wall-clock sleep between ``poll_session`` calls. Defaults to a
        small value suitable for the E2E harness against an in-process
        CAO fake; a real ``cao-server`` typically uses a few seconds.
    max_poll_seconds:
        Wall-clock budget for one ``execute`` call. The CAO controller's
        own ``session_timeout_seconds`` is the authoritative cap on the
        session's life; this budget is the executor's own safety net to
        avoid an infinite poll loop if the controller's idle-settle
        never trips. The controller's timeout always fires first.
    """

    def __init__(
        self,
        controller: CaoSessionController,
        lane_registry: LaneRegistry,
        *,
        poll_interval_seconds: float = 0.05,
        max_poll_seconds: float | None = None,
    ) -> None:
        self._controller = controller
        self._lanes = lane_registry
        self._poll_interval = poll_interval_seconds
        # The executor's poll budget is a safety net for ``poll_session``
        # returning ``None`` past the controller's own idle-settle or
        # ``session_timeout_seconds`` rules. By default it exceeds the
        # controller's session timeout by 60s of grace so a session that
        # legitimately runs the full budget is reported by the controller,
        # not terminated by the executor (PR #73 review thread 17).
        # Callers may still set a tighter budget explicitly to test the
        # executor's own exhaustion path.
        if max_poll_seconds is None:
            controller_cfg = getattr(controller, "_config", None)
            controller_session_timeout = getattr(controller_cfg, "session_timeout_seconds", None)
            if controller_session_timeout is None:
                max_poll_seconds = 600.0
            else:
                max_poll_seconds = float(controller_session_timeout) + 60.0
        self._max_poll = max_poll_seconds

    def execute(
        self,
        lane: LaneIdentity,
        task_prompt: str,
        workdir: str,
        context: LaneExecutionContext,
        lease=None,
    ) -> LaneResult:
        """Run one unit of work on a CAO session and return its result.

        The lifecycle:

        1. Resolve the lane via the registry (a reconfigured registry
           wins here; this is the only reason a registry is needed).
        2. Build a :class:`SessionSpec` carrying the typed run/round
           context and the optional :class:`ModelLease`.
        3. ``start_session`` adopts an existing session if one is alive
           under the same deterministic name; otherwise it creates one.
        4. Poll ``poll_session`` until the controller reports terminal
           state or the wall-clock budget is exhausted.
        5. On any exception, ``terminate_session`` (idempotent) and
           re-raise so the foreman can classify the failure.
        """
        import time

        registered_lane = self._lanes.get(lane.lane)
        # PR #73 review thread 7: forward the broker's resolved model into the
        # CAO session environment so the agent process actually consumes the
        # leased model rather than running against the profile default. Layer
        # on top of any caller-supplied env so profile-required overrides are
        # preserved; the model key is the documented override name.
        lane_env: dict[str, str] = {"AIPRO_MODEL": lease.assignment.model_ref} if lease else {}
        spec = SessionSpec(
            lane=registered_lane,
            run_id=context.run_id,
            workdir=workdir,
            env=lane_env,
            context=context,
            command=task_prompt,
            model_lease=lease,
        )

        handle = self._controller.start_session(spec)
        # PR #73 review thread 1: on every invocation — including adoption —
        # submit the prompt as work. ``CaoSessionController.start_session``
        # deliberately does NOT send ``initial_message``, so an adopted
        # session would otherwise be polled immediately and return its
        # previous output. The executor must explicitly submit follow-up
        # work; otherwise a fix-round invocation never reaches the agent and
        # the foreman reports success without the requested fix.
        self._controller.submit_work(handle, task_prompt)
        # PR #73 review thread 2: refresh the per-turn context after adoption
        # so the durable session metadata carries the latest ``round_id``,
        # matching the worker-lane semantics. Without this, repeated reviewer
        # invocations on the same deterministic session would carry stale
        # round attribution.
        self._controller.update_turn_context(handle, context)
        deadline = time.monotonic() + self._max_poll
        try:
            while True:
                result = self._controller.poll_session(handle)
                if result is not None:
                    # PR #73 review thread 6: a reviewer lane whose output
                    # could not be parsed into structured findings must not
                    # be reported as "no findings" — that lets the foreman
                    # open a PR on unreviewed changes. If the controller
                    # returned a non-empty ``output_summary`` for a reviewer
                    # lane but no findings, treat that as a structured-output
                    # failure and escalate (the foreman re-raises it).
                    if (
                        registered_lane.role == "reviewer"
                        and not result.findings
                        and (result.output_summary or "").strip()
                    ):
                        raise ReviewerOutputUnparsedError(
                            f"reviewer lane {lane.lane!r} returned text that "
                            "could not be parsed into structured findings; "
                            "the foreman must escalate rather than treat "
                            "this as a clean review"
                        )
                    # ``poll_session`` already builds a LaneResult; rewrite
                    # the ``session`` field with the live handle so the
                    # foreman can address the session that produced it.
                    return LaneResult(
                        session=handle,
                        exit_code=result.exit_code,
                        output_summary=result.output_summary,
                        changed_files=list(result.changed_files),
                        findings=list(result.findings),
                        dispositions=list(result.dispositions),
                    )
                if time.monotonic() >= deadline:
                    self._controller.terminate_session(handle)
                    raise TimeoutError(
                        f"CaoLaneExecutor exceeded its {self._max_poll:.0f}s poll "
                        f"budget waiting for session {handle.session_id!r} on "
                        f"lane {lane.lane!r}; the controller's "
                        "session_timeout_seconds is the authoritative cap on "
                        "the session's life and should be configured below "
                        "this executor's max_poll_seconds"
                    )
                time.sleep(self._poll_interval)
        except BaseException:
            # Best-effort teardown so the CAO session is not orphaned; a
            # session CAO has already forgotten is a no-op here.
            self._controller.terminate_session(handle)
            raise


class ReviewerOutputUnparsedError(RuntimeError):
    """Raised when a reviewer lane returns text but no structured findings.

    The production CAO controller parses agent output as ``ReviewerFinding``
    objects; a reviewer that returns prose describing blockers must not
    masquerade as a clean review with zero findings (PR #73 review thread 6).
    """


__all__ = ["CaoLaneExecutor", "ReviewerOutputUnparsedError"]
