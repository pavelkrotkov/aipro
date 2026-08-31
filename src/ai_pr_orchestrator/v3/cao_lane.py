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

import time

from .cao import (
    CaoSessionController,
    CaoSessionNotFoundError,
    SessionBusyError,
    SessionIdentityUncertainError,
    session_name_for,
)
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
        max_poll_seconds: float = 600.0,
    ) -> None:
        self._controller = controller
        self._lanes = lane_registry
        self._poll_interval = poll_interval_seconds
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
        4. If launch is uncertain (CAO may have created the session
           before losing the response), reconcile by attempting
           ``adopt_session`` for the same deterministic name.
        5. Submit the prompt for this turn so an adopted session also
           receives work.
        6. Poll ``poll_session`` until the controller reports terminal
           state or the wall-clock budget is exhausted.
        7. On any exception, ``terminate_session`` (idempotent) and
           re-raise so the foreman can classify the failure. If the
           cleanup itself fails, preserve the original exception via
           ``__context__`` so the caller still sees the cause.
        """
        registered_lane = self._lanes.get(lane.lane)
        if registered_lane is None:
            # Without this guard, SessionSpec.__post_init__ would
            # TypeError when the LaneIdentity compares an unset field.
            raise LookupError(
                f"lane {lane.lane!r} is not registered in the LaneRegistry; "
                "the executor refuses to construct a SessionSpec for an "
                "unbound lane"
            )
        env: dict[str, str] = {}
        if lease is not None:
            # Forward the broker's model selection into the session
            # environment so the agent's startup can resolve the same
            # model the broker reserved. CAO forwards env_vars verbatim
            # to the agent (see cao.py start_session), so this is the
            # authoritative channel. We deliberately use an opaque
            # ``AIPRO_MODEL_REF`` key — never a vendor or model name —
            # and rely on the catalog to translate the ref to the
            # concrete provider at startup time.
            env["AIPRO_MODEL_REF"] = lease.assignment.model_ref
        spec = SessionSpec(
            lane=registered_lane,
            run_id=context.run_id,
            workdir=workdir,
            env=env,
            context=context,
            command=task_prompt,
            model_lease=lease,
        )

        deadline = time.monotonic() + self._max_poll
        handle = None
        # Round-2 finding, PR #71 #2: distinguish sessions this call
        # *created* from sessions it *adopted*. Adopted sessions are
        # owned by some other caller (or a previous foreman restart);
        # terminating them on the executor's cleanup path would kill
        # valid in-flight work. The flag is set True only after we know
        # we POSTed a brand-new session to CAO (i.e. start_session took
        # the new-session branch), and stays False for the adopted
        # branch and for adopt_session reconciliation.
        own_session = False
        try:
            try:
                # Probe whether an existing session is alive BEFORE
                # calling start_session, so we can mark ``own_session``
                # correctly. The controller's own adopt path is
                # implicit: if start_session returns a handle without a
                # POST, we did not create the session.
                name = session_name_for(spec.run_id, registered_lane.lane)
                preexisting = self._controller._lookup_session(name) is not None
                handle = self._controller.start_session(spec)
                own_session = not preexisting
            except SessionIdentityUncertainError:
                # Launch completion is uncertain: CAO may have created
                # the session before the response was lost. Reconcile
                # by adopting the same deterministic name and
                # constructing a handle from the recovered metadata.
                # If adoption also fails, the original uncertain
                # error propagates so the foreman can treat the work
                # item as un-attributable. Note: ``own_session`` stays
                # False; CAO MAY have created a session for the
                # uncertain launch, but we cannot claim ownership of
                # it and cleanup must not terminate it.
                from .interfaces import SessionHandle

                name = session_name_for(spec.run_id, registered_lane.lane)
                try:
                    self._controller.adopt_session(name, spec)
                except CaoSessionNotFoundError:
                    raise
                handle = SessionHandle(session_id=name, lane=registered_lane.lane)
                own_session = False

            # Refresh the per-turn attribution on every entry, whether
            # the session was created or adopted. Round-2 finding,
            # PR #71 #3: round_id / work_item_id are intentionally NOT
            # part of session-level adoption identity, but the
            # controller must still observe the latest turn so
            # downstream attribution (findings, dispositions) reports
            # against the current round. Safe to call on both
            # own-session and adopted-session paths.
            self._controller.update_turn_context(handle, context)

            # Submit the prompt for this turn so the session actually
            # receives the executor's task, including adopted sessions
            # (the controller's start_session path short-circuits before
            # delivery when adopting). A submission that races a
            # session deletion falls through to the poll loop, whose
            # first observe() will raise a typed error.
            self._controller.submit_work(handle, task_prompt)

            while True:
                result = self._controller.poll_session(handle)
                if result is not None:
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
                    raise TimeoutError(
                        f"CaoLaneExecutor exceeded its {self._max_poll:.0f}s poll "
                        f"budget waiting for session {handle.session_id!r} on "
                        f"lane {lane.lane!r}; the controller's "
                        "session_timeout_seconds is the authoritative cap on "
                        "the session's life and should be configured below "
                        "this executor's max_poll_seconds"
                    )
                time.sleep(self._poll_interval)
        except BaseException as primary_exc:
            # Best-effort teardown, but ONLY for sessions we created
            # in this call. An adopted session belongs to some other
            # caller; terminating it on the executor's cleanup path
            # would kill valid in-flight work (round-2 finding, PR
            # #71 #2). SessionBusyError in particular is the signal
            # that the adopted session is mid-work for someone else,
            # so even for sessions WE created, do not terminate a
            # session CAO is currently processing — re-raise so the
            # foreman can decide whether to retry/poll later.
            should_terminate = (
                handle is not None and own_session and not isinstance(primary_exc, SessionBusyError)
            )
            if should_terminate:
                assert handle is not None  # narrowing for pyright; should_terminate implies handle
                try:
                    self._controller.terminate_session(handle)
                except BaseException as cleanup_exc:
                    # Catch every cleanup error here so the primary
                    # exception stays visible to the caller. The type
                    # is intentionally broad: a failed teardown must
                    # never displace the typed lane error.
                    raise primary_exc from cleanup_exc
            raise primary_exc


__all__ = ["CaoLaneExecutor"]
