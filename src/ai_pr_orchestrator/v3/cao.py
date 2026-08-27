"""CAO control-plane adapter.

aipro V3 does not spawn agent processes. It asks CAO — over CAO's documented
HTTP control plane — to create, observe, feed, and tear down named agent
sessions in a working directory. This module is the only place in V3 that
speaks to CAO.

Three properties are load-bearing:

- **No terminal parsing.** Lifecycle comes from CAO's typed terminal ``status``
  field; final output comes from CAO's own provider-extracted "last response"
  view. aipro never reads a raw scrollback, prompt glyph, or tmux pane.
- **Model choice is recorded, never made.** The broker's
  :class:`~ai_pr_orchestrator.v3.domain.ModelAssignment` is stored on the
  session record and its resolved form travels in ``SessionSpec.env``, which
  is forwarded verbatim to the session. This module never selects, maps, or
  names a model, and no vendor name appears in it.
- **Session identity is durable and deterministic.** A session's name is a
  pure function of its run and lane, so a restarted aipro re-attaches to the
  session it already launched (:meth:`CaoSessionController.adopt_session`)
  instead of creating a second one.

See ``docs/V3_CAO.md`` for the minimum CAO version and profile provisioning.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from .config import CAOControlPlaneConfig
from .domain import LaneIdentity, LaneName, ModelAssignment, RunId
from .interfaces import LaneExecutionContext, LaneResult, SessionHandle, SessionSpec
from .lanes import LaneRegistry

#: Minimum CAO release this adapter is written against. 2.4.x is the first
#: line whose control plane exposes everything used here.
MINIMUM_CAO_VERSION = "2.4"

#: Normalized session lifecycle. ``started`` means the session exists but its
#: agent is not yet observable; ``disappeared`` means CAO no longer knows the
#: session (killed out of band, or the server lost its state).
SessionLifecycle = Literal[
    "started",
    "running",
    "completed",
    "blocked",
    "failed",
    "disappeared",
    "timed_out",
]

#: Lifecycle states from which a session will not make further progress.
TERMINAL_LIFECYCLE_STATES: frozenset[str] = frozenset(
    ("completed", "failed", "disappeared", "timed_out")
)

# CAO's own terminal status vocabulary, mapped to the normalized lifecycle.
# ``idle`` is deliberately absent: it is ambiguous (a session is also idle
# before it picks work up) and is resolved by the idle-streak rule in
# CaoSessionController._lifecycle.
_STATUS_LIFECYCLE: dict[str, SessionLifecycle] = {
    "processing": "running",
    "completed": "completed",
    "waiting_user_answer": "blocked",
    "error": "failed",
}

# CAO prepends this to any session name that lacks it. We supply it ourselves
# so the name we launch under is byte-identical to the name CAO stores — which
# is what lets a later lookup find the session rather than launch a second one.
_CAO_SESSION_PREFIX = "cao-"
# tmux caps session names at 64 characters.
_MAX_SESSION_NAME_LEN = 64
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


class CaoControlPlaneError(RuntimeError):
    """Base class for CAO control-plane failures."""


class CaoUnavailableError(CaoControlPlaneError):
    """The request provably never reached CAO, so nothing happened.

    Safe to retry: the connection was refused or the host was unreachable
    before any bytes were delivered.
    """


class CaoTransportError(CaoControlPlaneError):
    """The request reached the wire but its outcome is unknown.

    A read timeout or a dropped connection mid-flight cannot distinguish "CAO
    never ran it" from "CAO ran it and we lost the answer". Callers must
    reconcile before acting again, never blind-retry.
    """


class CaoSessionNotFoundError(CaoControlPlaneError):
    """CAO does not know the named session or terminal."""


class SessionBusyError(CaoControlPlaneError):
    """The session is not accepting input right now (CAO answered 409)."""


class SessionIdentityUncertainError(CaoControlPlaneError):
    """A launch failed in a way that leaves session identity undetermined.

    The POST may have created a session before the answer was lost. Retrying
    it would risk a second live agent on the same lane and worktree, so the
    caller must instead reconcile ``session_name`` with
    :meth:`CaoSessionController.adopt_session`.
    """

    def __init__(self, session_name: str, detail: str) -> None:
        super().__init__(
            f"Launch of CAO session {session_name!r} did not confirm: {detail}. "
            "The session may or may not exist; reconcile by name instead of retrying."
        )
        self.session_name = session_name


class SessionNotRegisteredError(CaoControlPlaneError):
    """The handle names a session this controller has not launched or adopted."""


@dataclass(frozen=True)
class CaoSessionMetadata:
    """Everything needed to find, attribute, and audit one CAO session.

    This record is both aipro's durable session state and the payload pushed
    into CAO's own per-terminal metadata, which is what makes
    :meth:`CaoSessionController.adopt_session` able to rebuild it after a
    restart from nothing but the session name.
    """

    session_name: str
    terminal_id: str
    lane: LaneIdentity
    workdir: str
    context: LaneExecutionContext
    model_assignment: ModelAssignment | None = None
    launched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def parent_run_id(self) -> RunId:
        return self.context.run_id

    @property
    def profile_template(self) -> str:
        """The lane's profile — the isolated agent home this session runs in."""
        return self.lane.profile_template

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_name": self.session_name,
            "terminal_id": self.terminal_id,
            "lane": self.lane.to_dict(),
            "workdir": self.workdir,
            "run_id": self.context.run_id,
            "round_id": self.context.round_id,
            "work_item_id": self.context.work_item_id,
            "model_assignment": (
                self.model_assignment.to_dict() if self.model_assignment is not None else None
            ),
            "launched_at": self.launched_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaoSessionMetadata:
        assignment = data.get("model_assignment")
        launched_at = data.get("launched_at")
        return cls(
            session_name=data.get("session_name", ""),
            terminal_id=data.get("terminal_id", ""),
            lane=LaneIdentity.from_dict(data["lane"]),
            workdir=data["workdir"],
            context=LaneExecutionContext(
                run_id=data["run_id"],
                round_id=data.get("round_id"),
                work_item_id=data.get("work_item_id"),
            ),
            model_assignment=(
                ModelAssignment.from_dict(assignment) if assignment is not None else None
            ),
            launched_at=(
                datetime.fromisoformat(launched_at)
                if isinstance(launched_at, str)
                else datetime.now(UTC)
            ),
        )


@dataclass(frozen=True)
class SessionObservation:
    """One normalized reading of a session's lifecycle.

    ``cao_status`` is CAO's own status string, carried verbatim for audit; the
    policy engine branches on ``state``, never on ``cao_status``.
    """

    metadata: CaoSessionMetadata
    state: SessionLifecycle
    cao_status: str | None = None
    detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_LIFECYCLE_STATES


def session_name_for(run_id: RunId, lane: LaneName) -> str:
    """Return the deterministic CAO session name for a run's lane.

    Determinism is the whole reconcile story: the same run and lane always
    resolve to the same session, so a restarted process can look the session
    up instead of launching a second one.
    """
    raw = f"{_CAO_SESSION_PREFIX}aipro-{run_id}-{lane}"
    sanitized = _UNSAFE_NAME_CHARS.sub("-", raw)
    if len(sanitized) <= _MAX_SESSION_NAME_LEN:
        return sanitized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[: _MAX_SESSION_NAME_LEN - 9]}-{digest}"


class CaoSessionController:
    """Drives CAO sessions for V3 lanes over CAO's HTTP control plane.

    Implements :class:`~ai_pr_orchestrator.v3.interfaces.CAOSessionController`
    and adds the async-submission, follow-up, output, and reconcile operations
    the policy engine needs beyond the minimal protocol.
    """

    def __init__(
        self,
        config: CAOControlPlaneConfig,
        lanes: LaneRegistry | None = None,
        *,
        client: httpx.Client | None = None,
        idle_settle_polls: int = 3,
    ) -> None:
        """``idle_settle_polls`` is how many consecutive idle readings mark a
        session done. A session is idle both before it picks work up and after
        it finishes, and providers do not always emit an explicit completion
        marker; requiring the reading to persist separates "hasn't started" from
        "has finished" without inspecting any terminal content."""
        if idle_settle_polls < 1:
            raise ValueError(f"idle_settle_polls must be >= 1, got {idle_settle_polls}")
        self._config = config
        self._lanes = lanes if lanes is not None else LaneRegistry.default()
        self._idle_settle_polls = idle_settle_polls
        self._client = client if client is not None else self._build_client(config)
        self._sessions: dict[str, CaoSessionMetadata] = {}
        self._idle_readings: dict[str, int] = {}

    @staticmethod
    def _build_client(config: CAOControlPlaneConfig) -> httpx.Client:
        return httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CaoSessionController:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- Launch ------------------------------------------------------------

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        """Launch (or re-attach to) the session for ``spec``'s run and lane.

        ``spec.command`` is the opaque initial work message; when it is empty
        the session is created idle and work is submitted later with
        :meth:`submit_work`. ``spec.env`` is forwarded to the session verbatim
        — it is how the broker's resolved model reaches the agent's startup
        without this module interpreting it.
        """
        lane = self._registered_lane(spec.lane)
        name = session_name_for(spec.run_id, lane.lane)

        adopted = self._lookup_session(name)
        if adopted is not None:
            self._register(adopted)
            return SessionHandle(session_id=adopted.session_name, lane=lane.lane)

        metadata = CaoSessionMetadata(
            session_name=name,
            terminal_id="",
            lane=lane,
            workdir=spec.workdir,
            context=spec.context,
            model_assignment=(
                spec.model_lease.assignment if spec.model_lease is not None else None
            ),
        )
        body: dict[str, Any] = {
            "env_vars": spec.env or None,
            "group": [spec.run_id, lane.lane],
            "metadata": metadata.to_dict(),
        }
        if spec.command:
            body["initial_message"] = spec.command

        try:
            response = self._request(
                "POST",
                "/sessions",
                params={
                    "agent_profile": lane.profile_template,
                    "session_name": name,
                    "working_directory": spec.workdir,
                },
                json=body,
            )
        except CaoTransportError as exc:
            raise SessionIdentityUncertainError(name, str(exc)) from exc
        self._raise_for_status(response, f"launch session {name!r}")

        terminal = response.json()
        launched = dataclasses.replace(
            metadata,
            session_name=terminal["session_name"],
            terminal_id=terminal["id"],
        )
        self._register(launched)
        return SessionHandle(session_id=launched.session_name, lane=lane.lane)

    # --- Work submission ---------------------------------------------------

    def submit_work(self, handle: SessionHandle, message: str) -> None:
        """Send work to the session and return without waiting for a result.

        Used for both the first task and every follow-up: CAO keeps the
        session alive between messages, so a follow-up is the same call. On
        :class:`CaoTransportError` the delivery outcome is unknown — observe
        the session before deciding whether to resend, since a resend can
        double-apply a side-effectful instruction.
        """
        if not message:
            raise ValueError("message must be non-empty")
        metadata = self._metadata_for(handle)
        response = self._request(
            "POST",
            f"/terminals/{metadata.terminal_id}/input",
            params={"message": message},
        )
        self._raise_for_status(response, f"submit work to session {metadata.session_name!r}")
        # New work invalidates any idle streak accumulated by the previous turn.
        self._idle_readings[metadata.session_name] = 0

    # --- Observation -------------------------------------------------------

    def observe(self, handle: SessionHandle, *, now: datetime | None = None) -> SessionObservation:
        """Return the session's normalized lifecycle state."""
        metadata = self._metadata_for(handle)
        response = self._request("GET", f"/terminals/{metadata.terminal_id}")
        if response.status_code == 404:
            self._idle_readings.pop(metadata.session_name, None)
            return SessionObservation(
                metadata=metadata,
                state="disappeared",
                detail=f"CAO no longer knows terminal {metadata.terminal_id!r}",
            )
        self._raise_for_status(response, f"observe session {metadata.session_name!r}")

        status = response.json().get("status")
        state = self._lifecycle(metadata.session_name, status)
        if state not in TERMINAL_LIFECYCLE_STATES and self._is_expired(metadata, now):
            return SessionObservation(
                metadata=metadata,
                state="timed_out",
                cao_status=status,
                detail=(
                    f"session exceeded {self._config.session_timeout_seconds}s in state {state!r}"
                ),
            )
        return SessionObservation(metadata=metadata, state=state, cao_status=status)

    def poll_session(self, handle: SessionHandle) -> LaneResult | None:
        """Return the lane result once the session has finished, else ``None``."""
        observation = self.observe(handle)
        if not observation.is_terminal:
            return None
        if observation.state == "disappeared":
            output = observation.detail
        else:
            output = self.final_output(handle) or observation.detail
        return LaneResult(
            session=handle,
            exit_code=0 if observation.state == "completed" else 1,
            output_summary=output,
            changed_files=[],
        )

    def final_output(self, handle: SessionHandle) -> str:
        """Return the agent's last response as CAO extracted it.

        CAO owns the extraction; aipro receives a structured field and never
        inspects the underlying terminal.
        """
        metadata = self._metadata_for(handle)
        response = self._request(
            "GET",
            f"/terminals/{metadata.terminal_id}/output",
            params={"mode": "last"},
        )
        self._raise_for_status(response, f"read output of session {metadata.session_name!r}")
        return response.json().get("output", "")

    # --- Reconcile / teardown ----------------------------------------------

    def adopt_session(self, session_name: str) -> SessionObservation:
        """Re-attach to an existing session by its durable name.

        This is the restart path: given only the name a previous process
        recorded, rebuild the full session metadata from CAO and report the
        current lifecycle state, without launching anything.
        """
        metadata = self._lookup_session(session_name)
        if metadata is None:
            raise CaoSessionNotFoundError(f"CAO has no session named {session_name!r}")
        self._register(metadata)
        return self.observe(
            SessionHandle(session_id=metadata.session_name, lane=metadata.lane.lane)
        )

    def terminate_session(self, handle: SessionHandle) -> None:
        """Delete the session and drop its local bookkeeping.

        A session CAO has already forgotten is treated as terminated: cleanup
        is idempotent so a crash between kill and record-keeping is recoverable.
        """
        metadata = self._metadata_for(handle)
        response = self._request("DELETE", f"/sessions/{metadata.session_name}")
        if response.status_code != 404:
            self._raise_for_status(response, f"terminate session {metadata.session_name!r}")
        self._sessions.pop(metadata.session_name, None)
        self._idle_readings.pop(metadata.session_name, None)

    # --- Internals ---------------------------------------------------------

    def _registered_lane(self, lane: LaneIdentity) -> LaneIdentity:
        registered = self._lanes.get(lane.lane)
        if registered != lane:
            raise ValueError(
                f"SessionSpec lane {lane} disagrees with registered lane {registered}; "
                "the lane registry owns the lane-to-profile binding"
            )
        return registered

    def _lookup_session(self, session_name: str) -> CaoSessionMetadata | None:
        """Rebuild session metadata from CAO, or ``None`` if it does not exist."""
        response = self._request("GET", f"/sessions/{session_name}")
        if response.status_code == 404:
            return None
        self._raise_for_status(response, f"look up session {session_name!r}")

        terminals = response.json().get("terminals") or []
        if not terminals:
            return None
        terminal_id = terminals[0]["id"]

        detail = self._request("GET", f"/terminals/{terminal_id}")
        if detail.status_code == 404:
            return None
        self._raise_for_status(detail, f"look up terminal {terminal_id!r}")

        attribution = detail.json().get("metadata")
        if not attribution:
            raise CaoControlPlaneError(
                f"CAO session {session_name!r} carries no aipro attribution metadata; "
                "it was not launched by this controller and will not be adopted"
            )
        return dataclasses.replace(
            CaoSessionMetadata.from_dict(attribution),
            session_name=session_name,
            terminal_id=terminal_id,
        )

    def _register(self, metadata: CaoSessionMetadata) -> None:
        self._sessions[metadata.session_name] = metadata
        self._idle_readings.setdefault(metadata.session_name, 0)

    def _metadata_for(self, handle: SessionHandle) -> CaoSessionMetadata:
        metadata = self._sessions.get(handle.session_id)
        if metadata is None:
            raise SessionNotRegisteredError(
                f"Session {handle.session_id!r} is not known to this controller; "
                "call adopt_session() first to reconcile it"
            )
        return metadata

    def _lifecycle(self, session_name: str, status: str | None) -> SessionLifecycle:
        if status != "idle":
            self._idle_readings[session_name] = 0
            return _STATUS_LIFECYCLE.get(status or "", "started")
        readings = self._idle_readings.get(session_name, 0) + 1
        self._idle_readings[session_name] = readings
        return "completed" if readings >= self._idle_settle_polls else "running"

    def _is_expired(self, metadata: CaoSessionMetadata, now: datetime | None) -> bool:
        moment = now if now is not None else datetime.now(UTC)
        elapsed = (moment - metadata.launched_at).total_seconds()
        return elapsed > self._config.session_timeout_seconds

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise CaoUnavailableError(f"CAO control plane is unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            raise CaoTransportError(f"{method} {url} failed in flight: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        detail = response.text.strip()
        if response.status_code == 404:
            raise CaoSessionNotFoundError(f"Failed to {action}: {detail}")
        if response.status_code == 409:
            raise SessionBusyError(f"Failed to {action}: {detail}")
        raise CaoControlPlaneError(f"Failed to {action}: HTTP {response.status_code} {detail}")
