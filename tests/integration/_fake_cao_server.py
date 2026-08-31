"""In-process CAO control-plane fake for V3 E2E tests (issue #55, P1).

The real ``CaoSessionController`` in :mod:`ai_pr_orchestrator.v3.cao` speaks a
small, documented HTTP surface: ``POST /sessions``, ``POST /terminals/{tid}/input``,
``GET /terminals/{tid}``, ``GET /terminals/{tid}/output``, ``GET /sessions/{name}``,
``PATCH /terminals/{tid}/metadata``, ``DELETE /sessions/{name}``. This module
implements that surface using only ``http.server`` from the standard library so
the foreman can be exercised end-to-end against a deterministic CAO without
provisioning a real ``cao-server``.

Three properties are load-bearing and mirror the real CAO contract:

- The session **identity is byte-identical** to the name the controller asked
  for (``session_name_for(run_id, lane)``); the fake returns the requested name
  in the launch response, so a second ``start_session`` for the same spec
  adopts instead of creating a twin.
- The **metadata round-trips** through the fake exactly as it does in real
  CAO: ``POST /sessions`` body carries ``metadata``, and ``GET /terminals/{tid}``
  returns it. That is the only way ``CaoSessionController.adopt_session`` can
  rebuild its in-memory ``CaoSessionMetadata`` after a process restart.
- The **status sequence is scripted per session**, not driven by a real
  agent. Each session has a ``status_sequence``; ``GET /terminals/{tid}``
  returns the next element and stops at the last.

The fake also exposes **fault-injection hooks** (``add_fault``) so the soak
harness can inject 429 / 5xx / transport resets without changing the
production code path under test. These hooks return ``None`` (no fault) by
default.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

#: Default lifecycle status names a real CAO terminal reports. The fake uses
#: these as building blocks for the per-session status sequence.
STATUS_STARTED = "started"
STATUS_PROCESSING = "processing"
STATUS_WAITING_USER = "waiting_user_answer"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"
STATUS_IDLE = "idle"

#: Status sequence used when a test does not specify one. Walks the terminal
#: from started -> processing -> idle -> idle -> idle, which the controller's
#: idle-settle rule maps onto the normalized ``completed`` lifecycle after
#: enough idle polls (``_idle_settle_polls``).
DEFAULT_STATUS_SEQUENCE: tuple[str, ...] = (
    STATUS_STARTED,
    STATUS_PROCESSING,
    STATUS_IDLE,
    STATUS_IDLE,
    STATUS_IDLE,
)


@dataclass
class _SessionState:
    """In-memory state for one fake CAO session."""

    session_name: str
    terminal_id: str
    workdir: str
    agent_profile: str
    initial_message: str | None
    metadata: dict[str, Any]
    status_sequence: Sequence[str]
    output: str = ""
    status_index: int = 0
    exhausted: bool = False
    deleted: bool = False


@dataclass
class FaultSpec:
    """Per-request fault: when ``matches(method, path)`` is true, return
    ``status_code`` (an HTTP status to send instead of the real response) or
    ``transport_reset=True`` (close the connection mid-flight, which CAO's
    adapter maps to ``CaoTransportError``). ``status_code=200`` is a no-op.

    ``commit_then_reset=True`` extends ``transport_reset``: the handler still
    applies the normal state mutation (creating the session, advancing the
    status cursor, etc.) and only then closes the connection without writing
    a response. This exercises the committed-but-unacknowledged branch where
    the controller treats the request as uncertain and a retry would
    duplicate work."""

    method: str
    path_prefix: str
    status_code: int | None = None
    transport_reset: bool = False
    delay_seconds: float = 0.0
    commit_then_reset: bool = False

    def matches(self, method: str, path: str) -> bool:
        if self.method != method:
            return False
        return path.startswith(self.path_prefix)


@dataclass
class _PendingFault:
    """Result of matching a fault: what the handler should do."""

    status_code: int | None = None
    transport_reset: bool = False
    delay_seconds: float = 0.0
    commit_then_reset: bool = False


def _find_by_terminal(sessions: dict[str, _SessionState], terminal_id: str) -> _SessionState | None:
    for state in sessions.values():
        if state.terminal_id == terminal_id:
            return state
    return None


class _FakeHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` subclass that carries a typed reference to
    the :class:`FakeCAOServer` it serves. Avoids the per-request
    ``# type: ignore[attr-defined]`` on ``self.server.fake`` that the
    standard library makes otherwise unavoidable.
    """

    fake: FakeCAOServer | None = None


class FakeCAOServer:
    """In-process CAO control-plane fake.

    Use as a context manager: the server binds on ``127.0.0.1`` to a free port
    and runs the HTTP handler in a background thread. Call per-session
    helpers (:meth:`set_status_sequence`, :meth:`set_output`) **before** the
    controller launches the session, since the launch handler snapshots the
    sequence into session state. :meth:`add_fault` injects HTTP status
    overrides or transport resets for soak scenarios.

    Example::

        with FakeCAOServer() as cao:
            cao.set_status_sequence(name, [PROCESSING, IDLE, IDLE, IDLE])
            cao.set_output(name, "TOKEN")
            controller = CaoSessionController(
                CAOControlPlaneConfig(base_url=cao.url), LaneRegistry.default()
            )
            handle = controller.start_session(spec)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        # Per-session status sequence overrides; fall back to DEFAULT.
        self._status_overrides: dict[str, Sequence[str]] = {}
        # Per-session output payloads returned by /terminals/{tid}/output.
        # Stored both pre-launch (so set_output() before start_session
        # still wins) and on the live session state.
        self._outputs: dict[str, str] = {}
        # Per-request fault specs.
        self._faults: list[FaultSpec] = []
        # Monotonically increasing counter for terminal IDs. Independent
        # of len(self._sessions) so delete-then-relaunch (or a launch
        # failing the metadata check, finding #8) cannot reuse or skip
        # an ID that some other controller might still be talking to.
        self._next_terminal_seq = 1
        # Pre-bound socket so we can read the chosen port back to the caller.
        self._httpd = _FakeHTTPServer(("127.0.0.1", 0), _FakeHandler)
        self._httpd.fake = self
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def set_status_sequence(self, session_name: str, sequence: Sequence[str]) -> None:
        """Override the per-session status sequence returned by
        ``GET /terminals/{tid}``. Must be called before the session launches."""
        with self._lock:
            self._status_overrides[session_name] = tuple(sequence)

    def set_output(self, session_name: str, output: str) -> None:
        """Set the text returned by ``GET /terminals/{tid}/output?mode=last``.

        Safe to call before the session is launched: the launch handler
        snapshots the pre-set output into the new session state.
        """
        with self._lock:
            self._outputs[session_name] = output
            state = self._sessions.get(session_name)
            if state is not None:
                state.output = output

    def _output_for(self, session_name: str) -> str:
        return self._outputs.get(session_name, "")

    def add_fault(self, fault: FaultSpec) -> None:
        """Register a per-request fault injected before normal handling."""
        with self._lock:
            self._faults.append(fault)

    def _status_sequence_for(self, session_name: str) -> Sequence[str]:
        return self._status_overrides.get(session_name, DEFAULT_STATUS_SEQUENCE)

    def _match_fault(self, method: str, path: str) -> _PendingFault | None:
        with self._lock:
            faults = list(self._faults)
        for fault in faults:
            if not fault.matches(method, path):
                continue
            return _PendingFault(
                status_code=fault.status_code,
                transport_reset=fault.transport_reset,
                delay_seconds=fault.delay_seconds,
                commit_then_reset=fault.commit_then_reset,
            )
        return None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="FakeCAOServer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> FakeCAOServer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class _FakeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the fake CAO control plane.

    The shared :class:`FakeCAOServer` instance is reached via
    :attr:`_fake` (a typed property): ``BaseHTTPRequestHandler.__init__``
    already stores the ``ThreadingHTTPServer`` on ``self.server``, and
    ``FakeCAOServer`` uses a typed :class:`_FakeHTTPServer` subclass so the
    attribute is visible to static type-checkers without per-access ignores.
    """

    # Silence stderr access-log noise; tests assert on observable state.
    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def _fake(self) -> FakeCAOServer:
        server = self.server  # type: ignore[assignment]
        assert isinstance(server, _FakeHTTPServer)
        fake = server.fake
        assert fake is not None
        return fake

    # Per-request flag: when the matched FaultSpec has commit_then_reset,
    # the dispatch method must run its normal mutation then drop the
    # response. Without this, the committed-but-unacknowledged branch is
    # untestable (see finding #7 in PR #70).
    _commit_then_reset: bool = False

    def _maybe_drop_response(self) -> None:
        if self._commit_then_reset:
            # Flush any bytes the handler already wrote, then close the
            # socket so the client sees a transport error rather than a
            # successful response. If nothing was written yet, this also
            # closes before any response can be sent.
            try:
                self.wfile.flush()
            except Exception:
                pass
            self.connection.close()

    # -- Helpers --------------------------------------------------------

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_json(self, status: int, body: Any) -> None:
        if self._commit_then_reset:
            return
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_text(self, status: int, body: str) -> None:
        if self._commit_then_reset:
            return
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _query_params(self) -> dict[str, list[str]]:
        return parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}

    def _apply_fault_or(self, method: str, path: str) -> _PendingFault | None:
        fault = self._fake._match_fault(method, path)
        if fault is None:
            return None
        if fault.delay_seconds > 0:
            import time

            time.sleep(fault.delay_seconds)
        if fault.transport_reset and not fault.commit_then_reset:
            # Close the underlying socket so the client sees a transport error.
            self.connection.close()
            # Signal the handler to return without writing anything.
            return _PendingFault(transport_reset=True)
        if (
            fault.status_code is not None
            and fault.status_code != 200
            and not fault.commit_then_reset
        ):
            self._write_text(fault.status_code, f"injected fault {fault.status_code}")
            return _PendingFault(status_code=fault.status_code)
        # commit_then_reset (with or without transport_reset/status_code):
        # let the handler perform its normal mutation, then drop the
        # response at the end of dispatch via _finalize_commit_then_reset.
        return fault

    # -- Dispatch -------------------------------------------------------

    def _strip_query(self) -> str:
        return self.path.split("?", 1)[0]

    def do_POST(self) -> None:
        fault = self._apply_fault_or(self.command, self.path)
        if fault is not None and not fault.commit_then_reset:
            return
        self._commit_then_reset = bool(fault and fault.commit_then_reset)
        try:
            body = self._read_body()
            path = self._strip_query()
            if path == "/sessions":
                self._handle_launch(body)
            elif path.endswith("/input"):
                self._handle_input()
            else:
                self._write_text(404, f"unknown POST {self.path}")
        finally:
            self._maybe_drop_response()

    def do_GET(self) -> None:
        fault = self._apply_fault_or(self.command, self.path)
        if fault is not None and not fault.commit_then_reset:
            return
        self._commit_then_reset = bool(fault and fault.commit_then_reset)
        try:
            path = self._strip_query()
            if path.startswith("/sessions/"):
                self._handle_get_session()
            elif "/output" in path:
                self._handle_get_output()
            elif path.startswith("/terminals/"):
                self._handle_get_terminal()
            else:
                self._write_text(404, f"unknown GET {self.path}")
        finally:
            self._maybe_drop_response()

    def do_PATCH(self) -> None:
        fault = self._apply_fault_or(self.command, self.path)
        if fault is not None and not fault.commit_then_reset:
            return
        self._commit_then_reset = bool(fault and fault.commit_then_reset)
        try:
            body = self._read_body() or {}
            # Find the addressed terminal and merge the metadata payload so a
            # controller created after a restart can observe the updated
            # activity_seen marker. The real CAO persists this server-side.
            path = self._strip_query()
            if not path.startswith("/terminals/"):
                self._write_text(404, f"unknown PATCH {self.path}")
                return
            parts = path.split("/")
            # Path is /terminals/{tid}/metadata; the terminal id is the third
            # segment, the literal "metadata" segment must be the fourth.
            if len(parts) != 4 or parts[3] != "metadata":
                self._write_text(404, f"unknown terminal PATCH {self.path}")
                return
            terminal_id = parts[2]
            with self._fake._lock:
                state = _find_by_terminal(self._fake._sessions, terminal_id)
                if state is None or state.deleted:
                    self._write_text(404, f"no terminal {terminal_id}")
                    return
                if isinstance(body, dict):
                    for key, value in body.items():
                        state.metadata[key] = value
            self._write_text(204, "")
        finally:
            self._maybe_drop_response()

    def do_DELETE(self) -> None:
        fault = self._apply_fault_or(self.command, self.path)
        if fault is not None and not fault.commit_then_reset:
            return
        self._commit_then_reset = bool(fault and fault.commit_then_reset)
        try:
            if self._strip_query().startswith("/sessions/"):
                name = self.path[len("/sessions/") :].split("?", 1)[0]
                with self._fake._lock:
                    state = self._fake._sessions.get(name)
                    if state is not None and not state.deleted:
                        state.deleted = True
                        state.exhausted = True
                self._write_text(204, "")
            else:
                self._write_text(404, f"unknown DELETE {self.path}")
        finally:
            self._maybe_drop_response()

    # -- Handlers -------------------------------------------------------

    def _handle_launch(self, body: dict[str, Any] | None) -> None:
        params = self._query_params()
        session_name = params.get("session_name", [""])[0]
        agent_profile = params.get("agent_profile", [""])[0]
        workdir = params.get("working_directory", [""])[0]
        if not session_name:
            self._write_text(400, "session_name is required")
            return
        body = body or {}
        initial_message = body.get("initial_message")
        metadata = body.get("metadata") or {}

        with self._fake._lock:
            existing = self._fake._sessions.get(session_name)
            if existing is not None and not existing.deleted:
                # Re-attach path: same byte-identical name, same metadata.
                self._write_json(
                    200,
                    {
                        "id": existing.terminal_id,
                        "session_name": existing.session_name,
                    },
                )
                return
            terminal_id = f"term-{self._fake._next_terminal_seq:04d}"
            self._fake._next_terminal_seq += 1
            state = _SessionState(
                session_name=session_name,
                terminal_id=terminal_id,
                workdir=workdir,
                agent_profile=agent_profile,
                initial_message=initial_message,
                metadata=metadata,
                status_sequence=self._fake._status_sequence_for(session_name),
                output=self._fake._output_for(session_name),
            )
            self._fake._sessions[session_name] = state

        self._write_json(200, {"id": terminal_id, "session_name": session_name})

    def _handle_input(self) -> None:
        # Path: /terminals/{tid}/input
        parts = self._strip_query().split("/")
        if len(parts) != 4 or parts[3] != "input":
            self._write_text(404, f"unknown input path {self.path}")
            return
        terminal_id = parts[2]
        with self._fake._lock:
            for state in self._fake._sessions.values():
                if state.terminal_id == terminal_id and not state.deleted:
                    # In real CAO, accepted input means the session is
                    # processing again. Replay the scripted lifecycle from
                    # the start so a follow-up observe sees activity
                    # evidence before the controller clears its seen flag
                    # and polls forever waiting for non-idle.
                    if state.exhausted:
                        state.status_index = 0
                        state.exhausted = False
                    elif state.status_index < len(state.status_sequence) - 1:
                        state.status_index += 1
                    self._write_text(204, "")
                    return
        self._write_text(404, f"no session for terminal {terminal_id}")

    def _handle_get_session(self) -> None:
        # Path: /sessions/{name}
        name = self._strip_query()[len("/sessions/") :]
        with self._fake._lock:
            state = self._fake._sessions.get(name)
            if state is None or state.deleted:
                self._write_text(404, f"no session {name}")
                return
            self._write_json(
                200,
                {
                    "session_name": state.session_name,
                    "terminals": [{"id": state.terminal_id}],
                },
            )

    def _handle_get_terminal(self) -> None:
        # Path: /terminals/{tid} (no /output or /input suffix). Optional
        # ?peek=true skips the status-index advance so a metadata-only
        # lookup (e.g. adoption's _lookup_session) does not consume
        # activity evidence the next observe() needs.
        parts = self._strip_query().split("/")
        if len(parts) != 3:
            self._write_text(404, f"unknown terminal path {self.path}")
            return
        terminal_id = parts[2]
        params = self._query_params()
        peek = params.get("peek", ["false"])[0].lower() == "true"
        with self._fake._lock:
            state = _find_by_terminal(self._fake._sessions, terminal_id)
            if state is None or state.deleted:
                self._write_text(404, f"no terminal {terminal_id}")
                return
            if not state.status_sequence:
                # A misconfigured scripted session with no statuses would
                # otherwise raise IndexError on the first observe.
                self._write_text(500, "internal error: empty status sequence")
                return
            status = state.status_sequence[state.status_index]
            if not peek:
                # Advance the cursor so the next observe sees the next status.
                # Once we have walked past the end, the sequence is exhausted and
                # every subsequent observe reports the last status (a real CAO
                # terminal that has settled on "idle" keeps reporting "idle").
                if state.status_index < len(state.status_sequence) - 1:
                    state.status_index += 1
                else:
                    state.exhausted = True
            self._write_json(
                200,
                {
                    "id": state.terminal_id,
                    "session_name": state.session_name,
                    "status": status,
                    "metadata": state.metadata,
                },
            )

    def _handle_get_output(self) -> None:
        # Path: /terminals/{tid}/output?mode=last
        parts = self._strip_query().split("/")
        if len(parts) != 4 or parts[3] != "output":
            self._write_text(404, f"unknown output path {self.path}")
            return
        terminal_id = parts[2]
        with self._fake._lock:
            state = _find_by_terminal(self._fake._sessions, terminal_id)
            if state is None or state.deleted:
                self._write_text(404, f"no terminal {terminal_id}")
                return
            self._write_json(200, {"output": state.output})


@contextmanager
def scripted_sessions(sessions: Iterable[tuple[str, Sequence[str]]]) -> Iterator[FakeCAOServer]:
    """Context manager that starts a ``FakeCAOServer`` and pre-scripts the
    given ``(session_name, status_sequence)`` pairs."""
    server = FakeCAOServer()
    with server:
        for name, sequence in sessions:
            server.set_status_sequence(name, sequence)
        yield server


def find_free_port() -> int:
    """Return an unused TCP port on 127.0.0.1. Useful for tests that want to
    bind their own loopback server without colliding with this fake."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


__all__ = [
    "DEFAULT_STATUS_SEQUENCE",
    "STATUS_COMPLETED",
    "STATUS_ERROR",
    "STATUS_IDLE",
    "STATUS_PROCESSING",
    "STATUS_STARTED",
    "STATUS_WAITING_USER",
    "FakeCAOServer",
    "FaultSpec",
    "find_free_port",
    "scripted_sessions",
]
