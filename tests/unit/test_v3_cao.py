"""Tests for the V3 CAO control-plane adapter, against a faked HTTP transport."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from ai_pr_orchestrator.v3 import cao as cao_module
from ai_pr_orchestrator.v3.cao import (
    CaoAdoptionMismatchError,
    CaoControlPlaneError,
    CaoSessionController,
    CaoSessionMetadata,
    CaoSessionNotFoundError,
    CaoSessionNotReadyError,
    CaoUnavailableError,
    SessionBusyError,
    SessionIdentityUncertainError,
    SessionNotRegisteredError,
    session_name_for,
)
from ai_pr_orchestrator.v3.config import CAOControlPlaneConfig
from ai_pr_orchestrator.v3.domain import LaneIdentity, ModelAssignment
from ai_pr_orchestrator.v3.interfaces import (
    CAOSessionController,
    LaneExecutionContext,
    ModelLease,
    SessionSpec,
)
from ai_pr_orchestrator.v3.lanes import DEVELOPER_LANE, LaneRegistry

BASE = "http://cao.test:9889"
RUN_ID = "run-1"
SESSION = "cao-aipro-run-1-developer"
TERMINAL = "a1b2c3d4"
WORKDIR = "/worktrees/issue-1"


def make_config(**overrides):
    return CAOControlPlaneConfig(base_url=BASE, session_timeout_seconds=60, **overrides)


def make_controller(**kwargs):
    return CaoSessionController(make_config(), LaneRegistry.default(), **kwargs)


def make_spec(*, command="implement the change", env=None, lane=None):
    lane = lane if lane is not None else LaneRegistry.default().get(DEVELOPER_LANE)
    return SessionSpec(
        lane=lane,
        run_id=RUN_ID,
        workdir=WORKDIR,
        env=env if env is not None else {},
        context=LaneExecutionContext(run_id=RUN_ID, round_id="round-2", work_item_id="wi-9"),
        command=command,
        model_lease=ModelLease(
            lease_id="lease-1",
            assignment=ModelAssignment(lane=DEVELOPER_LANE, model_ref="high-capability"),
        ),
    )


def terminal_payload(*, status=None, metadata=None, terminal_id=TERMINAL):
    return {
        "id": terminal_id,
        "name": f"developer-{terminal_id}",
        "session_name": SESSION,
        "agent_profile": "aipro-developer",
        "status": status,
        "metadata": metadata,
    }


def attribution_payload():
    return CaoSessionMetadata(
        session_name=SESSION,
        terminal_id=TERMINAL,
        lane=LaneRegistry.default().get(DEVELOPER_LANE),
        workdir=WORKDIR,
        context=LaneExecutionContext(run_id=RUN_ID, round_id="round-2", work_item_id="wi-9"),
        model_assignment=ModelAssignment(lane=DEVELOPER_LANE, model_ref="high-capability"),
    ).to_dict()


def launch(controller, respx_mock, **spec_kwargs):
    """Drive a controller through a successful launch and return the handle."""
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    respx_mock.post(f"{BASE}/sessions").mock(
        return_value=httpx.Response(201, json=terminal_payload())
    )
    return controller.start_session(make_spec(**spec_kwargs))


def status_route(respx_mock, *statuses):
    return respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        side_effect=[httpx.Response(200, json=terminal_payload(status=s)) for s in statuses]
    )


# --- Naming / conformance --------------------------------------------------


def test_session_name_is_deterministic_for_a_run_and_lane():
    assert session_name_for(RUN_ID, DEVELOPER_LANE) == SESSION
    assert session_name_for(RUN_ID, DEVELOPER_LANE) == session_name_for(RUN_ID, DEVELOPER_LANE)


def test_session_name_is_sanitized_and_length_bounded():
    name = session_name_for("owner/repo#42:very-long-" + "x" * 80, "architecture-reviewer")

    assert len(name) <= 64
    assert name.startswith("cao-aipro-")
    assert all(char.isalnum() or char in "_-" for char in name)


def test_long_names_stay_distinct_after_truncation():
    a = session_name_for("x" * 100 + "-alpha", DEVELOPER_LANE)
    b = session_name_for("x" * 100 + "-beta", DEVELOPER_LANE)

    assert a != b


def test_controller_satisfies_the_v3_session_protocol():
    assert isinstance(make_controller(), CAOSessionController)


def test_adapter_imports_no_provider_specific_code():
    source = Path(cao_module.__file__).read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines() if line.lstrip().startswith(("import ", "from "))
    ]

    assert not [line for line in import_lines if "coders" in line or "reviewers" in line]


def test_idle_settle_polls_must_be_positive():
    with pytest.raises(ValueError, match="idle_settle_polls"):
        make_controller(idle_settle_polls=0)


# --- Launch ----------------------------------------------------------------


def test_start_session_launches_named_session_in_the_lane_profile(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))
    route = respx_mock.post(f"{BASE}/sessions").mock(
        return_value=httpx.Response(201, json=terminal_payload())
    )

    handle = controller.start_session(make_spec(env={"LANE_HOME": "/lanes/developer"}))

    assert handle.session_id == SESSION
    assert handle.lane == DEVELOPER_LANE
    request = route.calls.last.request
    assert dict(request.url.params) == {
        "agent_profile": "aipro-developer",
        "session_name": SESSION,
        "working_directory": WORKDIR,
    }
    import json

    body = json.loads(request.content)
    assert body["initial_message"] == "implement the change"
    assert body["env_vars"] == {"LANE_HOME": "/lanes/developer"}
    assert body["group"] == [RUN_ID, DEVELOPER_LANE]
    assert body["metadata"]["run_id"] == RUN_ID
    assert body["metadata"]["round_id"] == "round-2"
    assert body["metadata"]["work_item_id"] == "wi-9"
    assert body["metadata"]["workdir"] == WORKDIR
    assert body["metadata"]["lane"]["profile_template"] == "aipro-developer"
    assert body["metadata"]["model_assignment"] == {
        "lane": DEVELOPER_LANE,
        "model_ref": "high-capability",
    }


def test_start_session_without_a_command_omits_the_initial_message(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))
    route = respx_mock.post(f"{BASE}/sessions").mock(
        return_value=httpx.Response(201, json=terminal_payload())
    )

    controller.start_session(make_spec(command=None))

    import json

    assert "initial_message" not in json.loads(route.calls.last.request.content)


def test_start_session_adopts_an_existing_session_instead_of_duplicating_it(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(metadata=attribution_payload()))
    )
    launch_route = respx_mock.post(f"{BASE}/sessions")

    handle = controller.start_session(make_spec())

    assert handle.session_id == SESSION
    assert not launch_route.called


def test_start_session_rejects_a_lane_the_registry_does_not_know(respx_mock):
    controller = make_controller()
    stranger = LaneIdentity(lane="stranger", role="worker", profile_template="stranger-profile")
    spec = SessionSpec(
        lane=stranger,
        run_id=RUN_ID,
        workdir=WORKDIR,
        env={},
        context=LaneExecutionContext(run_id=RUN_ID),
    )

    with pytest.raises(LookupError, match="Unknown lane 'stranger'"):
        controller.start_session(spec)


def test_start_session_rejects_a_lane_bound_to_a_different_profile(respx_mock):
    controller = make_controller()
    impostor = LaneIdentity(lane=DEVELOPER_LANE, role="worker", profile_template="somewhere-else")
    spec = SessionSpec(
        lane=impostor,
        run_id=RUN_ID,
        workdir=WORKDIR,
        env={},
        context=LaneExecutionContext(run_id=RUN_ID),
    )

    with pytest.raises(ValueError, match="lane registry owns the lane-to-profile binding"):
        controller.start_session(spec)


# --- Launch failure semantics ----------------------------------------------


def test_launch_that_loses_its_answer_never_reports_a_retryable_failure(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))
    respx_mock.post(f"{BASE}/sessions").mock(side_effect=httpx.ReadTimeout("no answer"))

    with pytest.raises(SessionIdentityUncertainError) as excinfo:
        controller.start_session(make_spec())

    assert excinfo.value.session_name == SESSION
    assert "reconcile by name instead of retrying" in str(excinfo.value)


def test_launch_that_never_reached_cao_is_reported_as_unavailable(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with pytest.raises(CaoUnavailableError):
        controller.start_session(make_spec())


def test_launch_rejected_by_cao_is_a_definite_failure(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))
    respx_mock.post(f"{BASE}/sessions").mock(
        return_value=httpx.Response(400, text="invalid working_directory")
    )

    with pytest.raises(CaoControlPlaneError) as excinfo:
        controller.start_session(make_spec())

    assert not isinstance(excinfo.value, SessionIdentityUncertainError)
    assert "invalid working_directory" in str(excinfo.value)


# --- Lifecycle observation -------------------------------------------------


@pytest.mark.parametrize(
    ("cao_status", "expected"),
    [
        ("processing", "running"),
        ("completed", "completed"),
        ("waiting_user_answer", "blocked"),
        ("error", "failed"),
        ("unknown", "started"),
        (None, "started"),
    ],
)
def test_observe_normalizes_cao_status(respx_mock, cao_status, expected):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, cao_status)

    observation = controller.observe(handle)

    assert observation.state == expected
    assert observation.cao_status == cao_status
    assert observation.metadata.terminal_id == TERMINAL


def test_idle_becomes_completed_only_once_the_reading_settles(respx_mock):
    controller = make_controller(idle_settle_polls=3)
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "processing", "idle", "idle", "idle")

    assert controller.observe(handle).state == "running"
    assert controller.observe(handle).state == "running"
    assert controller.observe(handle).state == "running"
    assert controller.observe(handle).state == "completed"


def test_idle_on_a_session_that_never_ran_does_not_complete_it(respx_mock):
    controller = make_controller(idle_settle_polls=2)
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "idle", "idle", "idle", "idle", "idle")

    states = [controller.observe(handle).state for _ in range(5)]

    # A freshly created terminal is idle before it picks work up; idle alone
    # must never complete the lane while no activity has been observed.
    assert states == ["running", "running", "running", "running", "running"]


def test_activity_restarts_the_idle_streak(respx_mock):
    controller = make_controller(idle_settle_polls=2)
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "idle", "processing", "idle", "idle")

    states = [controller.observe(handle).state for _ in range(4)]

    assert states == ["running", "running", "running", "completed"]


def test_new_work_restarts_the_idle_streak(respx_mock):
    controller = make_controller(idle_settle_polls=2)
    handle = launch(controller, respx_mock)
    respx_mock.post(f"{BASE}/terminals/{TERMINAL}/input").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    status_route(respx_mock, "idle", "idle", "idle")

    assert controller.observe(handle).state == "running"
    controller.submit_work(handle, "one more thing")

    assert controller.observe(handle).state == "running"
    assert controller.observe(handle).state == "completed"


def test_a_session_cao_forgot_is_reported_as_disappeared(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(return_value=httpx.Response(404))

    observation = controller.observe(handle)

    assert observation.state == "disappeared"
    assert observation.is_terminal


def test_a_session_past_its_budget_is_reported_as_timed_out(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "processing")

    observation = controller.observe(handle, now=datetime.now(UTC) + timedelta(seconds=61))

    assert observation.state == "timed_out"
    assert observation.cao_status == "processing"


def test_a_finished_session_is_not_retroactively_timed_out(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "completed")

    observation = controller.observe(handle, now=datetime.now(UTC) + timedelta(seconds=600))

    assert observation.state == "completed"


# --- Results ---------------------------------------------------------------


def test_final_output_comes_from_cao_last_response_view(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    route = respx_mock.get(f"{BASE}/terminals/{TERMINAL}/output").mock(
        return_value=httpx.Response(200, json={"output": "patch applied", "mode": "last"})
    )

    assert controller.final_output(handle) == "patch applied"
    assert dict(route.calls.last.request.url.params) == {"mode": "last"}


def test_poll_session_returns_nothing_while_the_lane_is_working(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "processing")

    assert controller.poll_session(handle) is None


def test_poll_session_returns_a_successful_result_on_completion(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "completed")
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}/output").mock(
        return_value=httpx.Response(200, json={"output": "patch applied", "mode": "last"})
    )

    result = controller.poll_session(handle)

    assert result is not None
    assert result.exit_code == 0
    assert result.output_summary == "patch applied"
    assert result.session == handle


def test_poll_session_reports_a_crashed_lane_as_a_failure(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    status_route(respx_mock, "error")
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}/output").mock(
        return_value=httpx.Response(200, json={"output": "traceback", "mode": "last"})
    )

    result = controller.poll_session(handle)

    assert result is not None
    assert result.exit_code == 1
    assert result.output_summary == "traceback"


def test_poll_session_reports_a_disappeared_session_without_reading_output(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(return_value=httpx.Response(404))
    output_route = respx_mock.get(f"{BASE}/terminals/{TERMINAL}/output")

    result = controller.poll_session(handle)

    assert result is not None
    assert result.exit_code == 1
    assert not output_route.called


def test_poll_session_reports_a_timeout_without_reading_output(respx_mock):
    controller = CaoSessionController(
        CAOControlPlaneConfig(base_url=BASE, session_timeout_seconds=0),
        LaneRegistry.default(),
    )
    handle = launch(controller, respx_mock)
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(status="processing"))
    )
    output_route = respx_mock.get(f"{BASE}/terminals/{TERMINAL}/output").mock(
        side_effect=httpx.ConnectError("extraction endpoint gone")
    )

    result = controller.poll_session(handle)

    assert result is not None
    assert result.exit_code == 1
    assert "exceeded" in result.output_summary
    assert not output_route.called


# --- Work submission and follow-ups ----------------------------------------


def test_follow_up_work_goes_to_the_same_session(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    route = respx_mock.post(f"{BASE}/terminals/{TERMINAL}/input").mock(
        return_value=httpx.Response(200, json={"success": True})
    )

    controller.submit_work(handle, "address the review findings")
    controller.submit_work(handle, "and rerun the tests")

    assert len(route.calls) == 2
    messages = [dict(call.request.url.params)["message"] for call in route.calls]
    assert messages == ["address the review findings", "and rerun the tests"]


def test_submitting_work_to_a_busy_session_is_reported_not_swallowed(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    respx_mock.post(f"{BASE}/terminals/{TERMINAL}/input").mock(
        return_value=httpx.Response(409, text="terminal input blocked")
    )

    with pytest.raises(SessionBusyError, match="terminal input blocked"):
        controller.submit_work(handle, "hurry up")


def test_empty_work_is_rejected_before_it_reaches_cao(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    route = respx_mock.post(f"{BASE}/terminals/{TERMINAL}/input")

    with pytest.raises(ValueError, match="message must be non-empty"):
        controller.submit_work(handle, "")

    assert not route.called


# --- Reconcile after restart ------------------------------------------------


def test_a_restarted_process_adopts_the_session_from_its_durable_name(respx_mock):
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(
            200,
            json=terminal_payload(status="processing", metadata=attribution_payload()),
        )
    )
    launch_route = respx_mock.post(f"{BASE}/sessions")

    # A fresh controller, as after a restart: no in-memory session bookkeeping.
    observation = make_controller().adopt_session(SESSION)

    assert not launch_route.called
    assert observation.state == "running"
    metadata = observation.metadata
    assert metadata.session_name == SESSION
    assert metadata.terminal_id == TERMINAL
    assert metadata.lane.lane == DEVELOPER_LANE
    assert metadata.profile_template == "aipro-developer"
    assert metadata.workdir == WORKDIR
    assert metadata.parent_run_id == RUN_ID
    assert metadata.context.round_id == "round-2"
    assert metadata.model_assignment == ModelAssignment(
        lane=DEVELOPER_LANE, model_ref="high-capability"
    )


def test_an_adopted_session_can_be_driven_without_relaunching(respx_mock):
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(metadata=attribution_payload()))
    )
    input_route = respx_mock.post(f"{BASE}/terminals/{TERMINAL}/input").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    controller = make_controller()
    observation = controller.adopt_session(SESSION)

    controller.submit_work(
        cao_module.SessionHandle(
            session_id=observation.metadata.session_name,
            lane=observation.metadata.lane.lane,
        ),
        "continue",
    )

    assert input_route.called


def test_adopting_an_unknown_session_does_not_invent_one(respx_mock):
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))
    launch_route = respx_mock.post(f"{BASE}/sessions")

    with pytest.raises(CaoSessionNotFoundError, match=SESSION):
        make_controller().adopt_session(SESSION)

    assert not launch_route.called


def test_a_session_without_aipro_attribution_is_not_adopted(respx_mock):
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(metadata=None))
    )

    with pytest.raises(CaoControlPlaneError, match="no aipro attribution metadata"):
        make_controller().adopt_session(SESSION)


def test_driving_an_unadopted_session_asks_the_caller_to_reconcile(respx_mock):
    controller = make_controller()
    handle = cao_module.SessionHandle(session_id=SESSION, lane=DEVELOPER_LANE)

    with pytest.raises(SessionNotRegisteredError, match="adopt_session"):
        controller.observe(handle)


def test_session_metadata_round_trips_through_its_serialized_form():
    original = CaoSessionMetadata.from_dict(attribution_payload())

    assert CaoSessionMetadata.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("missing", ["lane", "workdir", "run_id"])
def test_incomplete_attribution_metadata_is_a_controlled_error(missing):
    payload = attribution_payload()
    del payload[missing]

    with pytest.raises(CaoControlPlaneError, match=missing) as excinfo:
        CaoSessionMetadata.from_dict(payload)

    assert not isinstance(excinfo.value, KeyError)


def test_an_adopted_session_is_refused_when_attribution_disagrees(respx_mock):
    controller = make_controller()
    spec = make_spec()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    foreign = attribution_payload()
    foreign["workdir"] = "/worktrees/someone-elses-issue"
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(metadata=foreign))
    )

    with pytest.raises(CaoAdoptionMismatchError, match="workdir"):
        controller.adopt_session(SESSION, spec)

    assert SESSION not in controller._sessions


def test_start_session_refuses_to_adopt_a_session_attributed_to_another_run(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": [{"id": TERMINAL}]})
    )
    foreign = attribution_payload()
    foreign["run_id"] = "run-999"
    respx_mock.get(f"{BASE}/terminals/{TERMINAL}").mock(
        return_value=httpx.Response(200, json=terminal_payload(metadata=foreign))
    )
    launch_route = respx_mock.post(f"{BASE}/sessions")

    with pytest.raises(CaoAdoptionMismatchError, match="run_id"):
        controller.start_session(make_spec())

    assert not launch_route.called


def test_a_provisioning_session_is_not_treated_as_absent(respx_mock):
    controller = make_controller()
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": []})
    )
    launch_route = respx_mock.post(f"{BASE}/sessions")

    with pytest.raises(CaoSessionNotReadyError, match="reconcile again"):
        controller.start_session(make_spec())

    assert not launch_route.called


def test_adopting_a_session_whose_terminal_is_still_provisioning_is_not_found(respx_mock):
    respx_mock.get(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"terminals": []})
    )

    with pytest.raises(CaoSessionNotReadyError):
        make_controller().adopt_session(SESSION)


def test_sanitized_names_carry_a_digest_so_lossy_names_stay_distinct():
    a = session_name_for("owner/foo/bar", DEVELOPER_LANE)
    b = session_name_for("owner:foo:bar", DEVELOPER_LANE)

    assert a != b
    assert a.startswith("cao-aipro-owner-foo-bar-")
    assert len(a) <= 64
    # A name that needed no sanitization stays digest-free.
    assert session_name_for(RUN_ID, DEVELOPER_LANE) == SESSION


# --- Teardown --------------------------------------------------------------


def test_terminate_deletes_the_session_and_forgets_it(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    route = respx_mock.delete(f"{BASE}/sessions/{SESSION}").mock(
        return_value=httpx.Response(200, json={"success": True, "deleted": [SESSION]})
    )

    controller.terminate_session(handle)

    assert route.called
    with pytest.raises(SessionNotRegisteredError):
        controller.observe(handle)


def test_terminating_an_already_gone_session_is_not_an_error(respx_mock):
    controller = make_controller()
    handle = launch(controller, respx_mock)
    respx_mock.delete(f"{BASE}/sessions/{SESSION}").mock(return_value=httpx.Response(404))

    controller.terminate_session(handle)


def test_controller_closes_its_client_on_exit():
    client = httpx.Client(base_url=BASE)
    with CaoSessionController(make_config(), LaneRegistry.default(), client=client):
        pass

    assert client.is_closed
