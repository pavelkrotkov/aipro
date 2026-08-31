# Known V3 gaps surfaced by the E2E harness (issue #55)

The E2E scenarios in `tests/integration/e2e/test_scenario_02_*` and
`test_scenario_03_*` / `test_scenario_04_*` are currently marked
`xfail`. The blocking issues are real V3 surface gaps, not test
fudges; once they close, the `xfail` markers come off and the
scenarios turn green.

## Gap A — multi-coder-invocation session attribution mismatch

**Symptom** (one example, scenario 2):
> foreman error: Session `'cao-aipro-…-developer-…'` attribution does
> not match the requested spec: `round_id None != requested 'review-1'`.
> It will not be adopted.

**Root cause:**

1. The foreman loop's `_run_lane` builds a `LaneExecutionContext` with
   `round_id=state.round_id` and passes it through `CaoLaneExecutor`
   to `CaoSessionController.start_session`.
2. `CaoSessionController` adopts an existing session under the
   deterministic name `(run, lane)` if one exists, but refuses
   adoption when the existing session's stored `round_id` does not
   match the requested one (`CaoAdoptionMismatchError`).
3. The first coder call writes the session with `round_id=None` (the
   state's default).
4. The second coder call (a fix round) requests the same session
   name with `round_id="review-1"` (the state now carries the first
   review round's id). Adoption is refused, the foreman escalates,
   and the item is marked `needs_human`.

**Why the unit tests don't catch this:** the foreman's unit tests
use `ScriptedExecutor`, which has no attribution check. The
adjudication logic (severity → fix/defer) is well-tested in
isolation; the E2E round-trip through the real `CaoLaneExecutor`
trips on the strict attribution check that the unit test bypasses.

**Two ways to close the gap** (either is acceptable; pick one):

1. **Reconcile the foreman's session-naming strategy with the
   controller's adoption rules.** Either the foreman's `round_id` in
   the `LaneExecutionContext` for the coder lane should stay
   stable across multiple invocations (matching the first call's
   `round_id=None` always), or the controller's adoption
   validation should not compare `round_id` for the worker lane.
   The deterministic session name `(run, lane)` already implies
   that `round_id` is *not* part of the session identity for worker
   lanes — making the validation skip `round_id` for worker lanes
   is the smaller, more focused change.
2. **Make the session name include the round.** The foreman
   computes a `(run, lane, round)` identity and the controller
   derives the deterministic name from that. This is a larger
   change with knock-on effects on every existing session-name
   lookup; it is the right answer only if multiple coder
   invocations in the same round are also expected to share a
   session (which the current deterministic-name design says they
   should — adoption is the explicit mechanism for that).

**Recommended path:** option 1 (skip `round_id` validation for
worker lanes). The deterministic `(run, lane)` name already
guarantees one session per (run, lane) across the foreman's
lifetime, which is what the restart-recovery path needs; the
`round_id` is round bookkeeping, not session identity.

**Tracking:** pavelkrotkov/aipro#73 (or the corresponding follow-up
issue opened in the same PR wave).

## Gap B — no developer-rebuttal path

**Symptom** (scenario 3):
The foreman adjudicates by severity only (blocker/major → fix;
minor → reply_deferred). There is no path that says "the developer
rebuts a finding → an independent reviewer round decides".

**Root cause:** the `DispositionAction` enum today is `{"fix",
"reply_deferred", "rebut", "accept"}` (verify in
`v3/findings.py` / `v3/foreman.py`), and the foreman never
chooses `"rebut"`; severity is the only signal. A real rebuttal
path needs the foreman to invoke the coder with the finding in
the prompt (already supported), record the coder's reply on the
disposition, then trigger a fresh reviewer round that either
confirms the rebuttal (`"accept"`) or reopens the finding
(`"fix"`).

**Why the unit tests don't catch this:** the foreman's severity
adjudication is a closed loop; the rebuttal is a feature that does
not exist yet.

**How to close the gap:** add a `rebut` disposition action; the
foreman's adjudication rule gains a third option for findings the
coder pushes back on. The `rebut` action requires an extra review
round before the foreman can mark the issue `done`.

**Tracking:** another sub-issue, paired with #73 above or as its
own.
