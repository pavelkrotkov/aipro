# V3 acceptance coverage and remaining gates

Scenarios 1–5 now traverse the real CAO HTTP/controller/executor path,
policy broker, GitHub CI adapter, foreman and serialized workflow state.
Scenario 4 verifies stronger, different-family adjudication of conflicting
claims and has no xfail. Explicit gate overrides remain available for focused
pending/error tests. GitHub, Git and model responses remain external test
boundaries; the fixture no longer offers a scripted lane executor.

Scenarios 7/8 verify actual dispatch against subscription reserves and catalog
promotion expiry. Scenario 10 and the safety matrix verify repeated-patch
escalation, fork policy, dispatch budgets, elapsed wall-clock limits and
credential stripping through the real policy path; native Git tests cover
commit/workflow-file limits. Continuous scheduling tests cover scenario 9 (cold developer
and reviewer recovery) and 12 (sequential issues with three reviewers).
The configurable multi-issue soak checks completed work and duplicate/leaked
side effects; see [V3_SOAK.md](../../../docs/V3_SOAK.md).

Control-plane 429 coverage is distinct from model-provider fallback. A bounded
native Hermes test on the Hermes host successfully switched from a loopback
provider returning 429 to a loopback fallback using invented credentials.
This does not establish live-provider availability or deployed 24/7 service.

Originating-issue authorization, trusted machine-state writers, stable opt-in
cancellation (including late parallel dispatch), and protected developer
checkout cleanup now have production guards and boundary tests.

The production TTL sweeper is invoked by the service and the real-Git soak.
Protected allocation records distinguish successful owned creation from intent;
cleanup confirms native lane absence independently of list results. Boundary
tests retain active/human resources and fail closed on changed authority or
ineffective runtime deletion.

The native Git soak also recovers cold developer/reviewer checkpoints without
replaying submissions. Shared CAO teardown confirms disappearance on ordinary,
timeout and replacement stops. Surviving sessions hold scheduler capacity, and
opt-out cancellation stops saved historical lanes after a configuration rename.

Replacement configuration, trigger and migration docs shipped in PR #133 before
P6 retirement. The Git soak exercises native branches, commits, reviewer clones
and local-remote pushes; CAO HTTP/GitHub and model output remain external
fixtures. OS isolation and physical container teardown are separately verified
on native Hermes/CAO. Historical PR #89 is superseded acceptance work, not proof
or an additional implementation to merge. See the migration runbook for the
ordered rollback and tracking-issue reopening procedure.

Legacy findings without run/model provenance remain readable; missing fields
are not retroactively proof of independent adjudication.
