# V3 acceptance coverage and remaining gates

Scenarios 1–5 now traverse the real CAO HTTP/controller/executor path,
policy broker, GitHub CI adapter, foreman and serialized workflow state.
Scenario 4 verifies stronger, different-family adjudication of conflicting
claims and has no xfail. Explicit gate overrides remain available for focused
pending/error tests. GitHub, Git and model responses remain external test
boundaries; the fixture no longer offers a scripted lane executor.

Scenarios 7/8 verify actual dispatch against subscription reserves and catalog
promotion expiry. Continuous scheduling tests cover scenario 9 (cold developer
and reviewer recovery) and 12 (sequential issues with three reviewers).
The configurable multi-issue soak checks completed work and duplicate/leaked
side effects; see [V3_SOAK.md](../../../docs/V3_SOAK.md).

Control-plane 429 coverage is distinct from model-provider fallback. A bounded
native Hermes test on the Hermes host successfully switched from a loopback
provider returning 429 to a loopback fallback using invented credentials.
This does not establish live-provider availability or deployed 24/7 service.

Cutover still requires the safety-parity matrix, originating-issue authorization
before resources, stable opt-in cancellation during active work, safe orphan
cleanup with complete runtime/Git inventories, and replacement configuration,
trigger and migration docs. Current fake Git boundaries do not prove OS-level
workspace isolation; native container authority checks provide separate evidence.
V1 remains supported until those gates pass. Closed issue #55 alone is not
acceptance evidence, and historical PR #89 must not be merged as proof.

Legacy findings without run/model provenance remain readable; missing fields
are not retroactively proof of independent adjudication.
