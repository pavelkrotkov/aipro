# V3 E2E coverage and remaining gaps

Scenarios 2 and 3 now exercise the real CAO HTTP/controller/executor path,
strict normalized finding/disposition parsing, foreman policy, and serialized
GitHub workflow state. They no longer use the `hybrid_findings` scripted
executor or carry an xfail marker.

- #74/#75 fixed adopted-session follow-up submission and per-turn attribution.
- #76 requires structured reviewer findings or fails closed.
- #87 records actual fix/rebut proposals and explicit independent acceptance,
  preserving both turns and rationale. Empty reviewer output never accepts a
  pending proposal. `test_v3_dispositions.py` covers malformed/stale/self-accepted
  output, rejected proposals, partial persistence and object-loss reconstruction.

The remaining scenario 4 xfail represents broader #51 stronger-family conflict
adjudication. #49 still owns immutable reviewed HEAD and isolated reviewer
laboratories; #53 owns autonomous cold-start scheduling. These tests use a fake
CAO HTTP service and fake external GitHub/Git/broker/gate boundaries; they do
not establish live Hermes reasoning, OS isolation, production deployment, or
#55 cutover/soak acceptance.
