---
name: aipro-foreman
description: Operate the persistent AIPRO V3 scheduler and inspect recovery requirements.
---

Read `docs/V3_FOREMAN.md` in the checkout named by `AIPRO_CHECKOUT` before
operating the service. `AIPRO_V3_CONFIG` is the shared policy file; its `cao`
section supplies control-plane access. Keep that endpoint coordinator-only.

Use `systemctl --user start|stop|restart aipro-foreman` for lifecycle control.
Inspect `systemctl --user status aipro-foreman` and
`journalctl --user -u aipro-foreman -n 100 --no-pager` for current outcomes or
`reconciliation_required` errors. Start one service per repository. It
automatically reconciles active work before selecting the next ready issue.

For an uncertain outcome, inspect the named GitHub issue's machine state,
recorded branch/PR/head, and attributed CAO session. Use `aipro reconcile` as
a diagnostic plan; its unsupported `--apply` is not a recovery shortcut.
Do not delete attribution, reset a fingerprint, change a lease, or resend a
prompt to force progress. Report the concrete unresolved identity or effect.

Needs-human work remains stopped until an operator explicitly changes its
GitHub workflow state. Apply valid implementation/review fixes through the
developer lane and deterministic policy; never ask review bots to implement.
