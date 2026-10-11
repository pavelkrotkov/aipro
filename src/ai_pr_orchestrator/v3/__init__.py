"""Deterministic issue-to-PR policy over GitHub, CAO and Hermes.

GitHub holds authoritative business state; CAO owns agent lifecycle. The
coordinator schedules work, binds catalog models, enforces review/CI/safety
policy, performs native Git mutations and reconciles uncertain effects.
Hermes owns worker reasoning and provider fallback in separate lane homes.
See docs/V3_ARCHITECTURE.md and docs/V3_FOREMAN.md for deployment boundaries.
"""
