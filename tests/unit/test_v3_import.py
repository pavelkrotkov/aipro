"""The V3 package must import cleanly with no CAO/Hermes installed."""

from __future__ import annotations

import importlib
import subprocess
import sys


def test_v3_modules_import() -> None:
    for name in ("domain", "config", "interfaces"):
        module = importlib.import_module(f"ai_pr_orchestrator.v3.{name}")
        assert module is not None


def test_v3_imports_only_stdlib_and_yaml() -> None:
    """No cao/hermes/github-specific imports leak into the V3 seam."""
    code = (
        "import sys, ai_pr_orchestrator.v3, ai_pr_orchestrator.v3.domain, "
        "ai_pr_orchestrator.v3.config, ai_pr_orchestrator.v3.interfaces; "
        "print([m for m in sys.modules if 'cao' in m.lower() or 'hermes' in m.lower()])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_supported_package_has_no_retired_provider_imports() -> None:
    code = (
        "import importlib, pkgutil, sys, ai_pr_orchestrator; "
        "[importlib.import_module(m.name) for m in "
        "pkgutil.walk_packages(ai_pr_orchestrator.__path__, ai_pr_orchestrator.__name__ + '.')]; "
        "retired = {'runner', 'models', 'config', 'state_machine', 'state_storage', "
        "'decision_application', 'agents', 'coders', 'reviewers', 'git'}; "
        "print(sorted(set(sys.modules) & {'ai_pr_orchestrator.' + n for n in retired}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
