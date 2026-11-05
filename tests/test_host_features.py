"""A host whose hook registry cannot tell every hook about a stopped call must not run the obligation ledger: without
``tool_blocked`` a follow-up could ask for work that another hook blocked. The check is made at build time, and only when the
host says what it offers (``RunContext.host_features``)."""

import dataclasses
import sys
import types

import pytest

from polisys.adapters import cuga as cuga_adapter
from polisys.hooks import HOST_FEATURES, HookRegistry, host_features_of
from polisys.model import Obligation
from polisys.package import from_build
from polisys.policy_config import PolicyConfig
from polisys.runtime import RuntimeWiringError, build_runtime
from polisys.dry_run import RecordingModels

from .test_producers import CTX, STEPS


def ledger_config(producers=("package",)):
    return PolicyConfig.from_dict(
        {
            "package": "p.json",
            "sources": {"request": bool(producers)},  # the ledger reads the request
            "decisions": {"mode": "off"},
            "checkers": {"rules": "on", "gate": {"mode": "off"}, "verifier": {"mode": "off"}},
            "organizers": {
                "procedure_tools": "off", "guide": "off", "loop": "off", "sop_tracker": "off", "navigator": "off",
                "requirements_tracker": "off",
            },
            "obligations": {"producers": list(producers), "model": "m/o", "item_level": "observe"} if producers else {"producers": []},
        }
    )


def package():
    pkg = from_build(STEPS)
    pkg.obligations = [
        Obligation(
            id="ob_task",
            origin="package",
            scope={"procedure": "p1"},
            effect={"kind": "write", "tools": ["create_task"]},
            level="follow_up",
            source={"section": "p1", "quote": "Create the task."},
        )
    ]
    return pkg


def build(features, producers=("package",)):
    ctx = dataclasses.replace(CTX, host_features=features)
    return build_runtime(ledger_config(producers), package(), ctx, models=RecordingModels())


def test_polisys_own_registry_offers_every_feature():
    assert host_features_of(HookRegistry) == frozenset(HOST_FEATURES)
    assert host_features_of(HookRegistry()) == frozenset(HOST_FEATURES)


def test_an_older_registry_is_seen_only_by_the_method_it_has():
    class Older:
        pass

    class Middle:
        def tool_blocked(self, tool, args, message):
            return None

    assert host_features_of(Older) == frozenset()
    assert host_features_of(Middle) == frozenset({"tool_blocked"})  # failure_report cannot be seen, so it is missing


def test_a_registry_that_lists_features_is_believed_but_only_for_known_names():
    class Declared:
        FEATURES = frozenset({"tool_blocked", "something_new"})

    assert host_features_of(Declared) == frozenset({"tool_blocked"})


def test_the_ledger_is_refused_on_a_host_without_tool_blocked():
    with pytest.raises(RuntimeWiringError) as raised:
        build(frozenset())
    assert "tool_blocked" in str(raised.value) and "obligations" in str(raised.value)


def test_the_ledger_builds_on_a_host_that_has_tool_blocked():
    parts = build(frozenset({"tool_blocked"}))
    assert any(type(h).__name__ == "ObligationLedger" for h in parts.hooks)


def test_a_host_that_is_not_told_is_not_checked():
    assert any(type(h).__name__ == "ObligationLedger" for h in build(None).hooks)


def test_a_profile_without_the_ledger_builds_on_any_host():
    assert build(frozenset(), producers=()).hooks  # rules only: nothing needs tool_blocked


def test_a_missing_failure_report_is_a_warning_and_not_an_error():
    from loguru import logger

    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        build(frozenset({"tool_blocked"}))
    finally:
        logger.remove(sink)
    assert any("does not report a hook that raises" in m for m in messages)


def test_the_cuga_adapter_reads_the_agents_registry_and_says_empty_without_hooks(monkeypatch):
    class Registry:
        def tool_blocked(self, tool, args, message):
            return None

    hooks = types.ModuleType("cuga.backend.hooks")
    hooks.HookRegistry = Registry
    monkeypatch.setitem(sys.modules, "cuga", types.ModuleType("cuga"))
    monkeypatch.setitem(sys.modules, "cuga.backend", types.ModuleType("cuga.backend"))
    monkeypatch.setitem(sys.modules, "cuga.backend.hooks", hooks)
    assert cuga_adapter.host_features() == frozenset({"tool_blocked"})

    monkeypatch.setitem(sys.modules, "cuga.backend.hooks", None)  # an import of it raises ImportError: no hooks
    assert cuga_adapter.host_features() == frozenset()
