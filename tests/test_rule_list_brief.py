"""The brief of a rule list (Task 5.3): each rule with its exceptions, the guidance that names no tool, and no SOP tracker.

Offline: no model call. A measured build keeps the brief it had: both settings are off unless the run asks for them.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pytest

from polisys.levels import components_for, resolve
from polisys.model import PolicyPackage
from polisys.package import exception_links, from_build
from polisys.policy_config import PolicyConfig, PolicyConfigError
from polisys.runtime import RunContext, RuntimeWiringError, build_runtime
from polisys.ruleset import RuleSetHooks
from polisys.validate import validate_package

TOOL = "change_booking"
SPECS = [
    {
        "name": TOOL,
        "description": "Change a booking.",
        "inputSchema": {"type": "object", "properties": {"booking_id": {"type": "string"}}},
    },
    {
        "name": "cancel_booking",
        "description": "Cancel a booking.",
        "inputSchema": {"type": "object", "properties": {"booking_id": {"type": "string"}}},
    },
]
HEAD = "## Policy rules checked before each write (from the policy text)"
BASE = {
    "id": "r_base",
    "type": "verify",
    "tools": [TOOL],
    "statement": "A basic economy booking cannot change flights.",
    "source": "§5",
    "statement_ids": ["s:5"],
}
EXCEPTION = {
    "id": "r_cabin",
    "type": "verify",
    "tools": [TOOL],
    "statement": "All bookings, also basic economy, can change the cabin.",
    "source": "§11",
    "statement_ids": ["s:11"],
}
OTHER = {
    "id": "r_other",
    "type": "verify",
    "tools": ["cancel_booking"],
    "statement": "A cancelled booking is not refunded twice.",
    "source": "§7",
    "statement_ids": ["s:7"],
}
GUIDE = {
    "id": "g_no_advice",
    "type": "guidance",
    "text": "Give no advice that no tool or the user gave.",
    "source": "§1",
    "statement_ids": ["s:1"],
}
GUIDE_TOOL = {
    **GUIDE,
    "id": "g_tool",
    "text": "Be brief when you change a booking.",
    "tools": [TOOL],
    "source": "§2",
    "statement_ids": ["s:2"],
}
LINK = {"type": "exception_of", "src": "s:11", "dst": "s:5", "source": {}, "quote": ""}


def package(*rules: Dict[str, Any], links: List[Dict[str, Any]] = ()) -> PolicyPackage:
    return from_build({"rules": [dict(r) for r in rules], "procedures": [], "links": list(links)})


def brief_of(hooks: RuleSetHooks) -> str:
    return hooks.before_llm_call(None).message or ""


def hooks_of(pkg: PolicyPackage, **kwargs: Any) -> RuleSetHooks:
    return RuleSetHooks.from_package(pkg, SPECS, **kwargs)


# -- the links reach the hooks ----------------------------------------------------------------------------------------


def test_a_link_joins_the_rules_that_hold_its_statements():
    pkg = package(BASE, EXCEPTION, OTHER, links=[LINK])
    assert exception_links(pkg) == {"r_base": ["r_cabin"]}
    both = package(
        {**BASE, "statement_ids": ["s:5", "s:6"]},
        {**OTHER, "statement_ids": ["s:5"]},
        EXCEPTION,
        links=[LINK],
    )
    assert exception_links(both) == {"r_base": ["r_cabin"], "r_other": ["r_cabin"]}
    # a rule that holds both ends has the exception folded in: it is neither end of the link
    folded = package(
        {**BASE, "statement_ids": ["s:5", "s:11"]}, OTHER, {**EXCEPTION, "id": "r_exc2"}, links=[LINK]
    )
    assert exception_links(folded) == {}
    one_folded = package(
        {**BASE, "statement_ids": ["s:5", "s:11"]}, {**BASE, "id": "r_plain"}, EXCEPTION, links=[LINK]
    )
    assert exception_links(one_folded) == {"r_plain": ["r_cabin"]}
    other_type = package(BASE, EXCEPTION, links=[{**LINK, "type": "refers_to"}])
    assert exception_links(other_type) == {}
    assert (
        exception_links(package(BASE, links=[{**LINK, "src": "s:5"}])) == {}
    )  # a rule is not its own exception


def test_the_package_links_reach_the_hooks_only_when_asked():
    pkg = package(BASE, EXCEPTION, links=[LINK])
    assert hooks_of(pkg).exception_links == {}
    assert hooks_of(pkg, with_exceptions=True).exception_links == {"r_base": ["r_cabin"]}
    given = hooks_of(pkg, with_exceptions=True, exception_links={"r_cabin": ["r_base"]})
    assert given.exception_links == {"r_cabin": ["r_base"]}  # an argument wins


def test_an_exception_link_with_no_rule_at_one_end_is_a_validate_finding():
    findings = lambda pkg: [f for f in validate_package(pkg, None) if f.code == "exception_without_rule"]  # noqa: E731
    assert findings(package(BASE, EXCEPTION, links=[LINK])) == []
    missing = findings(package(BASE, links=[LINK]))
    assert len(missing) == 1 and missing[0].severity == "warning"
    assert (
        "s:11 (exception)" in missing[0].message and "s:5" not in missing[0].message.split("no rule holds")[1]
    )
    both = findings(package(OTHER, links=[LINK]))
    assert "s:11 (exception) and s:5 (rule)" in both[0].message
    assert findings(package(BASE, EXCEPTION, links=[{**LINK, "type": "refers_to", "src": "x"}])) == []
    assert findings(package(links=[LINK])) == []  # no rule at all: not a rule list
    # a statement that is a step of a procedure is held: the exception is in the procedure
    with_step = from_build(
        {
            "rules": [dict(BASE)],
            "procedures": [
                {"id": "p1", "steps": [{"id": "s:11", "action": "", "text": "Cabin can change."}]}
            ],
            "links": [LINK],
        }
    )
    assert findings(with_step) == []


# -- the brief --------------------------------------------------------------------------------------------------------


def test_a_rule_with_an_exception_shows_both_and_the_exception_is_not_listed_twice():
    brief = brief_of(hooks_of(package(BASE, EXCEPTION, OTHER, links=[LINK]), with_exceptions=True))
    lines = brief.splitlines()
    assert lines[0] == HEAD
    base = lines.index(f"- [§5] {BASE['statement']}")
    assert lines[base + 1] == f"  - exception [§11]: {EXCEPTION['statement']}"
    assert brief.count(EXCEPTION["statement"]) == 1
    assert lines[1] == f"- [§7] {OTHER['statement']}"  # by tool name: cancel_booking comes first
    assert len(lines) == 4


def test_the_brief_is_the_same_text_when_the_exceptions_are_not_asked_for():
    pkg = package(BASE, EXCEPTION, OTHER, GUIDE, links=[LINK])
    plain = brief_of(hooks_of(pkg))
    assert plain == "\n".join(
        [
            HEAD,
            f"- [§7] {OTHER['statement']}",
            f"- [§5] {BASE['statement']}",
            f"- [§11] {EXCEPTION['statement']}",
        ]
    )  # by tool name, then the order of the rules
    no_links = package(BASE, EXCEPTION, OTHER, GUIDE)
    assert brief_of(hooks_of(no_links, with_exceptions=True)) == plain  # asked for, no link: the same text
    assert hashlib.sha256(brief_of(hooks_of(pkg, brief_guidance=False)).encode()).hexdigest() == (
        hashlib.sha256(plain.encode()).hexdigest()
    )


def test_an_exception_of_a_rule_the_brief_does_not_show_stays_out_and_stays_where_it_was():
    # the target is for a tool that is not in the run: the exception keeps its own place
    hooks = hooks_of(
        package(BASE, EXCEPTION, links=[LINK]), with_exceptions=True, tools_in_run=["cancel_booking"]
    )
    assert brief_of(hooks) == ""
    pkg = package(BASE, {**EXCEPTION, "tools": ["cancel_booking"]}, links=[LINK])
    hooks = hooks_of(pkg, with_exceptions=True, tools_in_run=["cancel_booking"])
    assert brief_of(hooks).splitlines()[1] == f"- [§11] {EXCEPTION['statement']}"
    shown = hooks_of(pkg, with_exceptions=True)  # both tools in the run: the exception is under its rule
    assert brief_of(shown).splitlines()[2].startswith("  - exception [§11]")


def test_exceptions_of_exceptions_and_a_cycle_show_every_rule_once():
    third = {
        **OTHER,
        "id": "r_third",
        "tools": [TOOL],
        "statement": "No cabin change in the last day.",
        "source": "§12",
        "statement_ids": ["s:12"],
    }
    chain = package(BASE, EXCEPTION, third, links=[LINK, {**LINK, "src": "s:12", "dst": "s:11"}])
    lines = brief_of(hooks_of(chain, with_exceptions=True)).splitlines()
    assert lines[1:] == [
        f"- [§5] {BASE['statement']}",
        f"  - exception [§11]: {EXCEPTION['statement']}",
        f"    - exception [§12]: {third['statement']}",
    ]
    cycle = package(BASE, EXCEPTION, links=[LINK, {**LINK, "src": "s:5", "dst": "s:11"}])
    brief = brief_of(hooks_of(cycle, with_exceptions=True))
    assert brief.count(BASE["statement"]) == 1 and brief.count(EXCEPTION["statement"]) == 1


def test_an_exception_the_agent_cannot_see_is_not_shown_under_its_rule():
    hidden = {**EXCEPTION, "type": "grounded_values", "may_deny": False}
    pkg = package(BASE, hidden, links=[LINK])
    hooks = hooks_of(pkg, with_exceptions=True)
    assert hooks._observe_only(hooks.rules[1])
    assert brief_of(hooks).splitlines() == [HEAD, f"- [§5] {BASE['statement']}"]


def test_guidance_with_no_tool_shows_only_when_the_switch_is_on():
    pkg = package(BASE, GUIDE, GUIDE_TOOL)
    off = brief_of(hooks_of(pkg)).splitlines()
    assert f"- [§1] {GUIDE['text']}" not in off  # never shown today
    assert f"- [§2] {GUIDE_TOOL['text']}" in off  # guidance that names a tool is shown as before
    on = brief_of(hooks_of(pkg, brief_guidance=True)).splitlines()
    assert f"- [§1] {GUIDE['text']}" in on
    assert on.count(f"- [§1] {GUIDE['text']}") == 1
    assert [x for x in on if x not in off] == [f"- [§1] {GUIDE['text']}"]
    assert brief_of(hooks_of(package(GUIDE))) == ""  # a brief with nothing else says nothing
    assert brief_of(hooks_of(package(GUIDE), brief_guidance=True)).splitlines() == [
        HEAD,
        f"- [§1] {GUIDE['text']}",
    ]


def test_guidance_that_is_removed_leaves_the_guidance_list():
    hooks = hooks_of(package(GUIDE), brief_guidance=True)
    assert hooks._global_guidance
    hooks = RuleSetHooks([{**GUIDE, "quote": GUIDE["text"]}], SPECS, brief_guidance=True)
    assert hooks.remove_rules_for(GUIDE["text"]) == 1 and hooks._global_guidance == []


def test_the_brief_detail_counts_the_exceptions_it_wrote():
    hooks = hooks_of(package(BASE, EXCEPTION, links=[LINK]), with_exceptions=True)
    assert hooks.before_llm_call(None).details[0]["exceptions"] == 1
    plain = hooks_of(package(BASE, EXCEPTION, links=[LINK])).before_llm_call(None).details[0]
    assert "exceptions" not in plain


# -- through build_runtime --------------------------------------------------------------------------------------------

OFF = {
    "rules": "on",
    "gate": {"mode": "off"},
    "verifier": {"mode": "off"},
    "write_checks": "off",
}
ORGANIZERS = {
    k: "off" for k in ("procedure_tools", "guide", "loop", "sop_tracker", "navigator", "requirements_tracker")
}
CTX = RunContext(
    task_id="t", policy_version="v1", request="Change it.", tool_specs=SPECS, agent_model="m/agent"
)


def config(shapes=(), notes=None, organizers=None, checkers=None, **top) -> PolicyConfig:
    return PolicyConfig.from_dict(
        {
            "package": "p.json",
            "shapes": list(shapes),
            "checkers": {**OFF, **(checkers or {})},
            "organizers": {**ORGANIZERS, **(organizers or {})},
            "decisions": {"mode": "off"},
            "notes": notes or {},
            **top,
        }
    )


def runtime_brief(cfg: PolicyConfig, pkg: PolicyPackage) -> str:
    parts = build_runtime(cfg, pkg, CTX, models=None)
    hooks = [h for h in parts.hooks if isinstance(h, RuleSetHooks)]
    assert len(hooks) == 1
    return brief_of(hooks[0])


PKG = package(BASE, EXCEPTION, OTHER, GUIDE, links=[LINK])


def test_a_rule_list_shows_the_exceptions_and_another_shape_keeps_the_brief_as_it_was():
    # an auto component makes the declared shape read; the shape is what turns the exceptions on
    auto = {"verifier": {"mode": "off"}, "rules": "on"}
    rule_list = config(["rules"], organizers={"guide": "auto"}, checkers=auto)
    assert "  - exception [§11]" in runtime_brief(rule_list, PKG)
    other = config(["data"], organizers={"guide": "auto"}, checkers=auto, sources={"data": False})
    # a declared shape that is not a rule list: the brief is the old one
    assert "exception" not in runtime_brief(other, PKG)
    no_shape = config([], checkers=auto)  # inferred from the package: rules and no procedure is a rule list
    assert "  - exception [§11]" in runtime_brief(no_shape, PKG)


def test_brief_guidance_is_a_setting_that_the_rule_hooks_read():
    base = runtime_brief(config([]), PKG)
    assert GUIDE["text"] not in base
    on = runtime_brief(config([], notes={"brief_guidance": True}), PKG)
    assert GUIDE["text"] in on
    assert PolicyConfig().notes.brief_guidance is False
    with pytest.raises(RuntimeWiringError, match="notes.brief_guidance.*nothing reads it"):
        build_runtime(config([], notes={"brief": "off", "brief_guidance": True}), PKG, CTX, models=None)
    with pytest.raises(RuntimeWiringError, match="notes.brief_guidance.*nothing reads it"):
        build_runtime(
            config([], notes={"brief_guidance": True}, checkers={"rules": "off"}), PKG, CTX, models=None
        )
    with pytest.raises(PolicyConfigError):
        PolicyConfig.from_dict({"notes": {"brief_guidance": "yes"}})


# -- the SOP tracker --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_a_rule_list_has_no_sop_tracker_at_any_level(level):
    assert components_for(level, "rules")["sop_tracker"] is False
    assert components_for(level, ["rules"])["sop_tracker"] is False
    assert components_for(level, ["rules", "data"])["sop_tracker"] is False


def test_under_auto_a_rule_list_gets_no_sop_tracker_and_an_explicit_on_wins():
    tracker = {"model": "m/tracker"}
    for level in (2, 3):
        auto = config(
            ["rules"], organizers={"sop_tracker": {"mode": "auto", **tracker}}, control={"level": level}
        )
        out = resolve(auto, PKG, "m/agent", policy_text="# Policy\nText.")
        assert out.cfg.organizers.sop_tracker.mode == "off"
        assert "not on at level" in dict(out.notes)["auto:sop_tracker"]
        explicit = config(
            ["rules"], organizers={"sop_tracker": {"mode": "on", **tracker}}, control={"level": level}
        )
        assert (
            resolve(explicit, PKG, "m/agent", policy_text="# Policy\nText.").cfg.organizers.sop_tracker.mode
            == "on"
        )
    flow = config(["flow"], organizers={"sop_tracker": {"mode": "auto", **tracker}}, control={"level": 2})
    assert (
        resolve(flow, PKG, "m/agent", policy_text="# Policy\nText.").cfg.organizers.sop_tracker.mode == "on"
    )


# -- a rule shared by exceptions: the exception is written under every rule it is an exception of ---------------------------


def test_an_exception_of_two_rules_is_written_under_each():
    second = {**OTHER, "tools": [TOOL], "statement_ids": ["s:7"]}
    pkg = package(BASE, second, EXCEPTION, links=[LINK, {**LINK, "dst": "s:7"}])
    lines = brief_of(hooks_of(pkg, with_exceptions=True)).splitlines()
    assert lines[1:] == [
        f"- [§5] {BASE['statement']}",
        f"  - exception [§11]: {EXCEPTION['statement']}",
        f"- [§7] {OTHER['statement']}",
        f"  - exception [§11]: {EXCEPTION['statement']}",
    ]
    hooks = hooks_of(pkg, with_exceptions=True)
    assert hooks.before_llm_call(None).details[0]["exceptions"] == 2


def test_a_chain_and_a_cycle_still_end_and_show_every_rule():
    third = {
        **OTHER,
        "id": "r_third",
        "tools": [TOOL],
        "statement": "No cabin change in the last day.",
        "source": "§12",
        "statement_ids": ["s:12"],
    }
    # B is an exception of A and of C, and C of B: A -> B -> C -> B is a cycle below A
    pkg = package(
        BASE,
        EXCEPTION,
        third,
        links=[LINK, {**LINK, "src": "s:11", "dst": "s:12"}, {**LINK, "src": "s:12", "dst": "s:11"}],
    )
    brief = brief_of(hooks_of(pkg, with_exceptions=True))
    assert brief.splitlines()[1] == f"- [§5] {BASE['statement']}"
    assert brief.count("exception [§11]") >= 1 and brief.count("exception [§12]") >= 1
    assert len(brief.splitlines()) < 12  # the cycle ends


# -- the pending-check parts of a declared rule list --------------------------------------------------------------------

PROCEDURE_BUILD = {
    "rules": [dict(BASE), dict(EXCEPTION)],
    "procedures": [
        {
            "id": "p1",
            "name": "Change",
            "steps": [{"id": "s:3", "action": TOOL, "quote": "Change the booking."}],
        }
    ],
    "links": [{"type": "refers_to", "src": "s:3", "dst": "s:5", "source": {}, "quote": ""}, LINK],
}


def hook_names(cfg: PolicyConfig, pkg: PolicyPackage) -> List[str]:
    cfg.organizers.guide.mode = "auto"  # something left to the level, so that a declared shape is read
    return [type(h).__name__ for h in build_runtime(cfg, pkg, CTX, models=None).hooks]


def test_a_declared_rule_list_builds_no_procedure_monitor_and_the_inferred_shapes_keep_it():
    pkg = from_build(PROCEDURE_BUILD)
    assert hook_names(config(["rules"]), pkg) == ["RuleSetHooks"]
    assert hook_names(config([]), pkg) == [
        "RuleSetHooks",
        "ProcedureMonitor",
    ]  # inferred: procedures and flow
    assert hook_names(config(["procedures"]), pkg) == ["RuleSetHooks", "ProcedureMonitor"]
    assert hook_names(config(["rules", "procedures"]), pkg) == ["RuleSetHooks", "ProcedureMonitor"]
    assert hook_names(config(["rules", "flow"]), pkg) == ["RuleSetHooks", "ProcedureMonitor"]


# -- the SOP tracker needs a declared flow ----------------------------------------------------------------------------------


def test_the_tracker_under_auto_needs_a_declared_flow_and_an_inferred_one_does_not_count():
    pkg = from_build(PROCEDURE_BUILD)
    tracker = {"mode": "auto", "model": "m/tracker"}
    for level in (1, 2, 3):
        inferred = config([], organizers={"sop_tracker": tracker}, control={"level": level})
        out = resolve(inferred, pkg, "m/agent", policy_text="# Policy\nText.")
        assert out.shapes == ["procedures", "flow"]  # the structure says flow
        assert out.cfg.organizers.sop_tracker.mode == "off"
    for level, on in ((1, False), (2, True), (3, True)):
        declared = config(["flow"], organizers={"sop_tracker": tracker}, control={"level": level})
        out = resolve(declared, pkg, "m/agent", policy_text="# Policy\nText.")
        assert (out.cfg.organizers.sop_tracker.mode == "on") is on
    # the reason is in the audit
    out = resolve(
        config([], organizers={"sop_tracker": tracker}, control={"level": 2}), pkg, "m/agent", policy_text="x"
    )
    assert "a step flow is declared in policy.shapes, not inferred" in dict(out.notes)["auto:sop_tracker"]
    # an explicit on still wins
    explicit = config(
        [], organizers={"sop_tracker": {"mode": "on", "model": "m/tracker"}}, control={"level": 1}
    )
    assert resolve(explicit, pkg, "m/agent", policy_text="x").cfg.organizers.sop_tracker.mode == "on"
