"""``cli pack``: the step of the policy compiler skill that makes one package from a policy set and a draft. The worked
example in ``skill/SKILL.md`` is run here, so the skill text cannot drift from the validator."""

import json
import re
from pathlib import Path

import pytest

from polisys import cli
from polisys.dry_run import RecordingModels
from polisys.package import load_package
from polisys.policy_config import PolicyConfig
from polisys.procedure_store import ProcedureStore
from polisys.runtime import RunContext, build_runtime
from polisys.store_gate import StoreGate

SKILL = Path(cli.__file__).parent / "skill" / "SKILL.md"


def _example(name):
    """A fenced block of the skill whose info string is ``<language> <file name>``."""
    match = re.search(rf"^```\w+ {re.escape(name)}\n(.*?)^```$", SKILL.read_text(), re.S | re.M)
    assert match, f"{name} not found in SKILL.md"
    return match.group(1)


def _files(tmp_path):
    for name in ("policy.md", "tools.json", "draft.json", "tests.json"):
        (tmp_path / name).write_text(_example(name))
    assert cli.main(["convert", str(tmp_path / "policy.md"), "-o", str(tmp_path / "set.json")]) == 0


def _pack(tmp_path, *extra):
    return cli.main(
        [
            "pack",
            str(tmp_path / "set.json"),
            str(tmp_path / "draft.json"),
            "-o",
            str(tmp_path / "package.json"),
            "--harness",
            "test-harness",
            "--model",
            "test-model",
            *extra,
        ]
    )


def _validate(tmp_path):
    return cli.main(["validate", str(tmp_path / "package.json"), "--tools", str(tmp_path / "tools.json")])


def _draft(tmp_path):
    return json.loads((tmp_path / "draft.json").read_text())


def _save(tmp_path, draft):
    (tmp_path / "draft.json").write_text(json.dumps(draft))


def _errors(tmp_path, capsys):
    """Pack the draft, validate it, and give the findings as ``(code, severity)`` pairs."""
    assert _pack(tmp_path) == 0
    capsys.readouterr()
    _validate(tmp_path)
    out = capsys.readouterr().out
    return {
        (m.group(2), {"ERROR": "error", "WARN": "warning"}[m.group(1)])
        for m in re.finditer(r"^(ERROR|WARN) (\w+) ", out, re.M)
    }, out


def test_the_skill_example_packs_and_validates_without_legacy(tmp_path, capsys):
    _files(tmp_path)
    assert (
        _pack(tmp_path, "--tools", str(tmp_path / "tools.json"), "--tests", str(tmp_path / "tests.json")) == 0
    )
    assert _validate(tmp_path) == 0
    assert "0 errors, 0 warnings" in capsys.readouterr().out
    pkg = load_package(tmp_path / "package.json")
    assert pkg.generator == {"kind": "skill", "harness": "test-harness", "model": "test-model"}
    assert pkg.version and pkg.policy_hash and len(pkg.documents) == 1
    assert {r.cls for r in pkg.rules} == {"hard", "judgment", "guidance"}
    assert [r.type for r in pkg.rules][:2] == ["required_args", "allowed_values"]  # filled from the rule
    assert pkg.rules[0].checker == "code" and pkg.rules[2].checker == "verifier"
    assert pkg.rules[2].why  # a verify rule says why
    assert pkg.procedures[0].must_not and pkg.obligations[0].values["status"]["from"] == "literal"
    assert pkg.policy_questions and len(pkg.tests) == 2
    modality = {st.id: st.modality for st in pkg.documents[0].statements}
    assert modality["policy:1:3"].value == "must_not" and modality["policy:2:6"] is None
    # a step left with only its id gets the statement's words; one that joins a split sentence keeps the join
    texts = [step.text for step in pkg.procedures[0].steps]
    assert texts[0] == "Read the asset record before changing it."
    assert texts[3].endswith("Subject: MOVE-[asset id].")


def test_the_skill_example_runs_in_the_procedure_store_and_in_build_runtime(tmp_path):
    """The package a skill user makes is read by the runtime: the store, and the gate profile's wiring (procedure tools and
    the must-not gate, models recorded and never called)."""
    _files(tmp_path)
    assert _pack(tmp_path) == 0
    pkg = load_package(tmp_path / "package.json")
    store = ProcedureStore.from_package(pkg)
    listing = store.list_procedures()
    assert "[§3] An audit hold is a flag" in listing and "- audit hold: a flag" in listing
    assert "MUST NOT: Never send a door code to a visitor." in listing
    text = store.get_procedure(["2"])
    assert "MUST NOT\n- Do not move an asset that is under audit hold." in text
    assert (
        "Dear [Manager], the asset [asset id] is moving to [new building]." in text
    )  # template 1, from `global`
    assert (
        "Output formats\n- subject of the move notice | MOVE-[asset id]" in text
    )  # a part reaches the reader
    assert store.add_days("12/24/2026", 1) == "12/28/2026 (Monday)"  # 12/25 is a holiday of the calendar
    cfg = PolicyConfig.from_dict(
        {  # the wiring of the gate profile: procedure tools plus the must-not gate over the opened procedures
            "package": "p.json",
            "sources": {"data": False, "discovery": False, "request": False},
            "authority": {"supersedes": False},
            "checkers": {
                "rules": "off",
                "gate": {"mode": "on", "model": "m/g", "scope": "active_procedures", "min_strength": "none"},
                "verifier": {"mode": "off"},
            },
            "organizers": {
                "procedure_tools": "on",
                "guide": "off",
                "loop": "off",
                "sop_tracker": "off",
                "navigator": "off",
                "requirements_tracker": "off",
            },
            "obligations": {"producers": []},
            "decisions": {"mode": "off"},
        }
    )
    ctx = RunContext(
        task_id="t",
        request="Move asset A-1.",
        system_prompt="policy",
        tool_specs=json.loads((tmp_path / "tools.json").read_text()),
        world_time="2026-11-05T00:00:00",
        policy_version="v",
    )
    parts = build_runtime(cfg, pkg, ctx, models=RecordingModels())
    assert [type(h) for h in parts.hooks] == [StoreGate]
    assert [t.name for t in parts.tools][:2] == ["policy__list_procedures", "policy__get_procedure"]
    gate = parts.hooks[0]
    parts.components["procedure_tools"].get_procedure(["2"])
    assert "[2] MUST NOT: Do not move an asset that is under audit hold." in gate._rules()
    assert "[escalation] MUST NOT: Never send a door code to a visitor." in gate._rules()
    # a decision shows its action as well as its condition
    assert (
        "[2] DECISION audit_hold: an asset that is under audit hold; then: Do not move an asset; else: Update the asset "
        "record with the new building" in gate._rules()
    )


@pytest.mark.parametrize(
    "edit, code, severity",
    [
        (
            lambda g: g.__setitem__("calendar", {"workdays": ["Mon"], "holidays": []}),
            "invalid_field",
            "error",
        ),
        (lambda g: g.__setitem__("calendar", {"holidays": ["25 December"]}), "invalid_field", "error"),
        (lambda g: g.__setitem__("routing", {"Assets": "front desk"}), "invalid_field", "error"),
        (lambda g: g.__setitem__("priority", {"levels": "P1"}), "invalid_field", "error"),
        (lambda g: g.__setitem__("housekeeping", ["one step"]), "invalid_field", "error"),
        (
            lambda g: g.__setitem__("templates", {"1": {"name": "n", "text": "t", "number": "2"}}),
            "invalid_field",
            "error",
        ),
        (lambda g: g.__setitem__("templates", {"1": {"number": "1", "name": "n"}}), "invalid_field", "error"),
        (lambda g: g.__setitem__("definitions", [{"meaning": "m"}]), "invalid_field", "error"),
        (lambda g: g.__setitem__("global_rules", [{"id": "G", "rule": "r"}]), "invalid_field", "error"),
        (lambda g: g.__setitem__("hours", {"open": "8"}), "unknown_field", "warning"),
    ],
)
def test_a_global_value_the_runtime_cannot_read_is_reported(tmp_path, capsys, edit, code, severity):
    _files(tmp_path)
    draft = _draft(tmp_path)
    edit(draft["global"])
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert (code, severity) in found, out


def test_invented_content_is_reported_as_not_verbatim(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    p = draft["procedures"][0]
    p["must_not"].append("Never lend the master key to a visitor.")  # invented
    p["steps"][1]["text"] = "Change the building on the asset."  # not the statement
    p["templates"].append({"step": 1, "template": "x", "quote": "Always attach the move form to the notice."})
    draft["obligations"][0]["source"]["quote"] = "Set the asset status to moved once the audit is done."
    draft["global"]["escalation"]["must_not"] = ["Do not email the dean."]
    _save(tmp_path, draft)
    assert _pack(tmp_path) == 1  # the step text no longer contains its statement: pack stops
    assert "the text does not contain the statement policy:2:3" in capsys.readouterr().out
    del p["steps"][1]["text"]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("quote_not_verbatim", "error") in found
    for where in (
        "procedure 2: must_not",
        "procedure 2: templates[1].quote",
        "obligation ob_2_status",
        "global.escalation",
    ):
        assert where in out, out


def test_a_must_not_may_be_a_condition_and_a_prohibition_each_verbatim(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["procedures"][0]["must_not"] = [
        "The asset is under audit hold. Do not move an asset that is under audit hold.",  # not both verbatim
        "When an asset moves to another building: Do not move an asset that is under audit hold.",  # both
        "Do not move an asset that is under audit hold.",  # whole
    ]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert out.count("must_not is not the words") == 1 and "The asset is under audit hold." in out


def test_a_role_the_policy_never_names_is_an_error(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["rules"].append(
        {
            "id": "r_roles",
            "class": "hard",
            "level": "remind",
            "statement_ids": ["policy:1:1"],
            "rule": {"type": "role_permission", "tools": ["close_ticket"], "roles": ["auditor"]},
        }
    )
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("unknown_role", "error") in found and "'auditor'" in out


def test_value_binding_is_checked(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    ob = draft["obligations"][1]
    ob["effect"]["match"]["channel"] = {
        "pattern": "{ghost}"
    }  # an argument of no tool, and a value that is not there
    ob["values"]["extra"] = {"from": "case_fact", "field": "f", "record": "r"}  # in no pattern
    del draft["obligations"][0]["values"]["status"]["value"]  # a literal with no value
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert {"unknown_arg", "unknown_value_name", "unfilled_slot"} <= {code for code, _ in found}
    assert ("unbound_value", "warning") in found and "unmatched value extra" in out


def test_only_the_literal_a_question_names_as_open_is_refused(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    question = {
        "statement_ids": ["policy:2:4"],
        "question": "Is in_transit the right name?",
        "default_level": "remind",
    }
    draft["policy_questions"].append(question)  # about the statement, but it names no value
    _save(tmp_path, draft)
    found, _ = _errors(tmp_path, capsys)
    assert not any(code == "literal_open_question" for code, _ in found)
    question["open_values"] = ["ob_2_status.status"]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("literal_open_question", "error") in found and "value status is a literal" in out
    question["open_values"] = ["ob_2_status.nothing"]
    _save(tmp_path, draft)
    found, _ = _errors(tmp_path, capsys)
    assert ("unknown_open_value", "error") in found


def test_a_verify_rule_of_a_skill_package_needs_why(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    del draft["rules"][2]["why"]
    _save(tmp_path, draft)
    found, _ = _errors(tmp_path, capsys)
    assert found == {("verify_without_why", "warning")}


def test_a_step_action_must_be_a_tool(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["procedures"][0]["steps"][0]["action"] = "read_asset"
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("unknown_tool", "error") in found and "steps[0].action" in out


def test_pack_lists_statements_with_no_modality(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    del draft["modality"]["policy:1:3"]  # a must-not the author forgot
    _save(tmp_path, draft)
    assert _pack(tmp_path) == 0
    out = capsys.readouterr().out
    assert "8 statements have no modality" in out and "policy:1:3" in out  # must-style words come first
    listing = (tmp_path / "package.unmarked.txt").read_text().splitlines()
    assert len(listing) == 8 and listing[0].startswith("policy:1:3\tLOUD")


def test_a_must_statement_with_no_outcome_fails_validate(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["coverage"] = []  # the badge statement now has no outcome
    _save(tmp_path, draft)
    assert _pack(tmp_path) == 0
    assert _validate(tmp_path) == 1
    assert "ERROR coverage_incomplete policy:3:1" in capsys.readouterr().out


def test_pack_refuses_ghost_ids_a_bad_test_and_a_typo_and_writes_nothing(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["rules"][0]["statement_ids"] = ["policy:9:9"]
    draft["modality"]["policy:8:8"] = "must"
    draft["modality"]["policy:1:4"] = "ought"
    draft["modality"]["policy:1:2"] = ["must"]  # a list is an error, not a crash
    draft["procedures"][0]["steps"].append({"id": "policy:7:7"})  # a ghost step
    draft["procedures"][0]["links"] = [{"type": "next", "to": "9.9"}]  # a ghost target
    draft["obligations"][0]["scope"]["procedure"] = "7"
    draft["policy_questions"][0]["rule_ids"] = ["r_nothing"]
    draft["global"]["definitions"][0]["source"]["statement"] = "policy:6:6"
    draft["tests"] = [
        {"tool": "no_such_tool", "expected": "allow"},
        {"tool": "close_ticket"},
        {"tool": "close_ticket", "expected": "deny", "statement_id": "policy:7:7"},
    ]
    draft["rulez"] = []
    _save(tmp_path, draft)
    assert _pack(tmp_path, "--tools", str(tmp_path / "tools.json")) == 1
    out = capsys.readouterr().out
    for text in (
        "rules[0]: 'policy:9:9'",
        "'policy:8:8' is not a statement",
        "'ought' is not one of",
        "modality of policy:1:2",
        "steps[5]: 'policy:7:7'",
        "links[0]: '9.9' is not a procedure",
        "scope.procedure '7'",
        "rule_ids 'r_nothing'",
        "definitions[0]: 'policy:6:6'",
        "unknown tool no_such_tool",
        "tests[1]",
        "tests[2]: 'policy:7:7'",
        "unknown draft keys ['rulez']",
    ):
        assert text in out, text
    assert not (tmp_path / "package.json").exists()


def test_a_value_the_tool_does_not_accept_is_a_warning_with_schema_conflict(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    rule = draft["rules"][1]
    rule["rule"]["values"] = ["low", "normal", "urgent", "critical"]
    _save(tmp_path, draft)
    assert _pack(tmp_path) == 0
    assert _validate(tmp_path) == 1  # not declared: an error
    rule["rule"]["schema_conflict"] = True
    rule["level"] = "observe"
    _save(tmp_path, draft)
    assert _pack(tmp_path) == 0
    capsys.readouterr()
    assert _validate(tmp_path) == 0
    assert "WARN schema_conflict" in capsys.readouterr().out


def test_a_minimal_hand_written_package_validates(tmp_path, capsys):
    """The shape of the brief: a rule, a procedure with a must-not, an obligation with a literal value, a question, a test."""
    package = {
        "schema": "policy-package/1",
        "version": "v1",
        "generator": {"kind": "skill", "harness": "h", "model": "m"},
        "documents": [
            {
                "id": "d",
                "kind": "md",
                "sections": [
                    {
                        "id": "1",
                        "statements": [
                            {
                                "id": "d:1:1",
                                "text": "A desk must always be locked at night.",
                                "modality": "must",
                            },
                            {
                                "id": "d:1:2",
                                "text": "Never lend the master key to a visitor.",
                                "modality": "must_not",
                            },
                            {
                                "id": "d:1:3",
                                "text": "Set the lock state to locked after closing.",
                                "modality": "must",
                            },
                            {"id": "d:1:4", "text": "Report any damage to the manager.", "modality": "must"},
                        ],
                    }
                ],
            }
        ],
        "rules": [
            {
                "id": "r1",
                "type": "required_args",
                "class": "hard",
                "level": "deny",
                "statement_ids": ["d:1:1"],
                "rule": {
                    "id": "r1",
                    "type": "required_args",
                    "tool": "lock_desk",
                    "args": ["desk"],
                    "quote": "A desk must always be locked at night.",
                },
            }
        ],
        "procedures": [
            {
                "id": "1",
                "steps": [{"id": "d:1:3", "text": "Set the lock state to locked after closing."}],
                "must_not": ["Never lend the master key to a visitor."],
                "exceptions": [
                    {"quote": "Never lend the master key to a visitor.", "when": "a visitor asks"}
                ],
            }
        ],
        "obligations": [
            {
                "id": "ob1",
                "origin": "package",
                "scope": {"procedure": "1"},
                "effect": {"kind": "write", "tools": ["lock_desk"], "match": {"desk": {"pattern": "{desk}"}}},
                "values": {"desk": {"from": "case_fact", "field": "desk", "record": "the desk record"}},
                "check": "code",
                "level": "observe",
                "source": {"section": "1", "quote": "Set the lock state to locked after closing."},
            }
        ],
        "policy_questions": [{"statement_ids": ["d:1:4"], "question": "Who is the manager at night?"}],
        "tests": [{"tool": "lock_desk", "args": {}, "expected": "deny", "statement_id": "d:1:1"}],
    }
    tools = [
        {"name": "lock_desk", "inputSchema": {"type": "object", "properties": {"desk": {"type": "string"}}}}
    ]
    (tmp_path / "package.json").write_text(json.dumps(package))
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    assert _validate(tmp_path) == 0
    assert "0 errors, 0 warnings" in capsys.readouterr().out


def test_a_graph_without_parts_gives_the_same_procedure_text_and_a_package_with_parts_shows_them():
    graph = {
        "id": "g",
        "procedures": [{"id": "1", "name": "N", "steps": ["Do it."], "must_not": ["Do not."], "links": []}],
    }
    plain = ProcedureStore(graph).procedure_text(graph["procedures"][0])
    assert plain == "# 1 N []\nApplies when: \n## Steps\n- Do it.\n## MUST NOT\n- Do not."
    with_parts = {**graph["procedures"][0], "exceptions": [{"when": "w", "then": "t", "quote": "q q q q"}]}
    assert "## Exceptions\n- w | t" in ProcedureStore(graph).procedure_text(with_parts)


def test_the_sop_index_reads_the_text_of_a_step_that_has_no_quote():
    from polisys.sop_retrieval import proc_text

    assert "Read the record." in proc_text({"name": "n", "steps": [{"id": "s", "text": "Read the record."}]})
    assert "From a build." in proc_text({"name": "n", "steps": [{"quote": "From a build.", "text": "other"}]})


def test_one_skill_package_serves_the_gate_and_the_guide(tmp_path):
    """Steps with ``action`` give the guide its tools; the SOP index reads the step text (a step has no ``quote``)."""
    from polisys.package import to_build
    from polisys.procedure_guide import ProcedureGuide
    from polisys.sop_retrieval import SopIndex

    _files(tmp_path)
    assert _pack(tmp_path) == 0
    pkg = load_package(tmp_path / "package.json")
    guide = ProcedureGuide.from_package(pkg, model=None, gate=True, templates=False)
    assert [s.get("action") for s in guide.procs[0]["steps"]][:4] == [
        "get_asset",
        "update_asset",
        "update_asset",
        "send_message",
    ]
    index = SopIndex(to_build(pkg))
    from polisys.sop_retrieval import proc_text

    assert "Set the asset status to in_transit" in proc_text(index.procs[0])


CODE_TOOLS = [
    {"name": "executePython", "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}}},
    {"name": "executeBash", "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {
        "name": "post_note",
        "inputSchema": {
            "type": "object",
            "properties": {"payload": {"type": "string"}, "to": {"type": "string"}},
        },
    },
]


def test_an_obligation_on_both_code_tools_needs_a_pattern_for_each_code_argument(tmp_path, capsys):
    _files(tmp_path)
    (tmp_path / "tools.json").write_text(
        json.dumps(json.loads((tmp_path / "tools.json").read_text()) + CODE_TOOLS)
    )
    draft = _draft(tmp_path)
    ob = draft["obligations"][0]
    ob["effect"] = {
        "kind": "write",
        "tools": ["executePython", "executeBash"],
        "match": {"code": {"pattern": "{what}"}},
    }
    ob["values"] = {"what": {"from": "literal", "value": "kiln_log.csv"}}
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("unknown_arg", "error") in found and "executeBash takes its code in `command`" in out
    ob["effect"]["match"]["command"] = {"pattern": "{what}"}
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert not any(code == "unknown_arg" for code, _ in found), out
    # another argument must be in every tool, not in one of them
    ob["effect"]["tools"] = ["post_note", "update_asset"]
    ob["effect"]["match"] = {"payload": {"pattern": "{what}"}}
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("unknown_arg", "error") in found and "not an argument of update_asset" in out


def test_a_cross_cutting_obligation_has_an_empty_scope(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["obligations"][0]["scope"] = {}
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == set(), out


def test_text_that_is_not_in_the_policy_is_a_warning(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    p, g = draft["procedures"][0], draft["global"]
    p["applies_when"] = "Whenever the kiln is cold."
    p["notes"] = ["Rush orders are first."]
    p["decisions"][0]["rule"] = "the engineer is on leave"
    p["decisions"][0]["then"] = "Continue with Step 2"  # flow text is not checked
    g["global_rules"].append({"id": "G-new", "src": "§9", "rule": "Doors lock at ten at night."})
    g["escalation"]["steps"] = ["Call the director."]
    g["templates"]["1"]["text"] = "Dear [Manager], we are moving the asset [asset id]."
    g["definitions"][0]["meaning"] = "a mark kept by the front desk"
    ob = draft["obligations"][1]
    ob["values"]["subject"]["pattern"] = "SHIPPED-{asset_id}"
    ob["effect"]["tools"] = ["send_message"]
    ob["values"]["greeting"] = {"from": "literal", "value": "Hello there friend"}
    ob["effect"]["match"]["body"] = {"pattern": "{greeting}"}
    draft["obligations"][0]["values"]["status"]["value"] = "parked"  # enum argument: the tool's own value
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert {c for c, _ in found} == {"not_in_text"}
    for where in (
        "procedure 2: applies_when",
        "procedure 2: notes[0]",
        "decisions[0].rule",
        "global.global_rules[1]",
        "global.escalation.steps[0]",
        "global.templates.1",
        "global.definitions[0].meaning",
        "obligation ob_2_notice: value subject",
        "obligation ob_2_notice: value greeting",
    ):
        assert where in out, where
    assert "ob_2_status" not in out  # a literal on an enum argument is not looked up


def test_a_must_not_that_cannot_be_read_alone_is_a_warning(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["procedures"][0]["must_not"] = [
        "Do not move an asset that is under audit hold.",
        "Do not move",
    ]
    draft["global"]["escalation"]["must_not"] = ["Never send a door code to a visitor."]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == {("bare_must_not", "warning")} and out.count("bare_must_not") == 1
    draft["procedures"][0]["must_not"] = ["That asset is under audit hold. They must not move it."]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("quote_not_verbatim", "error") in found


def test_a_dash_or_a_curly_quote_does_not_make_a_quote_differ(tmp_path, capsys):
    from polisys.validate import _fold

    assert _fold("Mon \u2013 Fri, \u201cdone\u201d") == _fold('mon - fri, "done"')
    _files(tmp_path)
    (tmp_path / "policy.md").write_text(
        (tmp_path / "policy.md")
        .read_text()
        .replace(
            "The desk is closed on 2026-12-25.", "The desk is closed \u2014 all day \u2014 on 2026-12-25."
        )
    )
    assert cli.main(["convert", str(tmp_path / "policy.md"), "-o", str(tmp_path / "set.json")]) == 0
    draft = _draft(tmp_path)
    draft["global"]["global_rules"].append(
        {"id": "G-closed", "src": "§5", "rule": "The desk is closed - all day - on 2026-12-25."}
    )
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == set(), out


def test_pack_gives_an_error_and_not_a_traceback_for_an_id_that_is_not_text(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["rules"][0]["id"] = ["r"]
    draft["rules"][0]["statement_ids"] = [["policy:1:1"]]
    draft["procedures"][0]["steps"][0]["id"] = ["a"]
    draft["procedures"][0]["links"] = [{"type": "next", "to": ["z"]}]
    draft["obligations"][0]["scope"] = {"procedure": ["q"]}
    draft["policy_questions"][0]["rule_ids"] = [["r"]]
    draft["global"]["definitions"][0]["source"] = {"statement": ["k"]}
    draft["tests"] = [
        {"tool": ["t"], "expected": "allow"},
        {"tool": "close_ticket", "expected": "deny", "statement_id": ["x"]},
    ]
    _save(tmp_path, draft)
    assert _pack(tmp_path, "--tools", str(tmp_path / "tools.json")) == 1
    out = capsys.readouterr().out
    assert out.count("ERROR pack") >= 7 and "Traceback" not in out


def test_a_priority_is_shown_to_the_agent_and_a_graph_without_one_is_unchanged():
    graph = {"id": "g", "procedures": [{"id": "1", "name": "N", "steps": ["Do it."], "links": []}]}
    plain = ProcedureStore(graph).procedure_text(graph["procedures"][0])
    assert "Priority" not in plain
    graph["procedures"][0]["priority"] = "P1 when the deadline is today"
    shown = ProcedureStore(graph).procedure_text(graph["procedures"][0])
    assert shown.splitlines()[2] == "Priority: P1 when the deadline is today"


def _text(*statements):
    from polisys.model import Document, PolicyPackage, Section, Statement
    from polisys.validate import _Text

    doc = Document(
        id="d",
        kind="md",
        sections=[
            Section(
                id="1", statements=[Statement(id=f"d:1:{i}", text=t) for i, t in enumerate(statements, 1)]
            )
        ],
    )
    return _Text(PolicyPackage(documents=[doc]))


def test_a_must_not_is_found_when_the_longest_first_run_leaves_too_little():
    # "... never hand" and "never hand it over" are in the text in two places, so the split is
    # "if a visitor asks for a spare key" + "never hand it over", not the longest first run
    text = _text("If a visitor asks for a spare key never hand", "Staff know this. Never hand it over.")
    assert text.must_not_ok("if a visitor asks for a spare key never hand it over")
    assert not text.must_not_ok("if a visitor asks for a spare key always hand it over")
    assert not _text("only one place").must_not_ok("only one place and a made up rest of it")


def test_an_obligation_id_with_dots_can_be_named_in_open_values(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    draft["obligations"][0]["id"] = "ob.2.status"
    draft["policy_questions"][0]["open_values"] = ["ob.2.status.status"]
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert ("literal_open_question", "error") in found and ("unknown_open_value", "error") not in found, out


def test_an_optional_boolean_or_enum_argument_is_not_looked_up(tmp_path, capsys):
    from polisys.validate import _tool_typed

    assert _tool_typed({"anyOf": [{"type": "boolean"}, {"type": "null"}]})
    assert _tool_typed({"type": ["boolean", "null"]})
    assert _tool_typed({"oneOf": [{"enum": ["a"]}, {"type": "null"}]})
    assert not _tool_typed({"anyOf": [{"type": "string"}, {"type": "null"}]})
    _files(tmp_path)
    tools = json.loads((tmp_path / "tools.json").read_text())
    for tool in tools:
        if tool["name"] == "update_asset":
            tool["inputSchema"]["properties"]["is_read"] = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
            tool["inputSchema"]["properties"]["note"] = {"type": "string"}
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    draft = _draft(tmp_path)
    ob = draft["obligations"][0]
    ob["effect"]["match"] = {"is_read": {"pattern": "{flag}"}}
    ob["values"] = {"flag": {"from": "literal", "value": "true"}}
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == set(), out
    ob["effect"]["match"] = {"note": {"pattern": "{flag}"}}  # a plain text argument is looked up
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == {("not_in_text", "warning")} and "value flag" in out


def test_only_the_rule_of_a_decision_is_looked_up(tmp_path, capsys):
    _files(tmp_path)
    draft = _draft(tmp_path)
    decision = draft["procedures"][0]["decisions"][0]
    decision["then"], decision["else"] = "Continue with Step 2", "Go back to Step 1 and wait"
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == set(), out
    decision["rule"] = "the kiln is cold"
    _save(tmp_path, draft)
    found, out = _errors(tmp_path, capsys)
    assert found == {("not_in_text", "warning")} and "decisions[0].rule" in out
