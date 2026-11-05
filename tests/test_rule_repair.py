"""The repairs of ``rule_repair`` run inside ``build_policy_set`` (scripted model, no paid call)."""

import json

from polisys import rule_repair
from polisys.builder import build_policy_set
from polisys.documents import load_document
from polisys.model import PolicySet, to_json
from polisys.package import from_build, to_build
from polisys.rule_repair import normalize_when, partial_update_overreach
from polisys.ruleset import RuleSetHooks
from polisys.validate import validate_package


def _tool(name, *args):
    return {
        "name": name,
        "inputSchema": {"type": "object", "properties": {a: {"type": "string"} for a in args}},
    }


SPECS = [
    _tool("create_order", "customer_id", "coupon", "tier", "owner_id", "assigned_to", "status"),
    _tool("update_order", "order_id", "coupon", "tier", "owner_id", "assigned_to", "status"),
    _tool("delete_order", "order_id"),
    _tool("create_invoice", "order_id", "amount"),
    _tool("get_items", "order_id"),
]
POLICY = """## 1. Orders

- An order with a coupon must have the tier promo.
- Every order must have an owner_id.
- The assigned_to person must be active.
- The weather must be kind to the staff.
- When an order is closed, its items must be listed.

## 2. Roles

| Role | Access |
|---|---|
| Admin (admin) | Full CRUD on all tables. |
| Clerk (clerk) | Full CRUD on orders. READ access to all other tables. |
"""


class Scripted:
    """Answers each prompt of the pipeline by its system text. Every prose statement is a ``rule`` with no valid rule,
    so it starts as a ``verify`` rule, except the duty (a bare value in ``then.when``)."""

    model_name = "scripted"

    def __init__(self):
        self.calls = []
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def __call__(self, system, user):
        self.usage["calls"] += 1
        self.calls.append(system.split(".")[0])
        if "classify" in system:
            n = sum(1 for x in user.splitlines() if x[:1].isdigit() and ". " in x[:4])
            return json.dumps(
                {
                    "statements": [
                        {"n": i, "types": ["open_judgment"], "modality": "must"} for i in range(1, n + 1)
                    ]
                }
            )
        if "typed rules" in system:  # prose
            answers = []
            for i, line in enumerate((x for x in user.splitlines() if x.startswith("S")), 1):
                if "items must be listed" in line:
                    rule = {
                        "type": "duty",
                        "quote": "When an order is closed, its items must be listed",
                        "trigger": {"tool": "update_order", "when": {"status": "closed"}},
                        "then": {"tool": "get_items", "when": {"order_id": "<order_id>"}},
                    }
                    answers.append({"s": i, "class": "rule", "rules": [rule]})
                else:
                    answers.append({"s": i, "class": "rule", "rules": []})
            return json.dumps({"statements": answers})
        if (
            "code-checkable" in system
        ):  # retype: the statement is on the first line (the catalog names arguments too)
            statement = user.split("\n")[0]
            if "coupon" in statement:
                rule = {
                    "type": "allowed_values",
                    "tool": "create_order",
                    "arg": "tier",
                    "values": ["promo"],
                    "when": {"coupon": "*"},
                }
                return json.dumps({"rules": [rule]})
            if "owner_id" in statement:
                return json.dumps(
                    {
                        "rules": [
                            {"type": "required_args", "tool": "create_order", "args": ["owner_id"]},
                            {"type": "required_args", "tool": "update_order", "args": ["owner_id"]},
                        ]
                    }
                )
            return json.dumps({"rules": []})
        if "role permission table" in system:
            if "Admin (admin)" in user:
                return json.dumps({"role": "Admin", "all": True, "allowed": []})
            return json.dumps(
                {"role": "Clerk", "all": False, "allowed": ["create_order", "update_order", "delete_order"]}
            )
        return json.dumps({"links": []})


def _build(**kwargs):
    model = Scripted()
    ps = PolicySet(documents=[load_document(POLICY, meta={"id": "shop"})])
    return build_policy_set(ps, SPECS, model, **kwargs), model, ps


def _by_type(build):
    out = {}
    for r in build.rules:
        out.setdefault(r.type, []).append(r)
    return out


def test_build_runs_every_repair():
    build, model, ps = _build()
    rules = _by_type(build)
    # verify tools: the argument name written in the statement
    verify = [r for r in rules["verify"] if "assigned_to" in r.rule["statement"]]
    assert verify[0].rule["tools"] == ["create_order", "update_order"]
    assert verify[0].rule["tools_by"] == "statement text"
    # retype: the coupon statement became a typed rule, and the bare "*" is a list
    typed = [r for r in rules["allowed_values"] if r.rule.get("retyped_from")]
    assert typed[0].rule["when"] == {"coupon": ["*"]} and typed[0].statement_ids
    # overreach: the owner rule binds creation only
    owner = rules["required_args"]
    assert [r.rule["tool"] for r in owner] == ["create_order"]
    # the duty: a bare status is a list and a placeholder is "an argument is given"
    duty = rules["duty"][0].rule
    assert duty["trigger"]["when"] == {"status": ["closed"]} and duty["then"]["when"] == {"order_id": ["*"]}
    # grants: Admin has everything, Clerk lacks the invoice tool; remind only; the table rows need no verify rule
    grants = rules["role_permission"]
    assert [g.id for g in grants] == ["grant_clerk"]
    assert grants[0].rule["tools"] == ["create_invoice"] and grants[0].may_deny is False
    assert grants[0].statement_ids == [ps.documents[0].sections[-1].statements[1].id]
    assert not [
        r for r in rules.get("verify", []) if "Clerk" in r.rule["statement"] or "Admin" in r.rule["statement"]
    ]
    assert not [r for r in rules.get("guidance", []) if "Admin" in r.rule["text"]]
    outcome = {c.statement_id: c.kind for c in build.coverage}
    admin, clerk = (st.id for st in ps.documents[0].sections[-1].statements)
    assert outcome[admin] == "out_of_scope" and outcome[clerk] == "hard_rule"
    # a verify rule no write tool can trigger became guidance, and none is left without a tool
    assert any("weather" in r.rule["text"] for r in rules["guidance"])
    assert all(r.rule.get("tools") for r in rules.get("verify", []))
    assert build.stats["repair"]["grant_rules"] == 1 and build.stats["repair"]["verify_to_guidance"] >= 1
    assert not [c for c in build.coverage if c.kind == "verify" and "weather" in c.quote]
    # every coverage item that says what became of a statement uses an outcome of design section 9
    assert {c.kind for c in build.coverage} <= {
        "hard_rule",
        "judgment_rule",
        "guidance",
        "procedure_step",
        "definition",
        "out_of_scope",
        "question",
        "classify_note",
    }
    # the "must" statement that no tool fits is guidance and also a policy question, which the package carries
    weather = next(r for r in rules["guidance"] if "weather" in r.rule["text"])
    assert [q["statement_ids"] for q in build.policy_questions] == [weather.statement_ids]
    assert build.stats["repair"]["questions"] == 1

    # the package passes the checks that new packages must pass, for slots and for triggers
    doc = {
        "policy_hash": "x",
        "rules": [r.rule for r in build.rules],
        "build": to_json(build),
        "policy_set": to_json(ps),
    }
    pkg = from_build(doc)
    found = {f.code for f in validate_package(pkg, SPECS, legacy=False)}
    assert not found & {"invalid_slots", "rule_no_trigger", "unfilled_slot", "unknown_tool", "unknown_arg"}
    assert pkg.policy_questions == build.policy_questions
    assert to_build(pkg) == doc  # the questions come back in the build, where they were written


def test_grant_rule_reminds_and_does_not_deny():
    build, _, ps = _build()
    pkg = from_build({"policy_hash": "x", "rules": [r.rule for r in build.rules], "build": to_json(build)})
    grant = next(r for r in pkg.rules if r.id == "grant_clerk")
    assert grant.level == "remind"
    hooks = RuleSetHooks.from_package(pkg, SPECS, principal={"id": "u", "role": "clerk"}, brief=False)
    hooks.before_tool_call("create_invoice", {"order_id": "o1", "amount": "5"})  # reminds, never raises
    assert [r["outcome"] for r in hooks.records if r.get("rule_id") == "grant_clerk"] == ["remind"]


def test_grants_deny_when_asked():
    build, _, _ = _build(grant_may_deny=True)
    grant = next(r for r in build.rules if r.id == "grant_clerk")
    assert grant.may_deny is True and grant.rule["may_deny"] is True


def test_without_repair_the_build_is_what_the_prose_step_gave():
    build, model, _ = _build(repair=False)
    kinds = {r.type for r in build.rules}
    assert "guidance" not in kinds and "role_permission" not in kinds and "repair" not in build.stats
    assert any(r.type == "verify" and not r.rule.get("tools") for r in build.rules)
    assert not [c for c in model.calls if c.startswith("You read one row")]


def test_normalize_when_reads_values_as_the_runtime_does():
    rule = {
        "type": "duty",
        "when": {"a": True, "b": ["x"], "c": 3},
        "trigger": {"tool": "t", "when": {"status": "closed"}},
        "then": {"tool": "u", "when": {"id": "<the id>", "folder": "General", "flags": {"k": 1}}},
    }
    assert normalize_when(rule) == 5
    assert rule["when"] == {"a": [True], "b": ["x"], "c": [3]}
    assert rule["trigger"]["when"] == {"status": ["closed"]}
    assert rule["then"]["when"] == {
        "id": ["*"],
        "folder": ["General"],
        "flags": {"k": 1},
    }  # a dict is left to validate
    assert normalize_when(rule) == 0


def test_a_duty_with_a_normalized_when_is_met_by_the_call_it_asks_for():
    """The runtime reads a string in ``when`` as its characters, so a bare ``General`` never matched ``General``."""
    duty = {
        "id": "d",
        "type": "duty",
        "trigger": {"tool": "update_order", "when": {"status": "closed"}},
        "then": {"tool": "get_items", "when": {"order_id": "<order_id>", "tier": "gold"}},
    }
    assert normalize_when(duty) == 3
    hooks = RuleSetHooks([duty], SPECS, brief=False)
    hooks.after_tool_call("update_order", {"order_id": "o1", "status": "closed"}, {"ok": True})
    assert "still open" in hooks.before_reply("done").message
    hooks.after_tool_call("get_items", {"order_id": "o1", "tier": "gold"}, {"items": []})
    assert hooks.before_reply("done").message is None


def test_partial_update_overreach_needs_update_words():
    rule = {"type": "required_args", "tool": "update_order", "args": ["owner_id"]}
    assert partial_update_overreach(rule, "Every order must have an owner_id.")
    assert not partial_update_overreach(rule, "An update must keep the owner_id.")
    assert not partial_update_overreach(
        {**rule, "tool": "create_order"}, "Every order must have an owner_id."
    )
    # a rule with a condition binds only the calls that meet it: "cannot become resolved without a note"
    when = {**rule, "when": {"status": ["resolved"]}}
    assert not partial_update_overreach(when, "An order cannot become resolved without an owner_id.")


def test_map_verify_tools_is_gone_and_an_old_build_still_loads():
    assert not hasattr(rule_repair, "map_verify_tools")
    assert not hasattr(rule_repair, "MAP_USER")
    old = {  # a build made with the removed step: its tools came from a model map
        "policy_hash": "old",
        "rules": [
            {
                "id": "v1",
                "type": "verify",
                "statement": "Orders need an owner.",
                "tools": ["create_order"],
                "tools_by": "model map",
            }
        ],
    }
    pkg = from_build(old)
    assert pkg.rules[0].rule["tools_by"] == "model map"
    RuleSetHooks.from_package(pkg, SPECS, brief=False)


def _retype(statement, answer, rule_extra=None):
    """``retype_rule`` for one verify rule, with a model that returns ``answer`` and counts its calls."""
    calls = []

    def model(system, user):
        calls.append(user)
        return json.dumps(answer)

    rule = {"id": "v", "type": "verify", "statement": statement, **(rule_extra or {})}
    by_name = {s["name"]: s for s in SPECS}
    good, status = rule_repair.retype_rule(rule, by_name, rule_repair._write_catalog(SPECS), model)
    return good, status, calls


def test_retype_keeps_only_arguments_the_statement_names():
    every_arg = {"type": "required_args", "tool": "create_order", "args": ["customer_id", "owner_id", "tier"]}
    good, _, _ = _retype("Every order must have an owner_id.", {"rules": [every_arg]})
    assert good == []  # a general statement filled in with every argument of a tool
    one_arg = {"type": "required_args", "tool": "create_order", "args": ["owner_id"]}
    good, _, _ = _retype("Every order must have an owner.", {"rules": [one_arg]})
    assert [g["args"] for g in good] == [["owner_id"]] and good[0]["id"] == "v_t0"
    assert good[0]["retyped_from"] == "v" and good[0]["may_deny"] is True


def test_retype_leaves_a_delayed_change_to_the_system_and_skips_the_call():
    answer = {
        "rules": [{"type": "allowed_values", "tool": "update_order", "arg": "status", "values": ["closed"]}]
    }
    good, status, calls = _retype("Orders in status shipped will become closed after 72 hours.", answer)
    assert good == [] and status == "delayed" and calls == []


def test_retype_carries_the_exceptions_of_the_statement():
    unless = [{"condition": "unless the owner is away"}]
    one_arg = {"type": "required_args", "tool": "create_order", "args": ["owner_id"]}
    good, _, _ = _retype("Every order must have an owner.", {"rules": [one_arg]}, {"unless": unless})
    assert good[0]["unless"] == unless


POLICY_MUST_DEFINITION = """## 1. Orders

- The warehouse ships what is packed.
- Every order must have an owner_id.
- The assigned_to person must be active.
"""


def test_a_must_statement_called_a_definition_goes_through_verify_and_is_never_a_gap():
    class Model(Scripted):
        def __call__(self, system, user):
            if "classify" in system:
                return json.dumps(
                    {
                        "statements": [
                            {"n": i, "types": ["open_judgment"], "modality": "must"} for i in (1, 2, 3)
                        ]
                    }
                )
            if "typed rules" in system:
                return json.dumps(
                    {
                        "statements": [
                            {"s": 1, "class": "definition", "reason": "describes the system"},
                            {"s": 2, "class": "out_of_scope"},
                            {"s": 3, "class": "definition"},
                        ]
                    }
                )
            return super().__call__(system, user)

    ps = PolicySet(documents=[load_document(POLICY_MUST_DEFINITION, meta={"id": "shop"})])
    build = build_policy_set(ps, SPECS, Model())
    ships, owner, assigned = (st.id for st in ps.documents[0].sections[0].statements)
    kinds = {}
    for c in build.coverage:
        kinds.setdefault(c.statement_id, set()).add(c.kind)
    # no tool fits the first: guidance, and a question that the owner can answer
    assert kinds[ships] == {"guidance", "question"}
    assert [q["statement_ids"] for q in build.policy_questions] == [[ships]]
    # the second names an argument: the retype step finds a typed rule for it, which a "definition" call would have hidden
    assert kinds[owner] == {"hard_rule"}
    assert [r.rule["tool"] for r in build.rules if r.type == "required_args"] == ["create_order"]
    # the third has tools by the argument it names and no typed rule: a judgment rule, checked at run time
    assert kinds[assigned] == {"judgment_rule"}
    assert [r.rule["tools"] for r in build.rules if r.type == "verify"] == [["create_order", "update_order"]]
    doc = {
        "policy_hash": "x",
        "rules": [r.rule for r in build.rules],
        "build": to_json(build),
        "policy_set": to_json(ps),
    }
    found = validate_package(from_build(doc), SPECS, legacy=False)
    assert not [f for f in found if f.code in ("coverage_incomplete", "rule_no_trigger")]


def test_a_may_statement_called_a_definition_stays_a_coverage_item():
    class Model(Scripted):
        def __call__(self, system, user):
            if "classify" in system:
                return json.dumps({"statements": [{"n": 1, "types": ["definition"], "modality": "may"}]})
            if "typed rules" in system:
                return json.dumps({"statements": [{"s": 1, "class": "definition", "reason": "a term"}]})
            return super().__call__(system, user)

    ps = PolicySet(documents=[load_document("## 1. Terms\n\n- A tier is a price level.\n", meta={"id": "t"})])
    build = build_policy_set(ps, SPECS, Model())
    assert build.rules == [] and [c.kind for c in build.coverage] == ["definition"]


def test_the_prose_step_keeps_a_rule_with_a_condition_on_an_update_tool():
    """ "An order cannot become resolved without an owner_id" binds the update that sets the status."""

    class Model(Scripted):
        def __call__(self, system, user):
            if "classify" in system:
                return json.dumps(
                    {"statements": [{"n": 1, "types": ["required_field"], "modality": "must_not"}]}
                )
            if "typed rules" in system:
                rule = {
                    "type": "required_args",
                    "tool": "update_order",
                    "args": ["owner_id"],
                    "when": {"status": ["resolved"]},
                    "quote": "An order cannot become resolved without an owner_id",
                }
                return json.dumps({"statements": [{"s": 1, "class": "rule", "rules": [rule]}]})
            return super().__call__(system, user)

    policy = "## 1. Orders\n\n- An order cannot become resolved without an owner_id.\n"
    ps = PolicySet(documents=[load_document(policy, meta={"id": "shop"})])
    build = build_policy_set(ps, SPECS, Model())
    assert [(r.type, r.rule["tool"], r.rule["when"]) for r in build.rules] == [
        ("required_args", "update_order", {"status": ["resolved"]})
    ]


ITEM_SPECS = [
    {
        "name": "link_item",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item identifier (e.g., IT_001)"},
                "kind": {"type": "string", "enum": ["a", "b"]},
                "owner_id": {"type": "string"},
            },
        },
    }
]


def test_a_list_of_values_cannot_fit_an_identifier_argument():
    from polisys.validate import check_rule

    by_name = {s["name"]: s for s in ITEM_SPECS}
    statement = "Only items with status in_use or maintenance can be linked."
    rule = {"type": "allowed_values", "tool": "link_item", "arg": "item", "values": ["in_use", "maintenance"]}
    rule["quote"] = statement
    assert "record identifier" in check_rule(rule, by_name, statement)
    assert "record identifier" in check_rule(
        {**rule, "arg": "owner_id"}, by_name, statement
    )  # name ends in _id
    # the statement writes the argument's name: an allow list of identifiers is possible
    assert check_rule(rule, by_name, statement + " The item must be an IT_001 style value.") is None
    # an enum is a list of words by the tool's own definition
    assert check_rule({**rule, "arg": "kind", "values": ["a"]}, by_name, statement) is None


def test_validate_package_reports_a_list_of_values_on_an_identifier_argument():
    rule = {
        "id": "r1",
        "type": "allowed_values",
        "tool": "link_item",
        "arg": "item",
        "values": ["in_use"],
        "quote": "Only items with status in_use can be linked.",
    }
    pkg = from_build({"policy_hash": "x", "rules": [rule]})
    found = [f for f in validate_package(pkg, ITEM_SPECS, legacy=False) if f.code == "identifier_values"]
    assert len(found) == 1 and found[0].severity == "error"


def test_role_rows_take_table_rows_that_have_no_role_header():
    from polisys.model import Statement, StatementType

    row = Statement(
        id="s1",
        text="| Agent (agent) | Frontline | Full CRUD on orders |",
        types=[StatementType.ROLE_PERMISSION],
        source={"table_header": ["Manager (manager)", "Supervision", "Full CRUD"], "section": "2", "line": 5},
    )
    headed = Statement(
        id="s2", text="Clerk (clerk) | Packs orders", source={"table_header": ["Role", "Duty"]}
    )
    prose = Statement(
        id="s3", text="Only a clerk may delete an order.", types=[StatementType.ROLE_PERMISSION]
    )
    other = Statement(id="s4", text="Orders have a tier.", types=[StatementType.ALLOWED_VALUES])
    rows = rule_repair.role_rows([row, headed, prose, other])
    assert [r["id"] for r in rows] == ["s1", "s2"]
    assert rows[0]["header"] == [] and rows[1]["header"] == [
        "Role",
        "Duty",
    ]  # a header that is no role header is dropped
    assert rule_repair.role_rows(
        [{"id": "s5", "text": "| Clerk (clerk) | x |", "types": ["role_permission"]}]
    )


def test_failed_model_calls_are_counted_not_taken_for_nothing_to_do():
    class Broken(Scripted):
        def __call__(self, system, user):
            if "typed rules" in system or "classify" in system or "find links" in system:
                return super().__call__(system, user)
            if "role permission table" in system:
                raise RuntimeError("gateway timeout")
            return "not json at all"  # the retype answer

    ps = PolicySet(documents=[load_document(POLICY, meta={"id": "shop"})])
    build = build_policy_set(ps, SPECS, Broken())
    stats = build.stats["repair"]
    assert stats["grant_errors"] == 2 and stats["grant_rules"] == 0
    assert stats["retype_errors"] >= 3 and stats["verify_retyped"] == 0
    ok = _build()[0].stats["repair"]
    assert ok["grant_errors"] == 0 and ok["retype_errors"] == 0


def test_a_delayed_verify_rule_with_tools_becomes_guidance():
    verify = {
        "id": "v1",
        "type": "verify",
        "statement": "Orders in status shipped will become closed after 72 hours.",
        "tools": ["update_order"],
    }
    rule = rule_repair._wrap(verify, ["s1"], {}, "verifier")
    rules, changed, questions, stats = rule_repair.repair_rules([rule], [], SPECS, Scripted())
    assert [r.type for r in rules] == ["guidance"] and rules[0].statement_ids == ["s1"]
    assert stats["delayed_to_guidance"] == 1 and stats["retype_errors"] == 0 and questions == []
    assert [(c.statement_id, c.kind) for c in changed] == [("s1", "guidance")]


def test_the_prose_prompt_lists_membership():
    from polisys.handlers import prose

    assert "- membership:" in prose.RULE_GUIDE


def test_a_rule_with_an_empty_slot_is_not_kept():
    from polisys.validate import slot_problem

    assert "values" in slot_problem(
        {"type": "allowed_values", "tool": "create_order", "arg": "tier", "values": []}
    )
    assert "text" in slot_problem({"type": "guidance", "quote": "x", "tools": []})
    assert (
        slot_problem({"type": "allowed_values", "tool": "create_order", "arg": "tier", "values": ["a"]})
        is None
    )
    # the retype step drops it, and the prose step sends the reason back and falls back to the statement text
    empty = {"type": "allowed_values", "tool": "create_order", "arg": "tier", "values": []}
    good, status, _ = _retype("Orders with a tier must be gold.", {"rules": [empty]})
    assert good == [] and status == "ok"

    class Model(Scripted):
        def __call__(self, system, user):
            if "classify" in system:
                return json.dumps({"statements": [{"n": 1, "types": ["open_judgment"], "modality": "must"}]})
            if "typed rules" in system:
                bare = {"type": "guidance", "quote": "Orders are kind"}  # no text
                return json.dumps({"statements": [{"s": 1, "class": "guidance", "rules": [bare]}]})
            return super().__call__(system, user)

    ps = PolicySet(
        documents=[load_document("## 1. Orders\n\n- Orders are kind to staff.\n", meta={"id": "o"})]
    )
    build = build_policy_set(ps, SPECS, Model())
    assert [r.rule.get("text") for r in build.rules] == ["Orders are kind to staff."]
    doc = {"policy_hash": "x", "rules": [r.rule for r in build.rules], "build": to_json(build)}
    assert not [
        f for f in validate_package(from_build(doc), SPECS, legacy=False) if f.code == "unfilled_slot"
    ]


def test_a_row_that_grants_every_tool_is_not_out_of_scope_when_a_rule_already_names_it():
    from polisys.model import Statement

    row = Statement(
        id="s1",
        text="| Admin (admin) | Full CRUD on all tables |",
        source={"table_header": ["Role", "Access"]},
    )
    guidance = rule_repair._wrap(
        {"id": "g1", "type": "guidance", "text": row.text, "tools": [], "quote": row.text},
        ["s1"],
        {},
        "brief",
    )
    verify = rule_repair._wrap(
        {"id": "v1", "type": "verify", "statement": row.text, "tools": []}, ["s1"], {}, "verifier"
    )

    def model(system, user):
        return json.dumps({"role": "admin", "all": True, "allowed": []})

    rules, changed, _, stats = rule_repair.repair_rules([guidance, verify], [row], SPECS, model)
    assert [r.id for r in rules] == ["g1"] and changed == [] and stats["role_rows_all"] == 1
