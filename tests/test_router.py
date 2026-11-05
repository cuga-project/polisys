"""Routers and expanders by name (design section 11), and priority classes (design section 10)."""

import json

import pytest

from polisys.hooks import HookRegistry
from polisys import router as R
from polisys.dry_run import RecordingModels
from polisys.model import (
    Document,
    Link,
    LinkType,
    PolicyPackage,
    Procedure,
    Section,
    Statement,
    Step,
    Trust,
    authority_of,
    priority_ranks,
    resolve_conflict,
    PolicySet,
)
from polisys.obligations import ObligationLedger
from polisys.package import from_build, from_store_graph
from polisys.procedure_store import ProcedureStore
from polisys.runtime import RunContext, RuntimeWiringError, build_runtime
from polisys.store_gate import StoreGate
from polisys.validate import resolve_conflicts, validate_package

from .test_runtime_assembly import CTX, GATE_ON, config

GRAPH = {
    "id": "sop",
    "escalation": {
        "name": "Escalate",
        "steps": ["Tell the lead."],
        "must_not": ["Do not reply to the customer."],
    },
    "procedures": [
        {
            "id": "1.1",
            "name": "Open a ticket",
            "applies_when": "a customer reports a problem",
            "steps": ["Open a ticket.", "Tell the customer."],
            "must_not": ["Do not open two tickets."],
            "decisions": [
                {"id": "d1", "rule": "Is the customer a VIP?", "then": "Escalate.", "else": "Go on."}
            ],
            "links": [{"type": "next", "to": "1.2"}],
        },
        {
            "id": "1.2",
            "name": "Close a ticket",
            "applies_when": "the problem is solved",
            "steps": ["Close the ticket."],
            "must_not": ["Do not close early."],
            "links": [{"type": "refers_to", "to": "1.3"}],
        },
        {
            "id": "1.3",
            "name": "Refund",
            "applies_when": "a customer wants money back",
            "steps": ["Pay the refund."],
            "must_not": ["Do not refund twice."],
            "decisions": [{"id": "d2", "rule": "Is it over the limit?"}],
            "links": [],
        },
        {
            "id": "1.4",
            "name": "Privacy request",
            "applies_when": "a customer asks to erase data",
            "steps": ["Erase the data."],
            "must_not": ["Do not erase without proof of identity."],
            "links": [{"type": "exception_of", "to": "1.1"}],
        },
        {
            "id": "1.5",
            "name": "Old rule",
            "steps": ["Do it the old way."],
            "must_not": ["Old must-not."],
            "links": [],
        },
        {
            "id": "1.6",
            "name": "New rule",
            "steps": ["Do it the new way."],
            "must_not": ["New must-not."],
            "links": [{"type": "supersedes", "to": "1.5"}],
        },
        {
            "id": "2.1",
            "name": "Cycle one",
            "steps": ["One."],
            "must_not": ["Cycle one must-not."],
            "links": [{"type": "refers_to", "to": "2.2"}],
        },
        {
            "id": "2.2",
            "name": "Cycle two",
            "steps": ["Two."],
            "must_not": ["Cycle two must-not."],
            "links": [{"type": "refers_to", "to": "2.3"}],
        },
        {
            "id": "2.3",
            "name": "Cycle three",
            "steps": ["Three."],
            "must_not": ["Cycle three must-not."],
            "links": [{"type": "refers_to", "to": "2.1"}],
        },
    ],
}


def legacy_rules(store):
    """``StoreGate._rules`` as it was before routers (gate ``fcb15283d``), for the comparison."""
    out = []
    for c in store.calls:
        for pid in c.get("expanded") or []:
            p = store.procs.get(pid) or {}
            out += [f"[{pid}] MUST NOT: {m}" for m in p.get("must_not") or []]
            out += [
                f"[{pid}] DECISION {d.get('id')}: {d.get('rule')}"
                + (f"; then: {d['then']}" if d.get("then") else "")
                + f"; else: {d.get('else', '')}"
                for d in p.get("decisions") or []
            ]
    esc = (store.g.get("escalation") or {}).get("must_not") or []
    out += [f"[escalation] MUST NOT: {m}" for m in esc]
    return list(dict.fromkeys(out))


def model(replies=None):
    prompts = []

    def call(system, user):
        prompts.append(user)
        return (replies or ['{"breaks": false}'])[0]

    call.prompts = prompts
    return call


# -- the scope of the gate as measured ---------------------------------------------------------------------------------

OPENINGS = [
    [],
    [["1.1"]],
    [["1.2"]],
    [["1.2"], ["1.1"]],
    [["1.1", "1.3"], ["1.1"]],
    [["1.2", "nope"]],
    [["nope"]],
    [["§1.1"]],
    [["1.4", "1.6", "2.1"]],
    [list(p["id"] for p in GRAPH["procedures"])],
]


@pytest.mark.parametrize("opened", OPENINGS)
def test_agent_opened_with_next_links_gives_exactly_the_scope_the_gate_always_had(opened):
    pkg = from_store_graph(GRAPH)
    store = ProcedureStore.from_package(pkg)
    by_package = StoreGate.from_package(pkg, model(), store=store)
    alone = StoreGate(ProcedureStore(GRAPH), model())  # a gate on a store only: its own graph is the package
    for ids in opened:
        store.get_procedure(ids)
        alone.store.get_procedure(ids)
    store.get_template(["1"])  # a call that opens no procedure
    assert by_package._rules() == legacy_rules(store)
    assert alone._rules() == legacy_rules(alone.store)
    assert by_package._rules() == alone._rules()
    assert by_package._rules("any query") == by_package._rules()  # this router ignores the query


def test_the_default_gate_takes_the_default_router_and_expander():
    pkg = from_store_graph(GRAPH)
    gate = StoreGate.from_package(pkg, model(), store=ProcedureStore.from_package(pkg))
    assert (gate.scope.router.name, gate.scope.expander.name) == ("agent_opened", "next_links")


# -- a method is swapped by a name, with no other code change ---------------------------------------------------------


class Dummy(R.Router):
    name = "dummy"

    def __init__(self, **_):
        pass

    def select(self, query, pkg):
        return [pkg.procedures[-1].id]  # the last procedure, whatever the agent opened


class NoEscalation(R.Expander):
    name = "no_escalation"

    def __init__(self, **_):
        pass

    def expand(self, ids, pkg):
        scope = R._scope_of(ids, pkg)
        scope.rules = [r for r in scope.rules if r.owner != "escalation"]
        return scope


@pytest.fixture
def dummy_methods():
    R.register_router("dummy", Dummy)
    R.register_expander("no_escalation", NoEscalation)
    yield
    del R.ROUTERS["dummy"], R.EXPANDERS["no_escalation"]


class Scripted(RecordingModels):
    """Models that answer with fixed text and keep what they were asked."""

    def __init__(self, answers=None):
        super().__init__()
        self.answers, self.calls = answers or {}, []

    def _scripted(self, kind, name):
        def call(*args):
            self.calls.append((kind, name, args))
            return self.answers.get(name, '{"breaks": false}' if kind == "json" else {"applies": 0.5})

        call.model_name, call.usage = name, {"calls": 0}
        return call

    def json_model(self, model, **kwargs):
        super().json_model(model, **kwargs)
        return self._scripted("json", model)

    def questions(self, model, **kwargs):
        super().questions(model, **kwargs)
        return self._scripted("questions", model)


def gate_runtime(router=None, **checkers):
    cfg = config(
        {**GATE_ON, "router": router or {}, **checkers},
        {"procedure_tools": "on"},
        package="p.json",
    )
    models = Scripted()
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    return parts, models


def gate_prompt(parts, models):
    """What the gate sends to its model for an acting call."""
    before = len(models.calls)
    parts.hooks[0].before_tool_call("update_ticket", {"status": "closed"})
    sent = [c for c in models.calls[before:] if c[1] == "m/g"]
    return sent[-1][2][1] if sent else ""


def test_a_router_registered_by_a_test_changes_the_scope_of_the_gate_with_no_other_code_change(dummy_methods):
    parts, models = (
        gate_runtime()
    )  # as measured: the agent opened nothing, so only the escalation rule is in scope
    assert "Do not refund" not in gate_prompt(parts, models)
    assert "Do not reply to the customer." in gate_prompt(parts, models)

    parts, models = gate_runtime({"kind": "dummy"})  # one more line in a profile
    gate = parts.hooks[0]
    prompt = gate_prompt(parts, models)
    assert gate.scope.router.name == "dummy"
    assert "[2.3] MUST NOT: Cycle three must-not." in prompt
    assert "Do not reply to the customer." in prompt  # next_links still adds the escalation rules


def test_an_expander_registered_by_a_test_changes_the_scope_with_no_other_code_change(dummy_methods):
    parts, models = gate_runtime({"kind": "dummy", "expand": "no_escalation"})
    prompt = gate_prompt(parts, models)
    assert "[2.3] MUST NOT: Cycle three must-not." in prompt
    assert "Do not reply to the customer." not in prompt


def test_the_ledger_has_no_retrieval_scope_and_a_router_with_only_a_ledger_on_is_an_unread_setting():
    import inspect

    from polisys.model import Obligation

    assert "scope" not in inspect.signature(ObligationLedger).parameters  # no router call for one audit line
    pkg = from_store_graph(GRAPH)
    pkg.obligations = [
        Obligation(
            id="ob1",
            origin="package",
            scope={},
            effect={"kind": "write", "tools": ["update_ticket"], "match": {}},
            values={},
            level="follow_up",
            source={"section": "1", "quote": "Close the ticket."},
        )
    ]
    cfg = config(
        {"router": {"kind": "bm25"}},
        {},
        package="p.json",
        sources={"request": True},
        obligations={"producers": ["package"], "model": "m/o"},
    )
    with pytest.raises(RuntimeWiringError, match="checkers.router.kind.*only the gate reads the scope"):
        build_runtime(cfg, pkg, CTX, models=Scripted())


def test_a_router_that_does_not_read_what_the_agent_opened_works_with_the_procedure_tools_off(dummy_methods):
    cfg = config({**GATE_ON, "router": {"kind": "dummy"}}, {}, package="p.json")  # procedure_tools off
    models = Scripted()
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    assert "[2.3] MUST NOT: Cycle three must-not." in gate_prompt(parts, models)
    default = config(GATE_ON, {}, package="p.json")  # agent_opened: nothing else opens a procedure
    with pytest.raises(RuntimeWiringError, match="agent_opened needs organizers.procedure_tools"):
        build_runtime(default, from_store_graph(GRAPH), CTX, models=Scripted())


# -- names ------------------------------------------------------------------------------------------------------------


def test_the_registered_names():
    assert R.router_names() == ["agent_opened", "bm25", "bm25_kev", "llm_cards"]
    assert R.expander_names() == ["k_hops", "next_links"]


@pytest.mark.parametrize(
    "router, path, bad",
    [
        ({"kind": "pagerank"}, "checkers.router.kind", "pagerank"),
        ({"expand": "walk"}, "checkers.router.expand", "walk"),
    ],
)
def test_an_unknown_name_in_a_profile_names_it_and_the_names_there_are(router, path, bad):
    with pytest.raises(RuntimeWiringError) as err:
        gate_runtime(router)
    text = str(err.value)
    assert path in text and f"'{bad}'" in text
    names = R.router_names() if "kind" in path else R.expander_names()
    assert all(n in text for n in names), text


def test_a_router_that_needs_a_model_says_so_when_the_profile_gives_none():
    with pytest.raises(RuntimeWiringError, match="its model is none"):
        gate_runtime({"kind": "llm_cards"})
    with pytest.raises(RuntimeWiringError, match="narrow_questions"):
        gate_runtime({"kind": "bm25_kev"})
    with pytest.raises(ValueError, match="needs a model"):
        R.make_router("llm_cards")


@pytest.mark.parametrize(
    "router, path",
    [
        ({"hops": 5}, "checkers.router.hops"),  # next_links takes no hops
        ({"top_k": 9}, "checkers.router.top_k"),  # agent_opened returns what the agent opened
        ({"kind": "bm25", "model": "m/x"}, "checkers.router.model"),  # bm25 asks no model
        ({"kind": "bm25", "hops": 2}, "checkers.router.hops"),
        ({"kind": "bm25", "expand": "k_hops", "model": "m/x"}, "checkers.router.model"),
    ],
)
def test_a_router_setting_that_the_chosen_methods_do_not_read_raises(router, path):
    with pytest.raises(RuntimeWiringError, match=f"{path}.*does not read it"):
        gate_runtime(router)


def test_the_settings_a_method_reads_build():
    gate_runtime({"kind": "bm25", "top_k": 2})
    gate_runtime({"kind": "bm25", "expand": "k_hops", "hops": 2, "top_k": 1})
    gate_runtime({"kind": "llm_cards", "model": "m/r", "top_k": 2})


def test_a_router_setting_with_nothing_to_read_it_raises():
    cfg = config({"router": {"kind": "bm25"}}, {}, package=None)
    with pytest.raises(RuntimeWiringError, match="checkers.router.kind"):
        build_runtime(cfg, None, CTX, models=Scripted())


# -- the registered routers -------------------------------------------------------------------------------------------


def test_bm25_ranks_the_cards_and_needs_no_model():
    pkg = from_store_graph(GRAPH)
    ids = R.make_router("bm25", top_k=None).select("pay the refund the customer wants money back", pkg)
    assert ids[0] == "1.3" and len(ids) == len(pkg.procedures)
    assert R.make_router("bm25", top_k=2).select("erase data identity", pkg)[0] == "1.4"
    assert R.make_router("bm25").select(R.Query("erase data"), pkg) == R.make_router("bm25").select(
        "erase data", pkg
    )


def test_a_router_runs_on_a_package_from_a_skill_whose_steps_have_text_and_no_quote():
    pkg = from_build(
        {
            "rules": [],
            "procedures": [
                {
                    "id": "p1",
                    "name": "Tickets",
                    "steps": [{"id": "s1", "action": "update_ticket", "text": "Close the ticket."}],
                },
                {
                    "id": "p2",
                    "name": "Refunds",
                    "steps": [{"id": "s2", "action": "refund", "quote": "Pay the refund."}],
                },
            ],
        }
    )
    assert R.make_router("bm25").select("close the ticket", pkg)[0] == "p1"
    from polisys.package import to_build
    from polisys.sop_retrieval import SopIndex

    assert [p["id"] for p in SopIndex(to_build(pkg)).rank("pay the refund")][0] == "p2"


def test_a_search_router_on_a_package_of_a_graph_has_no_section_and_still_lists_a_checklist():
    from polisys.package import to_build
    from polisys.sop_retrieval import SopIndex

    index = SopIndex(
        to_build(from_store_graph(GRAPH))
    )  # a procedure with no source: this raised KeyError "document"
    text = index.checklist(index.procs[0], "open a ticket")
    assert "Open a ticket." in text


def test_bm25_kev_reranks_with_the_decision_model_and_is_traced():
    models = Scripted()
    cfg = config(
        {**GATE_ON, "router": {"kind": "bm25_kev", "top_k": 2}, "narrow_questions": "k/kev"},
        {"procedure_tools": "on"},
        package="p.json",
    )
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    gate = parts.hooks[0]
    assert gate.scope.router.name == "bm25_kev"
    ids = gate.scope.router.select("close the ticket, it is solved", gate.scope.pkg)
    assert len(ids) == 2 and ids[0] == "1.2"
    assert models.calls and all(c[:2] == ("questions", "k/kev") for c in models.calls)
    assert len(models.calls) == 5  # one question for each of the five best candidates
    assert (
        "router" in parts.usage() and parts.usage()["router"]["model"] == "k/kev"
    )  # the traced one is in the run's usage


def test_llm_cards_asks_a_model_to_rank_the_cards_and_is_traced():
    models = Scripted({"m/r": json.dumps({"procedures": ["1.3", "nope", "1.3", "1.4", "1.2"]})})
    cfg = config(
        {**GATE_ON, "router": {"kind": "llm_cards", "model": "m/r", "top_k": 2}},
        {"procedure_tools": "on"},
        package="p.json",
    )
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    gate = parts.hooks[0]
    assert gate.scope.router.name == "llm_cards"
    gate.before_tool_call("update_ticket", {"status": "closed"})
    asked = [c for c in models.calls if c[1] == "m/r"]
    assert len(asked) == 1 and "1.3 | Refund | a customer wants money back" in asked[0][2][1]
    assert "update_ticket" in asked[0][2][1]  # the query is the acting call
    assert "at most 2. If no procedure" in asked[0][2][1]  # top_k is in the prompt
    assert (
        "Recent data the agent read:\n<<<\n(none)\n>>>" in asked[0][2][1]
    )  # nothing read yet: the measured wording
    rules = gate._rules("update_ticket")  # (asks the model again)
    assert (
        "[1.3] MUST NOT: Do not refund twice." in rules
        and "[1.4] MUST NOT: Do not erase without proof of identity." in rules
    )
    assert not any(
        "[1.1]" in r or "Old must-not" in r for r in rules
    )  # top_k 2 cut the third; unknown and repeated ids are gone
    assert parts.usage()["router"]["model"] == "m/r"


# -- expanders --------------------------------------------------------------------------------------------------------


def ids_of(scope):
    return scope.procedures


def test_k_hops_zero_is_next_links():
    pkg = from_store_graph(GRAPH)
    for ids in (["1.1"], ["1.2", "1.4"], [], ["2.1"]):
        assert R.make_expander("k_hops", hops=0).expand(ids, pkg) == R.make_expander("next_links").expand(
            ids, pkg
        )


def test_k_hops_walks_refers_to_exception_of_and_supersedes():
    pkg = from_store_graph(GRAPH)
    walk = lambda ids, hops: ids_of(R.make_expander("k_hops", hops=hops).expand(ids, pkg))  # noqa: E731
    assert walk(["1.1"], 0) == ["1.1", "1.2"]  # the next link
    assert walk(["1.1"], 1) == ["1.1", "1.2", "1.4", "1.3"]  # the exception of 1.1, and what 1.2 refers to
    assert walk(["1.4"], 1) == ["1.4", "1.1", "1.2"]  # an exception reaches the rule it is the exception of
    assert walk(["1.3"], 1) == ["1.3", "1.2"]  # refers_to goes both ways
    assert walk(["1.5"], 1) == ["1.5", "1.6"]  # the replaced procedure reaches the one that replaces it
    assert walk(["1.6"], 3) == ["1.6"]  # and not the other way: its rules no longer hold


def test_k_hops_ends_on_a_cycle():
    pkg = from_store_graph(GRAPH)
    for hops in (1, 2, 3, 50):
        got = ids_of(R.make_expander("k_hops", hops=hops).expand(["2.1"], pkg))
        assert sorted(got) == sorted(set(got)) and set(got) <= {"2.1", "2.2", "2.3"}
    assert ids_of(R.make_expander("k_hops", hops=50).expand(["2.1"], pkg)) == ["2.1", "2.2", "2.3"]
    assert ids_of(R.make_expander("k_hops", hops=1).expand(["2.1"], pkg)) == [
        "2.1",
        "2.2",
        "2.3",
    ]  # both ways: 2.3 is near


def test_k_hops_follows_links_of_the_package_between_sections_and_statements():
    def proc(pid, section):
        return Procedure(
            id=pid,
            name=pid,
            steps=[Step(id="s", action="a", quote="q", source={"document": "d", "section": section})],
            source={"document": "d", "section": section},
        )

    doc = Document(
        id="d",
        kind="file",
        sections=[
            Section(
                id="1", statements=[Statement(id="d:1:a", text="a", source={"document": "d", "section": "1"})]
            ),
            Section(
                id="2", statements=[Statement(id="d:2:a", text="b", source={"document": "d", "section": "2"})]
            ),
            Section(id="3", statements=[]),
        ],
    )
    pkg = PolicyPackage(
        documents=[doc],
        procedures=[proc("P1", "1"), proc("P2", "2"), proc("P3", "3")],
        links=[Link(LinkType.REFERS_TO, "d:1", "d:2:a"), Link(LinkType.DEFINES, "d:2", "d:3")],
    )
    graph = R.procedure_graph(pkg)
    assert graph == {"P1": ["P2"], "P2": ["P1"], "P3": []}  # `defines` is not a walked link
    assert ids_of(R.make_expander("k_hops", hops=1).expand(["P1"], pkg)) == ["P1", "P2"]


def test_an_expander_gives_all_the_escalation_rules_and_the_rules_of_its_procedures():
    scope = R.make_expander("next_links").expand(["1.1"], from_store_graph(GRAPH))
    assert [(r.owner, r.kind) for r in scope.rules] == [
        ("1.1", "must_not"),
        ("1.1", "decision"),
        ("1.2", "must_not"),
        ("escalation", "must_not"),
    ]
    assert scope.rules[1].decision["rule"] == "Is the customer a VIP?"


# -- priority classes ------------------------------------------------------------------------------------------------


def doc(doc_id, cls=None, date="2026-01-01", rank=0, trust=Trust.TRUSTED):
    return Document(
        id=doc_id,
        kind="file",
        trust=trust,
        effective_date=date,
        authority_rank=rank,
        priority_class=cls,
        sections=[
            Section(
                id="1",
                statements=[
                    Statement(id=f"{doc_id}:1:a", text="x", source={"document": doc_id, "section": "1"})
                ],
            )
        ],
    )


CLASSES = ["regulatory", "company", "department", "team"]


def test_the_classes_map_to_ranks_from_the_highest():
    assert priority_ranks(CLASSES) == {"regulatory": 4, "company": 3, "department": 2, "team": 1}
    assert authority_of(doc("a", "company"), CLASSES) == 3
    assert authority_of(doc("a", "company"), []) == 0  # no list: the document's own rank
    assert authority_of(doc("a", "other", rank=7), CLASSES) == 7  # a class that is not listed keeps the rank


def test_a_class_decides_when_the_dates_tie():
    old_team, company = doc("t", "team"), doc("c", "company")
    ps = PolicySet(documents=[old_team, company], links=[])
    a, b = old_team.sections[0].statements[0], company.sections[0].statements[0]
    assert resolve_conflict(a, b, ps) is None  # no list: both rank 0, and nothing decides
    assert resolve_conflict(a, b, ps, CLASSES) is b
    assert resolve_conflict(b, a, ps, CLASSES) is b


def test_a_newer_date_beats_a_class_and_a_tie_on_date_and_class_is_never_decided():
    newer_team, company = doc("t", "team", date="2026-06-01"), doc("c", "company", date="2026-01-01")
    ps = PolicySet(documents=[newer_team, company], links=[])
    a, b = newer_team.sections[0].statements[0], company.sections[0].statements[0]
    assert resolve_conflict(a, b, ps, CLASSES) is a  # the date comes first
    same_a, same_b = doc("x", "company"), doc("y", "company")
    ps = PolicySet(documents=[same_a, same_b], links=[])
    assert (
        resolve_conflict(same_a.sections[0].statements[0], same_b.sections[0].statements[0], ps, CLASSES)
        is None
    )


def conflicting_package(first, second):
    def rule(rid, doc_id, values):
        return {
            "id": rid,
            "type": "allowed_values",
            "tool": "update_ticket",
            "arg": "status",
            "values": values,
            "quote": "x",
            "statement_ids": [f"{doc_id}:1:a"],
        }

    from polisys.model import Rule

    pkg = PolicyPackage(
        version="v1",
        documents=[first, second],
        rules=[
            Rule(
                id="r1",
                type="allowed_values",
                statement_ids=[f"{first.id}:1:a"],
                rule=rule("r1", first.id, ["open"]),
                level="deny",
            ),
            Rule(
                id="r2",
                type="allowed_values",
                statement_ids=[f"{second.id}:1:a"],
                rule=rule("r2", second.id, ["closed"]),
                level="deny",
            ),
        ],
    )
    return pkg


def test_resolve_conflicts_uses_the_classes_and_never_settles_a_tie():
    pkg = conflicting_package(doc("t", "team"), doc("c", "company"))
    kept, _ = resolve_conflicts(pkg)  # no list: a tie, both stay and drop to remind with a question
    assert [r.id for r in kept.rules] == ["r1", "r2"] and len(kept.policy_questions) == 1
    assert all(r.level == "remind" for r in kept.rules)
    kept, notes = resolve_conflicts(pkg, CLASSES)  # company beats team: the team rule goes
    assert [r.id for r in kept.rules] == ["r2"] and not kept.policy_questions and "r2 wins" in notes[0]
    tie = conflicting_package(doc("x", "company"), doc("y", "company"))
    kept, _ = resolve_conflicts(tie, CLASSES)  # the same class and date: still a tie
    assert [r.id for r in kept.rules] == ["r1", "r2"] and len(kept.policy_questions) == 1


def test_an_unknown_class_is_a_validate_warning_and_a_class_with_no_statement_is_fine():
    pkg = conflicting_package(doc("t", "team"), doc("c", "board"))
    found = [
        f for f in validate_package(pkg, None, priority_classes=CLASSES) if f.code == "unknown_priority_class"
    ]
    assert len(found) == 1 and found[0].severity == "warning" and "'board'" in found[0].message
    assert found[0].statement_id == "c"
    assert [
        f for f in validate_package(pkg, None) if f.code == "unknown_priority_class"
    ] == []  # no list: nothing to compare
    ok = conflicting_package(doc("t", "team"), doc("c", None))
    assert [
        f for f in validate_package(ok, None, priority_classes=CLASSES) if f.code == "unknown_priority_class"
    ] == []


def test_the_config_reads_priority_classes_at_build_and_reports_an_unknown_class():
    pkg = from_store_graph(GRAPH)
    pkg.documents = [doc("t", "team"), doc("c", "board")]
    cfg = config({**GATE_ON}, {"procedure_tools": "on"}, package="p.json", priority_classes=CLASSES)
    parts = build_runtime(cfg, pkg, CTX, models=Scripted())
    notes = [x for x in parts.audit.lines if x["rule_id"] == "priority_classes"]
    assert len(notes) == 1 and "'board'" in notes[0]["evidence"]
    with pytest.raises(
        RuntimeWiringError, match="priority_classes"
    ):  # a list with no package: nothing reads it
        build_runtime(config(priority_classes=CLASSES), None, CTX, models=Scripted())


def test_every_line_of_the_audit_carries_the_package_version():
    pkg = from_build(
        {
            "rules": [
                {
                    "id": "r1",
                    "type": "allowed_values",
                    "tool": "update_ticket",
                    "arg": "status",
                    "values": ["open"],
                    "quote": "Only open.",
                }
            ],
            "procedures": [],
        }
    )
    pkg.version = "v-2026-11-05+abc"
    cfg = config({"rules": "on"}, {}, package="p.json")
    ctx = RunContext(
        task_id="task-7", policy_version="v-abc", request="x", system_prompt="", tool_specs=CTX.tool_specs
    )
    parts = build_runtime(cfg, pkg, ctx, models=Scripted())
    assert HookRegistry(parts.hooks).before_tool_call("update_ticket", {"status": "closed"})  # denied
    rows = [x for x in parts.audit.lines if x["verdict"] == "deny"]
    assert rows and all(x["package_version"] == "v-2026-11-05+abc" for x in parts.audit.lines)


# -- fix round 1 ------------------------------------------------------------------------------------------------------


def skill_package(links):
    """A package as a skill writes it: a procedure has no source, and its id is the id of its section."""
    section = lambda sid: Section(  # noqa: E731
        id=sid,
        statements=[
            Statement(id=f"policy:{sid}:1", text=f"s{sid}", source={"document": "policy", "section": sid})
        ],
    )
    doc = Document(id="policy", kind="file", sections=[section(x) for x in ("1", "2", "3")])
    procs = [
        Procedure(id=x, name=f"P{x}", steps=[Step(text="step")], must_not=[f"must not {x}"])
        for x in ("1", "2", "3")
    ]
    return PolicyPackage(version="v", documents=[doc], procedures=procs, links=links)


def test_a_procedure_with_no_source_owns_the_section_of_its_own_id_so_k_hops_walks_a_skill_package():
    pkg = skill_package(
        [
            Link(LinkType.REFERS_TO, "policy:1:1", "policy:2"),
            Link(LinkType.EXCEPTION_OF, "policy:3:1", "policy:2:1"),
        ]
    )
    assert R.procedure_graph(pkg) == {"1": ["2"], "2": ["1", "3"], "3": ["2"]}
    walk = R.make_expander("k_hops", hops=1)
    assert walk.expand(["1"], pkg).procedures == ["1", "2"]
    assert R.make_expander("k_hops", hops=2).expand(["1"], pkg).procedures == ["1", "2", "3"]
    assert R.make_expander("next_links").expand(["1"], pkg).procedures == [
        "1"
    ]  # the walk adds what next_links does not


def test_k_hops_that_changes_nothing_on_a_package_is_a_build_error_that_says_why():
    pkg = skill_package([Link(LinkType.DEFINES, "policy:1:1", "policy:2")])  # no walked link
    assert "no link to walk" in R.make_expander("k_hops").problem(pkg)
    assert "hops is 0" in R.make_expander("k_hops", hops=0).problem(
        skill_package([Link(LinkType.REFERS_TO, "policy:1:1", "policy:2")])
    )
    assert R.make_expander("next_links").problem(pkg) == ""
    pkg.rules, pkg.obligations = [], []
    graph = {
        "id": "sop",
        "escalation": {"must_not": ["No."]},
        "procedures": [{"id": "1.1", "name": "A", "steps": ["a"], "links": []}],
    }
    cfg = config({**GATE_ON, "router": {"kind": "bm25", "expand": "k_hops"}}, {}, package="p.json")
    with pytest.raises(
        RuntimeWiringError, match="checkers.router.expand k_hops: the package has no link to walk"
    ):
        build_runtime(cfg, from_store_graph(graph), CTX, models=Scripted())


def test_a_router_that_fails_is_an_error_line_and_the_call_goes_through():
    class Raising(Scripted):
        def _scripted(self, kind, name):
            def call(*args):
                raise RuntimeError("the router model is down")

            call.model_name, call.usage = name, {"calls": 0}
            return call

    class Mixed(Scripted):  # the router's model raises; the gate's model never gets asked
        def json_model(self, model, **kwargs):
            RecordingModels.json_model(self, model, **kwargs)
            return Raising._scripted(self, "json", model) if model == "m/r" else self._scripted("json", model)

    models = Mixed()
    cfg = config(
        {**GATE_ON, "router": {"kind": "llm_cards", "model": "m/r"}},
        {"procedure_tools": "on"},
        package="p.json",
    )
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    gate = parts.hooks[0]
    assert (
        HookRegistry(parts.hooks).before_tool_call("update_ticket", {"status": "closed"}) is None
    )  # goes through
    assert gate.last_decision == "error" and gate.records[-1]["gate"] == "error"
    assert "scope: RuntimeError: the router model is down" in gate.records[-1]["error"]
    rows = [x for x in parts.audit.lines if x["component"] == "gate"]
    assert rows and rows[-1]["verdict"] == "error" and "the router model is down" in rows[-1]["evidence"]
    assert not any(c[1] == "m/g" for c in models.calls)  # the gate's model was not asked


def test_llm_cards_gets_the_action_and_the_data_apart_in_the_measured_prompt():
    models = Scripted({"m/r": json.dumps({"procedures": ["1.3"]})})
    cfg = config(
        {**GATE_ON, "router": {"kind": "llm_cards", "model": "m/r"}},
        {"procedure_tools": "on"},
        package="p.json",
    )
    parts = build_runtime(cfg, from_store_graph(GRAPH), CTX, models=models)
    gate = parts.hooks[0]
    gate.after_tool_call("get_ticket", {}, {"customer": "Ann wants her money back"})
    gate.before_tool_call("update_ticket", {"status": "closed"})
    prompt = next(c for c in models.calls if c[1] == "m/r")[2][1]
    assert 'The agent\'s action:\nupdate_ticket({"status": "closed"})\n' in prompt
    assert "<<<\nget_ticket: {'customer': 'Ann wants her money back'}\n>>>" in prompt
    assert "Most relevant first, at most 3. If no procedure" in prompt  # the default of the measured router


def test_a_method_that_is_not_the_default_shows_in_the_description_and_in_the_audit_and_the_default_does_not(
    dummy_methods,
):
    from polisys.dry_run import describe_parts

    def attrs(router):
        parts, models = gate_runtime(router)
        return parts, models, describe_parts(parts, models)

    parts, models, default = attrs(None)
    assert "scope_names" not in str(default)  # the description of the measured wiring is what it was
    gate_prompt(parts, models)
    assert all("scope" not in str(x.get("evidence")) for x in parts.audit.lines if x["component"] == "gate")
    parts, models, bm = attrs({"kind": "bm25"})
    _, _, hops = attrs({"kind": "bm25", "expand": "k_hops"})
    assert "bm25 and next_links" in str(bm) and "bm25 and k_hops" in str(hops)
    assert str(default) != str(bm) != str(hops)
    gate_prompt(parts, models)
    line = [x for x in parts.audit.lines if x["component"] == "gate"][-1]
    assert line["verdict"] == "allow" and line["evidence"].startswith("scope bm25 and next_links: ")
