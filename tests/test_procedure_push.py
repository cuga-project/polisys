"""organizers.procedure_push: the system picks the procedures for the data just read and attaches their text once."""

from polisys.package import from_store_graph
from polisys.procedure_push import NOTE, ProcedurePush
from polisys.procedure_store import ProcedureStore
from tests.test_runtime_assembly import GRAPH


class Router:
    def __init__(self, ids):
        self.ids, self.queries = ids, []

    def select(self, query, pkg):
        self.queries.append(query)
        return self.ids


def test_a_read_gets_the_procedure_once_and_the_gate_store_records_it():
    pkg = from_store_graph(GRAPH)
    store = ProcedureStore.from_package(pkg)
    push = ProcedurePush(store, pkg, Router(["1.1"]), request="Close the tickets.")
    note = push.after_tool_call("read_inbox", {}, "x" * 300)
    assert note.startswith(NOTE) and "Open a ticket." in note
    assert store.calls and store.calls[-1]["ids"] == ["1.1"]  # what a gate with agent_opened reads
    assert push.after_tool_call("read_inbox", {}, "y" * 300) is None  # already handed over


def test_writes_short_results_and_policy_tools_do_not_trigger_a_choice():
    pkg = from_store_graph(GRAPH)
    router = Router(["1.1"])
    push = ProcedurePush(ProcedureStore.from_package(pkg), pkg, router)
    assert push.after_tool_call("create_ticket", {}, "x" * 300) is None
    assert push.after_tool_call("read_inbox", {}, "short") is None
    assert push.after_tool_call("policy__get_procedure", {}, "x" * 300) is None
    assert router.queries == []
