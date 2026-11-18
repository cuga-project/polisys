"""Procedure push (``organizers.procedure_push``): the policy system, not the agent, finds the SOP procedures that apply to
the work in progress and hands them to the agent.

After a read returns new data, a router (``llm_cards``: a model reads the procedure cards against the request and the data
just read) picks the procedures that govern it. Each one not handed over yet is opened in the shared ``ProcedureStore``
(so its ``next`` links come along, and a gate with the scope ``agent_opened`` checks against it) and its full text is
attached to that tool result as a note. The agent can still open procedures itself when the procedure tools are on.

Nothing here names a domain: the inputs are the request, the data the agent read and the package's procedure cards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .hooks import AgentHooks
from .router import Query

NOTE = (
    "SOP procedures that govern the data just read (chosen by the policy system from the request and that data). Follow "
    "their steps, decisions, must-not rules and templates for the records they cover:\n\n"
)


class ProcedurePush(AgentHooks):
    def __init__(
        self,
        store: Any,
        pkg: Any,
        router: Any,
        request: str = "",
        min_chars: int = 200,
        max_selections: int = 15,
        context_chars: int = 6000,
    ):
        self.store, self.pkg, self.router, self.request = store, pkg, router, request
        self.min_chars, self.max_selections, self.context_chars = min_chars, max_selections, context_chars
        self.pushed: Set[str] = set()
        self.selections = 0
        self.records: List[Dict[str, Any]] = []

    def session_start(self, thread_id: Optional[str]) -> None:
        self.pushed, self.selections = set(), 0

    def on_user_message(self, message: str, thread_id: Optional[str]) -> None:
        self.request = message or self.request

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> Optional[str]:
        from .store_gate import _acts

        text = str(result)
        if (
            _acts(tool)
            or tool.startswith("policy__")
            or len(text) < self.min_chars
            or self.selections >= self.max_selections
        ):
            return None
        self.selections += 1
        try:
            ids = self.router.select(
                Query(
                    action=f"The next step of this request: {self.request[:1500]}",
                    context=f"{tool}: {text[: self.context_chars]}",
                ),
                self.pkg,
            )
        except Exception as exc:  # noqa: BLE001 - a failed choice leaves the agent as it was
            self.records.append({"component": "procedure_push", "outcome": "error", "error": str(exc)[:200]})
            return None
        new = [i for i in ids if i not in self.pushed]
        self.records.append(
            {"component": "procedure_push", "outcome": "pushed" if new else "none", "tool": tool, "ids": new}
        )
        if not new:
            return None
        text = self.store.get_procedure(new)
        self.pushed.update(new)
        self.pushed.update((self.store.calls[-1] if self.store.calls else {}).get("expanded") or [])
        return NOTE + text
