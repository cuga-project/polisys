"""SOP navigator: before the agent hands a case off, list the SOP checks for that case it never did.

Support agents stop too early: they transfer to a human (or give up) while steps the SOP prescribes for the case
are still open. The missed steps are often in another part of the SOP that the case's section only refers to
("ensure mobile data works ... refer to the Mobile Data section"), so an agent reading the case's section does not
see them. The navigator walks the SOP as a graph:

1. the SOP's section tree comes from its headings; the intake build gives the ``refers_to`` links between
   sections, the statements of each section and the escalation conditions (``escalations`` of its procedures);
2. the case's chapter (a top-level section) is chosen from the customer's first messages by one yes/no question
   per chapter to a decision model;
3. the walk is the chapter's sections plus the sections they refer to (``depth`` hops);
4. an action statement is a statement that uses a tool: it names one (``tool()``) or contains every word of an
   agent tool's name (``enable_roaming`` for "check that the line is roaming enabled");
5. at a hand-off, when the customer has not confirmed the problem is fixed and no escalation condition of the walk
   holds, the action statements with no evidence in the conversation (their tool was not called and not named by
   the agent) are returned as a note, in walk order.

The caller decides the effect (for example: block the first transfer and show the note). Nothing here names a
domain: sections, links, tools and conditions all come from the SOP, the build and the tool list.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .trace import emit

YES = 0.5  # an answer of the decision model at or above this counts as yes; the audit reads the same line


def holds(p: Optional[float]) -> bool:
    """The decision model answered, and the answer is yes. No answer (None) is not a yes."""
    return p is not None and p >= YES


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
CALLED = re.compile(r"\b([a-z][a-z0-9_]{2,})\(\)")
TOPIC_Q = "Is the customer's problem about this topic: {title}?"
SOLVED_Q = (
    "Has the customer confirmed that their original problem is now fixed (for example, the thing they reported "
    "works again)?"
)
EXIT_Q = "Does the conversation show that {when}?"
DONE_Q = (
    "Is this SOP step completed in this conversation? A check is completed when its result is known; a fix is "
    "completed only when it was applied, or the conversation shows it is not needed. SOP step: {step}"
)
DONE_BATCH = 8
MAX_ITEMS = 5
MIN_CHAPTER_ACTIONS = 3


def _sections(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stack: List[int] = []
    for line in text.split("\n"):
        m = HEADING.match(line)
        if m:
            lv = len(m.group(1))
            while stack and out[stack[-1]]["level"] >= lv:
                stack.pop()
            out.append({"level": lv, "title": m.group(2).strip(), "parent": stack[-1] if stack else None})
            stack.append(len(out) - 1)
    for i, s in enumerate(out):
        s["children"] = [j for j, t in enumerate(out) if t["parent"] == i]
    return out


class SopNavigator:
    def __init__(
        self,
        policy_text: str,
        build: Dict[str, Any],
        agent_tools: Iterable[str],
        ask: Callable[[str, Dict[str, str]], Dict[str, Any]],
        depth: int = 1,
        max_items: int = MAX_ITEMS,
        min_chapter_actions: int = MIN_CHAPTER_ACTIONS,
        semantic_evidence: bool = True,
    ):
        self.secs = _sections(policy_text)
        docs = (build.get("policy_set") or {}).get("documents") or []
        bsecs = [s for d in docs for s in d.get("sections") or []]
        same = sum(b.get("title", "").strip() == s["title"] for b, s in zip(bsecs, self.secs))
        if len(bsecs) != len(self.secs) or same < 0.9 * len(self.secs):  # the build may rename a preamble
            raise ValueError("the build's sections do not match the policy headings")
        self.ids = [s["id"] for s in bsecs]
        self.agent_tools = [t for t in agent_tools if not t.startswith("get_")]
        self.ask, self.depth, self.max_items = ask, depth, max_items
        self.semantic_evidence = semantic_evidence
        self.items: Dict[int, List[Tuple[str, str, set]]] = defaultdict(list)
        for i, s in enumerate(bsecs):
            for st in s.get("statements") or []:
                ts = self.tools_in(st.get("text") or "")
                if ts:
                    self.items[i].append((st["id"], st["text"], ts))
        pos = {sid: i for i, sid in enumerate(self.ids)}
        self.links: Dict[int, set] = defaultdict(set)
        inner = (build.get("build") or {}).get("links") or []
        for lk in inner:
            if lk.get("type") != "refers_to":
                continue
            a = pos.get(lk["src"].split(":")[1]) if lk["src"].count(":") >= 1 else None
            b = pos.get(lk["dst"].split(":", 1)[1]) if ":" in lk["dst"] else None
            if a is not None and b is not None and a != b:
                self.links[a].add(b)
        self.exits = [
            (
                pos[e["from_section"]],
                re.sub(r"^(if|when)\s+", "", str(e.get("when") or "").strip(), flags=re.I),
            )
            for p in (build.get("build") or {}).get("procedures") or []
            for e in p.get("escalations") or []
            if e.get("from_section") in pos and e.get("when")
        ]
        self.chapters = [
            i
            for i, s in enumerate(self.secs)
            if s["parent"] is None
            and sum(len(self.items.get(j, [])) for j in self._subtree(i)) >= min_chapter_actions
        ]
        self.chapter: Optional[int] = None
        self.records: List[Dict[str, Any]] = []

    @classmethod
    def from_package(
        cls,
        pkg: Any,
        policy_text: str,
        agent_tools: Iterable[str],
        ask: Callable[[str, Dict[str, str]], Dict[str, Any]],
        **kwargs: Any,
    ) -> "SopNavigator":
        """A navigator over the sections, links and escalations of a ``PolicyPackage``. ``policy_text``: the SOP text as
        for the constructor, because a package does not keep the heading levels."""
        from .model import to_json
        from .package import procedure_dicts

        build = {
            "policy_set": {"documents": to_json(pkg.documents)},
            "build": {"links": to_json(pkg.links), "procedures": procedure_dicts(pkg)},
        }
        return cls(policy_text, build, agent_tools, ask, **kwargs)

    def tools_in(self, text: str) -> set:
        named = set(CALLED.findall(text)) | {
            t for t in self.agent_tools if re.search(r"\b" + t + r"\b", text)
        }
        pre = {w[:5] for w in re.findall(r"[a-z]{3,}", text.lower())}
        return named | {
            t for t in self.agent_tools if "_" in t and all(w[:5] in pre for w in t.split("_") if w)
        }

    def _subtree(self, i: int) -> List[int]:
        out = [i]
        for c in self.secs[i]["children"]:
            out += self._subtree(c)
        return out

    def walk(self, root: int) -> List[int]:
        seen: List[int] = []
        frontier = [root]
        for _ in range(self.depth + 1):
            nxt: List[int] = []
            for r in frontier:
                for i in self._subtree(r):
                    if i not in seen:
                        seen.append(i)
                        nxt += [j for j in self.links.get(i, ()) if j not in seen]
            frontier = nxt
        return seen

    def choose_chapter(self, customer_text: str) -> Optional[int]:
        if not self.chapters:
            return None
        p = self.ask(
            f"Customer messages:\n{customer_text}",
            {f"c{i}": TOPIC_Q.format(title=self.secs[i]["title"]) for i in self.chapters},
        )
        best = max(self.chapters, key=lambda i: p.get(f"c{i}") if p.get(f"c{i}") is not None else -1.0)
        return best if p.get(f"c{best}") is not None else None

    def check_handoff(self, turns: List[Dict[str, Any]], call_args: Any = None) -> Optional[str]:
        """The note for a hand-off, or None when the hand-off is fine (solved, an exit holds, nothing open)."""
        customer = [t["text"] for t in turns if t.get("role") == "customer" and t.get("text")]
        if self.chapter is None:
            self.chapter = self.choose_chapter("\n".join(customer[:2]))
        where = getattr(self, "_where", None)  # set by the guard: which hook, which tool, for the audit
        if self.chapter is None:
            emit(  # the hand-off goes through unchecked: no chapter, so no step is known to be open
                self,
                {
                    "hook": where[0] if where else "",
                    "tool": where[1] if where else "",
                    "rule": "no_chapter",
                    "outcome": "allow",
                    "reason": "the decision model chose no chapter for this conversation, so the hand-off was not checked",
                },
            )
            return None
        rec: Dict[str, Any] = {"chapter": self.secs[self.chapter]["title"]}
        if where:
            rec["hook"], rec["tool"] = where
        solved = self.ask("Last customer messages:\n" + "\n".join(customer[-2:]), {"solved": SOLVED_Q}).get(
            "solved"
        )
        rec["solved"] = solved
        if holds(solved):
            self.records.append(rec)
            return None
        walked = self.walk(self.chapter)
        conv = self._dialogue(turns)
        if call_args is not None:
            conv += "\nagent asks to hand off: " + json.dumps(call_args, default=str)[:800]
        exits = {f"e{k}": EXIT_Q.format(when=w) for k, (i, w) in enumerate(self.exits) if i in walked}
        if exits:
            ep = self.ask("Conversation (end):\n" + conv[-4000:], exits)
            rec["exits"] = ep
            if any(holds(v) for v in ep.values()):
                self.records.append(rec)
                return None
        called = {t.get("name") for t in turns if t.get("role") == "tool"}
        said = "\n".join(t.get("text") or "" for t in turns if t.get("role") == "agent")
        pending = [
            it
            for i in walked
            for it in self.items.get(i, [])
            if not any(x in called or re.search(r"\b" + re.escape(x) + r"\b", said) for x in it[2])
        ]
        if (
            pending and self.semantic_evidence
        ):  # steps done in words (no tool named): ask about each open step
            done = set()
            for k in range(0, len(pending), DONE_BATCH):
                batch = pending[k : k + DONE_BATCH]
                ans = self.ask(
                    "Conversation:\n" + conv[-6000:],
                    {f"d{j}": DONE_Q.format(step=it[1]) for j, it in enumerate(batch)},
                )
                done |= {it[0] for j, it in enumerate(batch) if holds(ans.get(f"d{j}"))}
            rec["done_in_words"] = sorted(done)
            pending = [it for it in pending if it[0] not in done]
        rec["pending"] = [it[0] for it in pending]
        self.records.append(rec)
        if not pending:
            return None
        lines = [f"- {it[1]}" for it in pending[: self.max_items]]
        return (
            "Before handing this case off: the SOP sections for this issue have checks or fixes that this "
            "conversation has not done yet:\n"
            + "\n".join(lines)
            + "\nDo these with the customer first. If the "
            "customer's problem is fixed, or an SOP escalation condition is met, you may hand off."
        )

    @staticmethod
    def _dialogue(turns: List[Dict[str, Any]]) -> str:
        out = []
        for t in turns:
            if t.get("role") == "tool":
                out.append(f"tool {t.get('name')}: {str(t.get('result') or '')[:300]}")
            elif t.get("text"):
                out.append(f"{t['role']}: {t['text'][:600]}")
        return "\n".join(out)


class HandoffGuard:
    """Where a navigator meets a conversation: the hand-off tools and the replies that hand the customer off.

    A premature hand-off is blocked at most ``max_blocks`` times over the conversation; the agent gets the navigator's
    note instead. The runner keeps two lists and passes them in: ``turns``, the conversation so far (customer, agent and
    tool entries, as ``check_handoff`` reads them), and ``live``, what happened in the current turn. ``wrap_tools`` fills
    ``live``, so that a check in the middle of a turn sees the calls of that turn.
    """

    def __init__(
        self,
        navigator: SopNavigator,
        handoff_text: Iterable[str] = (),
        handoff_tools: Iterable[str] = (),
        max_blocks: int = 2,
        message_tool: str = "",
    ):
        """``message_tool``: the tool that sends the customer a message in the middle of a turn and returns the reply."""
        self.navigator = navigator
        self.handoff_text = [m.strip().lower() for m in handoff_text if m.strip()]
        self.handoff_tools = sorted(set(handoff_tools))
        self.max_blocks = max_blocks
        self.message_tool = message_tool
        self.blocks = 0

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self.navigator.records

    def wiring(self) -> Dict[str, Any]:
        return {
            "navigator": self.navigator,
            "handoff_text": self.handoff_text,
            "handoff_tools": self.handoff_tools,
            "max_blocks": self.max_blocks,
            "message_tool": self.message_tool,
        }

    def _forwarded(self, hook: str, tool: str) -> None:
        """A hand-off goes through unchecked because the limit of blocks is reached. No record holds this decision."""
        emit(
            self,
            {
                "hook": hook,
                "tool": tool,
                "rule": "max_blocks",
                "outcome": "allow_after_cap",
                "reason": f"{self.blocks} hand-offs were blocked, and the limit is {self.max_blocks}",
                "checker": "code",
            },
        )

    def _is_handoff_text(self, text: str) -> bool:
        low = (text or "").lower()
        return any(mark in low for mark in self.handoff_text)

    def wrap_tools(
        self, tools: List[Any], turns: List[Dict[str, Any]], live: List[Dict[str, Any]]
    ) -> List[Any]:
        """The tools to give the agent. Every call is added to ``live`` as it happens; a hand-off call that the navigator
        finds premature gets the navigator's note as its result and is not forwarded. Tools with no coroutine pass as
        they are."""
        import asyncio

        from langchain_core.tools import StructuredTool

        def wrap(tool: Any) -> Any:
            orig = tool.coroutine

            async def guarded(*args: Any, **kwargs: Any) -> Any:
                if tool.name in self.handoff_tools and self.blocks >= self.max_blocks:
                    self._forwarded("before_tool_call", tool.name)
                if tool.name in self.handoff_tools and self.blocks < self.max_blocks:
                    loop = asyncio.get_running_loop()
                    self.navigator._where = ("before_tool_call", tool.name)
                    note = await loop.run_in_executor(
                        None, self.navigator.check_handoff, list(turns) + list(live), kwargs
                    )
                    if note:
                        self.blocks += 1
                        return note
                result = await orig(*args, **kwargs)
                if tool.name == self.message_tool:
                    live.extend(
                        [
                            {
                                "role": "agent",
                                "text": str(kwargs.get("message") or (args[0] if args else "")),
                            },
                            {"role": "customer", "text": str(result or "")},
                        ]
                    )
                else:
                    live.append(
                        {"role": "tool", "name": tool.name, "args": kwargs, "result": str(result)[:300]}
                    )
                return result

            return StructuredTool(
                name=tool.name, description=tool.description, args_schema=tool.args_schema, coroutine=guarded
            )

        return [wrap(t) if getattr(t, "coroutine", None) else t for t in tools]

    def after_reply(
        self, answer: str, turns: List[Dict[str, Any]], live: List[Dict[str, Any]]
    ) -> Optional[str]:
        """The new prompt for the agent when the reply is a premature hand-off, or ``None`` to send the reply."""
        if not self._is_handoff_text(answer):
            return None
        if self.blocks >= self.max_blocks:
            self._forwarded("before_reply", "")
            return None
        self.navigator._where = ("before_reply", "")
        note = self.navigator.check_handoff(list(turns) + list(live), {"message": answer})
        if not note:
            return None
        self.blocks += 1
        turns.extend(
            t for t in live if t["role"] != "tool"
        )  # the runner has already added the tool calls to turns
        return f"[Policy note — your hand-off message was not sent]\n{note}"
