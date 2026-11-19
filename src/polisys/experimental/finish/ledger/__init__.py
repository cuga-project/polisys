"""Obligation ledger: at the final reply, check by code what must exist (design section 5).

Write-time checks cannot see work that never happened. The ledger holds one item per obligation and case, and at the final
reply it compares the open items with the calls that succeeded. An item with no matching call goes back to the agent, with
what is known of its values.

**Where it acts.** ``before_reply`` fires after every ``agent.invoke``, so the ledger acts only at a final reply:

| Run type | Final reply |
|---|---|
| ``one_request`` | every ``before_reply``; CUGA acts on at most 2 send-backs per request, shared with other hooks |
| ``sop_loop`` | the loop's finish turn only (``loop_phase``); the loop's cases are reused |
| ``conversation`` | a hand-off: a hand-off tool call is blocked once, a hand-off in text gets a new prompt (``after_reply``) |

**Items.** ``package`` and ``data`` obligations bind at the final reply, through one model call (``bind_cases``) that lists
the cases with quoted evidence, each with its procedure, its decision branch, the answers to ``scope.when`` and the values.
``request`` obligations open at the user message, ``duty`` obligations at their trigger (``scope.trigger``: ``tools`` and
``when``, tested on the trigger call's arguments; a call before the trigger does not do the duty). The producers (Task 4.2)
make them: ``package.obligations_from_package``, ``ruleset.obligations_from_duties`` and, at the first user message through
``request_producer``, ``deliverables.obligations_from_request``. The items of a request and of duties need no case, so a failed
binding stops only the package items. An obligation with no case stays ``unbound`` and is never shown.

**Status of an item.**

| Status | Meaning | Sent |
|---|---|---|
| ``unbound`` | no case | no |
| ``open`` | no successful call matches | yes, when the levels allow it |
| ``met`` | a successful call matches | no |
| ``not_applicable`` | a ``scope.when`` answer with quoted evidence excludes it | no |
| ``held`` | a blocked call with no successful retry, or a hold of the whole case with quoted evidence | restated only |
| ``unknown`` | code cannot tell: no quoted answer, a code tool without a literal match, an inferred value that differs | no |
| ``wrong`` | a matched ``literal``, ``template`` or ``computed`` value that the call contradicts | no |

``wrong`` is recorded and not sent, because wrong items lowered the measured scores (design section 5).
``wrong_values: never`` records ``unknown`` instead.

**A follow-up never asks for blocked work** (change 64). Blocks attach by action, because a gate's ``case`` is free text and a
hard rule has none. ``tool_blocked`` is told of every call that a hook stopped, and a block stays for the whole run: it is
decided per item when the item is settled. A block that names a key of the case holds all write items of that case. A block
that names none holds an item of its tool, unless it names another case or a fixed text of the item differs. It clears for an
item only by the same call again, or by a later call of that tool that names the same keys and differs only in identifiers
(``_cleared``).

**Records are data.** The request and the data can hold text aimed at the agent. The binding prompt says so, and the fences
cannot be closed from inside. A value is one short line, a case is shown by its verified keys, and a value from a record is
labelled as such in the follow-up.

**Value tests** (design section 5, "Value binding"). A value is compared only through ``effect.match``. The test depends on
the argument: enum, boolean or number: full match. Code argument or JSON: contains, where not found is ``unknown``. ``to``,
``cc``, ``bcc`` and other arrays: one element. Any other text: contains, at word edges, so ``100`` is not in ``1000``. Case,
spaces, dashes and quotes are folded on both sides. A code call meets an item only if it holds the fragment and shows a sign
of writing. A call that does not meet an item says something about it only if it names a key of the case or passes an
identifying argument. With nothing compared (a value not known), a call gives ``unknown``.

The model sees only the package, the request and what the agent read. Nothing here names a benchmark.

**The parts of this folder**, in the order they build on each other. This file holds the component's ``Settings`` and
``build``.

- ``constants``: statuses, run types, limits, argument patterns.
- ``text``: text as it is compared and shown.
- ``value_patterns``: how an obligation writes a value.
- ``cases``: the cases of a run (``bind_cases``).
- ``values``: the values of an obligation for a case (``resolve_values``, ``plan_of``).
- ``call_check``: calls against an obligation (``check_calls``, ``match``).
- ``follow_up``: the items and the message for the agent.
- ``obligation_ledger``: ``ObligationLedger``, the hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from polisys.runtime.setting_types import Count
from polisys.runtime.context import Piece

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly


@dataclass
class Settings:
    # Empty: no ledger. Any producer turns the obligation ledger on, and then only the ledger sends follow-ups (the requirements
    # tracker is refused; the procedure monitor and the duty rules only record).
    producers: List[Literal["package", "request", "duty"]] = field(default_factory=list)
    max_follow_ups: Count = 1
    wrong_values: Literal["never", "known_origin_only"] = "known_origin_only"
    model: Optional[str] = (
        None  # model that lists the checklist of the request, and that binds the ledger's cases
    )
    # The level of every item of the ledger. None: ``build_runtime`` uses observe, so the level written in an obligation is not
    # read in a run. ``show`` is accepted and not built: only follow_up items are sent, so it acts as observe. Packages say
    # observe until a measurement chooses.
    item_level: Optional[Literal["observe", "show", "follow_up"]] = None


def build(a: Assembly) -> Piece:
    from polisys.experimental.finish.ledger.constants import CONVERSATION, ONE_REQUEST, SOP_LOOP
    from polisys.experimental.finish.ledger.obligation_ledger import ObligationLedger
    from polisys.experimental.finish.ledger.cases import cards_of
    from polisys.experimental.finish.request_checklist import obligations_from_request
    from polisys.package.files import obligations_from_package, runtime_rules
    from polisys.checks.rules import obligations_from_duties

    cfg = a.cfg.obligations
    producers = set(cfg.producers)
    if not a.ctx.tool_specs:
        raise a.fail("obligations", "the task gives no tool specs, so no argument test can be chosen")
    specs = [{k: v for k, v in t.items() if k != "app"} for t in a.ctx.tool_specs]
    if a.ctx.tools_in_run is not None:
        in_run = set(a.ctx.tools_in_run)
        specs = [t for t in specs if t["name"] in in_run]
    # The producers `package` and `duty` read the package; `request` reads the request, at the user message.
    pkg = a.package("obligations") if producers & {"package", "duty"} else a.pkg
    made: List[Any] = []
    if "package" in producers:
        made += obligations_from_package(pkg)
        if not made:
            raise a.fail("obligations", "the package has no obligations, so the ledger has nothing to check")
    if "duty" in producers:  # a policy may have no duty rule: that is no error while another producer is on
        made += obligations_from_duties(runtime_rules(pkg), specs)
    if not made and "request" not in producers:
        raise a.fail(
            "obligations",
            "the producers give no obligation (the package has no duty rule), so the ledger has nothing to check",
        )
    # The items of a request and of duties are `observe` until a measurement writes a level (Task 4.3), as the package's are.
    # With no `obligations.item_level` the level switch decides, and the ledger is on at `observe`: it records and sends
    # nothing until a replay allows it (spec section 5). An explicit `item_level` is what the profile asks for.
    effective_item = cfg.item_level or "observe"
    nav = a.cfg.organizers.navigator
    run_type = ONE_REQUEST
    if nav.handoff_tools or nav.handoff_text:
        if a.cfg.organizers.loop.mode == "on":
            raise a.fail(
                "obligations", "a run with hand-off phrases or tools and a loop has no one final reply"
            )
        if a.ctx.message_tool is None:
            raise a.fail(
                "obligations",
                "the task does not say which tool sends a message in the middle of a turn; give "
                'RunContext.message_tool, or "" when the benchmark has none',
            )
        in_run = set(
            [t["name"] for t in a.ctx.tool_specs] if a.ctx.tools_in_run is None else a.ctx.tools_in_run
        )
        missing = sorted(set(nav.handoff_tools) - in_run)
        if missing:
            raise a.fail("obligations", f"the hand-off tools {missing} are not in this run")
        run_type = CONVERSATION
    elif a.ctx.message_tool:
        raise a.fail(
            "obligations",
            "the task is a conversation, and there is no hand-off phrase and no hand-off tool, so the ledger would act on an "
            "ordinary turn; give organizers.navigator.handoff_text or handoff_tools",
        )
    elif a.cfg.organizers.loop.mode == "on":
        run_type = SOP_LOOP
    # The ledger's level limits its follow-ups (0 and 1: none, 2: one, 3: two). Under the level switch that is the run's level.
    # An explicit `item_level` keeps the earlier reading, which the profiles written for it depend on: the level of the items
    # says what the ledger may do.
    if cfg.item_level:
        level = {"observe": 0, "show": 1}.get(cfg.item_level, 3 if cfg.max_follow_ups >= 2 else 2)
    else:
        level = a.level
    policy_text = "\n".join(
        st.text
        for d in (pkg.documents if pkg is not None else [])
        for s in d.sections
        for st in s.statements
        if st.text
    )
    request_producer = None
    if "request" in producers:
        if run_type == CONVERSATION:
            raise a.fail(
                "obligations",
                "the producer request reads one request, and a conversation has none: turn it off",
            )
        ask = a.model("obligations", cfg.model)
        request_producer = lambda text: obligations_from_request(ask, text, specs)  # noqa: E731
    bindable = any(o.origin in ("package", "data") for o in made)
    ledger = ObligationLedger(
        made,
        a.model("obligations", cfg.model, timeout=300, max_tokens=16000) if bindable else None,
        cards=cards_of(made, pkg.procedures if pkg is not None else ()),
        tool_specs=a.ctx.tool_specs,
        tools_in_run=a.ctx.tools_in_run,
        run_type=run_type,
        level=level,
        max_follow_ups=cfg.max_follow_ups,
        wrong_values=cfg.wrong_values,
        item_level=effective_item,
        request=a.ctx.request,
        store=a.store("obligations") if pkg is not None else None,
        today=a.ctx.world_time[:10],
        policy_text=policy_text,
        handoff_tools=nav.handoff_tools,
        handoff_text=nav.handoff_text,
        message_tool=a.ctx.message_tool or "",
        request_producer=request_producer,
    )
    if a.loop is not None:
        a.loop._phase_listener = ledger.loop_phase
    return Piece(
        hooks=[ledger], after_reply=ledger.after_reply if run_type == CONVERSATION else None, target=ledger
    )

_PARTS = ("constants", "text", "value_patterns", "cases", "values", "call_check", "follow_up", "obligation_ledger")


def __getattr__(name: str):
    """A name of one of the parts, read on first use: ``polisys.experimental.finish.ledger`` was one module before 2026-11."""
    import importlib

    for part in _PARTS:
        try:
            return getattr(importlib.import_module(f"{__name__}.{part}"), name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
