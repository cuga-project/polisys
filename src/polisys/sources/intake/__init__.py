"""Online policy intake: policies found in tool output during the run (policy-flow.html, intake case 4/5).

Tier 0 (code, every tool output): policy words in any text field of a record (title, subject, body) or in the
call's arguments; later reads of a record whose ID was flagged are flagged too; seen content is skipped by hash.
Tier 1 (optional ``detector``, e.g. a decision model): scores every other text record of the output; records at or
above ``detect_threshold`` join the tier 0 candidates. Runs in the background thread, before tier 2.
Tier 2 (one model call per tool output): for each candidate record, is it a policy, is the source trusted, and the
rules as short statements. Records of one output are read together, so an update is read next to the rules it
changes. Without a detector, tiers 1 and 2 share this call.

Enforcement: before each write the pending intake is awaited (bounded), then a verifier call checks the planned
call against the trusted statements and the data: first the records that mention identifiers from the call's
arguments, then the most recent data. Untrusted statements only go into the brief. before_reply asks once, with
the data the agent read, whether a required action is still missing.

**The parts of this folder.** This file holds ``build``.

- ``prompts``: prompt texts, limits, small helpers.
- ``online_intake``: ``OnlinePolicyIntake``: settings and the state of a run.
- ``found_rules``, ``discovery``, ``write_checks``, ``message_checks``, ``reply``: the other parts of
  ``OnlinePolicyIntake``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polisys.runtime.context import Piece
from polisys._aliases import names_of

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly


def build(a: Assembly) -> Piece:
    from polisys.sources.intake.online_intake import OnlinePolicyIntake
    from polisys.checks.decisions import Decisions

    checkers, decisions = a.cfg.checkers, a.cfg.decisions
    verify = checkers.verifier.mode == "on"
    if verify and checkers.verifier.model != checkers.compile:
        raise a.fail(
            "intake",
            f"it has one model, and checkers.compile ({checkers.compile}) differs from checkers.verifier.model "
            f"({checkers.verifier.model})",
        )
    if a.cfg.sources.discovery and not a.ctx.data_search:
        raise a.fail(
            "intake", "sources.discovery is on, and the task gives no search over messages or documents"
        )
    if a.cfg.sources.listing and a.ctx.data_list is None:
        raise a.fail("intake", "sources.listing is on, and the task gives no listing of its records")
    if checkers.rules == "on" and not a.ctx.tool_specs:
        raise a.fail("intake", "checkers.rules is on, and the task gives no tool specs to compile rules for")
    detector = a.detector(checkers.detector) if checkers.detector else None
    hook = OnlinePolicyIntake(
        a.model("intake", checkers.compile),
        verify_writes=verify,
        # with a producer only the ledger sends follow-ups
        check_reply=verify and checkers.verifier.reply_check and not a.ledger_on,
        request_conditions=checkers.verifier.request_conditions,
        proven_only=checkers.verifier.proven_only,
        unasked_report=checkers.unasked_report,
        own_message_undo=checkers.own_message_undo,
        tool_docs={t["name"]: t.get("description") or "" for t in a.ctx.tool_specs or [] if t.get("name")}
        if checkers.own_message_undo != "off"
        else None,
        detector=detector,
        tool_specs=a.ctx.tool_specs if checkers.rules == "on" else None,
        method_choice=a.method_choice(),
        read_before_write=checkers.read_before_write,
        verifier_may_deny=checkers.verifier.may_deny,
        guard_code=checkers.guard_code,
        brief=a.brief(),
        note_channel=a.note_channel(),
        today=a.ctx.world_time[:10] or None,  # every check judges dates against it, not only guard code
        track_supersedes=a.cfg.authority.supersedes,
        discover=a.ctx.data_search if a.cfg.sources.discovery else None,
        listing=a.ctx.data_list if a.cfg.sources.listing else None,
        listing_data=a.cfg.sources.listing and a.cfg.sources.listing_data,
        data_map=a.cfg.sources.listing and a.cfg.sources.data_map,
        decisions=Decisions(a.questions("decisions", checkers.narrow_questions))
        if decisions.mode == "on"
        else None,
        model_effect=decisions.effect,
        cap_effect=decisions.after_repeated_denials,
        record_checks=decisions.record_checks,
        retry_effect=decisions.retry_effect,
        voice=a.cfg.followups.voice,
        evidence=a.cfg.followups.evidence,
    )
    return Piece(hooks=[hook])

_PARTS = ("prompts", "found_rules", "discovery", "write_checks", "message_checks", "reply", "online_intake")

# A name of one of the parts, read on first use: ``polisys.sources.intake`` was one module before 2026-11.
__getattr__ = names_of(__name__, *(f"{__name__}.{part}" for part in _PARTS), "polisys.checks.facts")
