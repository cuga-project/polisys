"""Which tools read and which act: one list of read verbs for the whole policy layer.

Two questions use this list, and they differ in one detail:

- ``ruleset.is_write_tool(name)``: the first verb in the name that is known (a read verb or a write verb) decides, and a
  name with no known verb is a write (checking a read only costs a call; a missed write skips every check).
- ``acts(tool)``: the first word of the name decides. ``mark_as_read`` is an action, because its first word is not a read
  verb. The procedure store's own tools (``policy__*``) do not act.

Both read the same verbs, so a tool that one of them treats as a read is a read for the other, except for the way the name
is read. ``count``, ``average`` and ``total`` are reads (a tool named ``count_incidents`` changes nothing).
"""

from __future__ import annotations

import re
from typing import Any, Dict

# An argument that names a record: ``ticket_id``, ``jobKey``, ``order_number``. One pattern for the whole layer.
ID_ARG = re.compile(r"(?i:(^|_)(id|key|number))$|[a-z](?:Id|Key|Number)$")

READ_FIRST = {
    "get",
    "find",
    "list",
    "search",
    "lookup",
    "query",
    "retrieve",
    "count",
    "average",
    "total",
    "fetch",
    "read",
    "check",
    "view",
    "describe",
    "download",
    "export",
}


def acts(tool: str) -> bool:
    """Tools that change state: the first word of the name is not a read verb (``mark_as_read`` is an action)."""
    if tool.startswith("policy__"):
        return False
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", tool.split("__", 1)[-1]).lower())
    return bool(words) and words[0] not in READ_FIRST


_acts = acts  # the name the store gate used


def case_of(args: Dict[str, Any]) -> str:
    """The case a call is about, as code can tell it: its identifying arguments (``*_id``, ``*_key``, ``*_number``) with
    their values, in name order; empty when the call names none. A rule that stops a call once (``block_once``) stops it once
    for each case, and the gate's own ``case`` is the same idea read by a model."""
    parts = [
        f"{key}={value}"
        for key, value in sorted((args or {}).items())
        if ID_ARG.search(key) and isinstance(value, (str, int)) and not isinstance(value, bool) and str(value)
    ]
    return ",".join(parts)
