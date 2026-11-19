"""What the agent has seen in a run (``Facts``): the text of the request and of the tool results, and the records by id."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from polisys.tool_kind import _name_tokens
from polisys.values import _walk, norm


def _is_id_key(key: str) -> bool:
    k = "_".join(_name_tokens(str(key)))
    return k == "id" or k.endswith("_id")


def _own_id(d: Dict[str, Any]) -> Optional[str]:
    for k, v in d.items():
        if "_".join(_name_tokens(str(k))) == "id" and isinstance(v, (str, int)):
            return str(v)
    for k, v in d.items():
        if _is_id_key(k) and isinstance(v, (str, int)):
            return str(v)
    return None


def _primary_id(d: Dict[str, Any]) -> Optional[str]:
    for k, v in d.items():
        if isinstance(v, str) and (k == "id" or k.endswith("_id") or k in ("number", "sys_id")):
            return v
    return None


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "d"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass
class Facts:
    """What the agent has seen: the request, tool results, records by ID, and the principal."""

    text: str = ""
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nodes: List[Dict[str, Any]] = field(default_factory=list)  # every record seen, for record_condition
    principal: Dict[str, Any] = field(default_factory=dict)

    def add_text(self, value: Any) -> None:
        self.text += " " + norm(value if isinstance(value, str) else json.dumps(value, default=str))

    def add_result(self, result: Any) -> None:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        self.add_text(result)
        for node in _walk(result):
            if isinstance(node, dict):
                pid = _primary_id(node)
                if pid:
                    self.records.setdefault(pid, {}).update(node)
                if any(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in node.values()):
                    self.nodes.append(node)

    def named(self, token: str) -> List[Dict[str, Any]]:
        """Records that a call names with ``token``: by the record's own primary id first ("id", or its first "... id"
        field such as "Unit ID"); otherwise by an exact value in a field that is not an id of another record
        (``owner_id`` = 7 does not name every record owned by 7)."""
        own = [n for n in self.nodes if _own_id(n) == token]
        if own:
            return own
        exact = [
            n
            for n in self.nodes
            if any(str(v) == token for k, v in n.items() if isinstance(v, (str, int)) and not _is_id_key(k))
        ]
        if exact:
            return exact
        return [
            n
            for n in self.nodes
            if any(
                isinstance(v, str) and token in v and len(v) < 200 and not _is_id_key(k) for k, v in n.items()
            )
        ]

    def seen(self, value: Any) -> bool:
        v = norm(value)
        return bool(v) and v in self.text
