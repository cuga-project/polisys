"""Read first (``notes.read_first``): before the agent's first message to people (a send or a post), remind it of the
sources the request names that it has not read yet. A reminder in the system prompt of each model call, never a block.

A sent message cannot be taken back. Traces showed reports sent before a source the request named (a chat channel, a
mailbox) was read; the correction in that source then came too late. The check is exact: the apps the request names by
name, and the ``#channel`` names it gives, against the read calls the agent made. It stops after the first send.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set

from .hooks import AgentHooks, HookResult

SENDS = re.compile(r"(^|_)(send|post|reply|forward)(_|$)", re.I)  # tool-name words of a message to people
CHANNEL = re.compile(r"(?<![\w#])#([a-z0-9][a-z0-9_-]{1,79})\b", re.I)

NOTE = (
    "Before your first message to people (an email, a chat post or a reply), read every source the request names. "
    "Not read yet: {missing}. A message cannot be taken back, and a correction in one of these sources would come too late."
)


def app_names(app: str) -> List[str]:
    """How a request may name an app: its words ("google sheets") and, for a two-word id, its last word ("sheets")."""
    words = [w for w in re.split(r"[_\s-]+", app.lower()) if w]
    names = [" ".join(words)] if words else []
    if len(words) == 2:
        names.append(words[1])
    return names


class ReadFirst(AgentHooks):
    def __init__(self, tool_apps: Dict[str, str], is_write: Callable[[str], bool], request: str = ""):
        self.tool_apps, self.is_write = tool_apps, is_write
        self.apps = sorted({a for a in tool_apps.values() if a})
        self.records: List[Dict[str, Any]] = []
        self._start(request)

    def _start(self, request: str) -> None:
        self.request = request or ""
        self.read_apps: Set[str] = set()
        self.read_text: List[str] = []
        self.channel_ids: Dict[str, Set[str]] = {}
        self.sent = False
        self.named_apps = self._named_apps(self.request)
        self.named_channels = sorted({m.lower() for m in CHANNEL.findall(self.request)})

    def _named_apps(self, text: str) -> List[str]:
        low = text.lower()
        return [a for a in self.apps if any(re.search(rf"\b{re.escape(n)}\b", low) for n in app_names(a))]

    def session_start(self, thread_id: Optional[str]) -> None:
        self._start("")

    def on_user_message(self, message: str, thread_id: Optional[str]) -> None:
        self._start(message)

    def missing(self) -> List[str]:
        """Named apps with no read call, and named channels that no read call asked for by name or id."""
        out = [a for a in self.named_apps if a not in self.read_apps]
        asked = " ".join(self.read_text).lower()
        for c in self.named_channels:
            keys = {c} | {i.lower() for i in self.channel_ids.get(c, set())}
            if not any(re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", asked) for k in keys):
                out.append(f"#{c}")
        return out

    def _learn_ids(self, result: Any) -> None:
        """A listing of channels gives each named channel's id, so a later read by id counts."""
        if isinstance(result, dict):
            name = str(result.get("name") or "").lower().lstrip("#")
            if name in self.named_channels and result.get("id"):
                self.channel_ids.setdefault(name, set()).add(str(result["id"]))
            for v in result.values():
                self._learn_ids(v)
        elif isinstance(result, list):
            for v in result:
                self._learn_ids(v)

    def before_llm_call(self, thread_id: Optional[str]) -> Optional[HookResult]:
        if self.sent:
            return None
        missing = self.missing()
        return HookResult(message=NOTE.format(missing=", ".join(missing))) if missing else None

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> None:
        if self.is_write(tool):
            if SENDS.search(tool) and not self.sent:
                self.sent = True
                self.records.append(
                    {"component": "read_first", "outcome": "first_send", "tool": tool, "unread": self.missing()}
                )
            return
        app = self.tool_apps.get(tool)
        if app:
            self.read_apps.add(app)
        self.read_text.append(json.dumps(args, default=str))
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                return
        self._learn_ids(result)
