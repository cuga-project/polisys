"""Policy lines per tool, word for word.

One model call per policy text and tool set maps each write tool to the policy lines that govern a call of it. The model
only selects: code keeps a line only if it is in the policy text word for word and its tool is a write tool of the run.
The lines are added to the tool's description, under a fixed heading, with no paraphrase and no interpretation. So the
agent reads the rule in the policy's own words at the place where it chooses the call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from polisys.llm import parse_json
from polisys.experimental.guidance.tool_notes import quoted_in, tool_lines

PROMPT_VERSION = "1"
HEADER = "Policy lines for this tool (word for word from the policy):"
MAX_LINES = 8
MAX_TOOL_CHARS = 1800
SYSTEM = "You select the lines of a written policy that govern each tool. Output JSON only."
USER = """POLICY:
{policy}

TOOLS (with their arguments):
{tools}

For each tool above, select the policy lines that a call of this tool must follow: a condition before the call, a value,
default or format of one of its arguments, a step the policy requires after it.
- Copy each line word for word, whole, from the policy. Do not shorten, join or rewrite lines.
- Select a general line (one that holds for every operation) for a tool only if it names something this tool writes.
- At most {max_lines} lines per tool, the most specific first. A tool with no such line gets none.

Return {{"lines": [{{"tool": "<name>", "quote": "<policy line>"}}]}}"""


def check_lines(
    raw: Any, policy_text: str, tool_specs: List[Dict[str, Any]], is_write: Callable[[str], bool]
) -> Dict[str, List[str]]:
    """Per write tool of the run, the selected lines that are in the policy text word for word (at most ``MAX_LINES``)."""
    names = {s["name"] for s in tool_specs if is_write(s["name"])}
    out: Dict[str, List[str]] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tool, quote = str(item.get("tool") or ""), " ".join(str(item.get("quote") or "").split())
        if tool not in names or not quoted_in(quote, policy_text):
            continue
        kept = out.setdefault(tool, [])
        if quote not in kept and len(kept) < MAX_LINES:
            kept.append(quote)
    return out


def build_lines(
    policy_text: str,
    tool_specs: List[Dict[str, Any]],
    model: Callable[[str, str], str],
    is_write: Callable[[str], bool],
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """``{"lines": {tool: [line]}, "cached": bool}``. With a cache folder, parallel runs share one answer."""
    write_specs = [s for s in tool_specs if is_write(s["name"])]

    def make() -> Dict[str, Any]:
        user = USER.format(policy=policy_text, tools=tool_lines(write_specs), max_lines=MAX_LINES)
        for _ in range(2):  # an answer with no list (cut, or not JSON) is asked once more, then fails
            reply = parse_json(model(SYSTEM, user))
            if isinstance(reply.get("lines"), list):
                return {"lines": check_lines(reply["lines"], policy_text, tool_specs, is_write), "cached": False}
        raise ValueError("the model's answer holds no list of lines")

    if not cache_dir:
        return make()
    import fcntl

    folder = Path(cache_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        [PROMPT_VERSION, policy_text, sorted(s["name"] for s in write_specs), getattr(model, "model_name", "")],
        ensure_ascii=False,
    )
    path = folder / f"lines-{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:24]}.json"
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists():
                return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
            data = make()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
            return data
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def describe(lines: Dict[str, List[str]]) -> Dict[str, str]:
    out = {}
    for tool, quotes in lines.items():
        text = "\n".join([HEADER] + [f'- "{q}"' for q in quotes])
        out[tool] = text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS].rsplit("\n", 1)[0]
    return out


class PolicyLines:
    """The lines of one run. ``wrap_tools`` gives the agent its tools with the lines in their descriptions; the lines are
    read at the first ``wrap_tools`` call (from the cache, or from the model), not when the run is wired."""

    def __init__(self, load: Callable[[], Dict[str, Any]]) -> None:
        self._load = load
        self.text: Optional[Dict[str, str]] = None
        self.records: List[Dict[str, Any]] = []

    def wrap_tools(self, tools: List[Any], turns: Any = None, live: Any = None) -> List[Any]:
        if self.text is None:
            try:
                data = self._load()
            except Exception as exc:  # the run goes on without the lines, and the audit says so
                data = {"lines": {}}
                self.records.append(
                    {"component": "policy_lines", "outcome": "lines_failed", "reason": f"{type(exc).__name__}: {exc}"[:200]}
                )
            self.text = describe(data.get("lines") or {})
            self.records.append(
                {
                    "component": "policy_lines",
                    "outcome": "lines_shown",
                    "tools": len(self.text),
                    "lines": sum(len(v) for v in (data.get("lines") or {}).values()),
                    "cached": bool(data.get("cached")),
                }
            )
        out = []
        for tool in tools:
            note = self.text.get(getattr(tool, "name", ""))
            out.append(tool.model_copy(update={"description": f"{tool.description or ''}\n\n{note}".strip()}) if note else tool)
        return out
