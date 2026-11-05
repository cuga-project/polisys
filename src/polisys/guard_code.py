"""Generated guard code: a policy statement becomes a small Python check for one write tool.

The model is good at reading a rule and bad at applying it to many records (dates, sums, thresholds, per-customer
values). So the model writes the check once per (statement, tool), and code runs it on every planned call with the
data the agent has read so far. This follows ToolGuard (policies compiled into guard functions), at run time.

    def check(call, data, today):
        # call  = {"tool": str, "args": dict}
        # data  = [{"tool": str, "args": dict, "records": [dict, ...]}, ...]   every read output so far, oldest first
        # today = "YYYY-MM-DD"
        return ["reason", ...]   # [] when the call complies, or the statement does not decide it

Safety: the code is checked before it runs (no imports, no names or attributes that reach the system, files or
interpreter internals), runs with a small set of builtins, and stops after a fixed number of steps.
"""

from __future__ import annotations

import ast
import builtins
import datetime as _dt
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_STEPS = 200_000
ALLOWED_MODULES = {"datetime", "re", "math"}


def _import(name, globals=None, locals=None, fromlist=(), level=0):  # only the modules the checks may use
    if name not in ALLOWED_MODULES:
        raise ImportError(f"module {name} is not allowed in a guard check")
    return __import__(name, globals, locals, fromlist, level)


FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
    "type",
    "object",
    "super",
    "classmethod",
    "staticmethod",
    "property",
    "dir",
    "id",
    "hasattr",
    "__builtins__",
    "os",
    "sys",
    "subprocess",
    "importlib",
    "builtins",
}
SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in (
        "len",
        "min",
        "max",
        "sum",
        "sorted",
        "any",
        "all",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "enumerate",
        "zip",
        "range",
        "isinstance",
        "map",
        "filter",
        "reversed",
        "ValueError",
        "TypeError",
        "KeyError",
        "Exception",
        "print",
    )
}

SYSTEM = """You turn one company policy statement into a Python check for one planned tool call.
Write exactly this function (no imports; `datetime`, `re`, `math` and the helpers below are already available):

def check(call, data, today):
    # call  = {"tool": str, "args": dict}          the planned call
    # data  = [{"tool": str, "args": dict, "records": [dict, ...]}, ...]   every read output so far, oldest first
    # today = "YYYY-MM-DD"
    # return a list of short reasons, one per violation; return [] if the call complies or the statement
    # does not decide this call

Helpers: num(x) -> float or None ("$24,000" -> 24000.0); date(x) -> datetime.date or None (ISO or common formats);
norm(x) -> lower-case stripped str; mentions(text, value) -> True if value appears in text as a whole word or phrase
(case-insensitive); records(data, tool=None) -> all records, optionally only from one tool.

Rules:
- Decide from the call's arguments and the records only. If a record you need is missing, return [].
- A call can act on records by ID or by name inside free text (an email body, a sheet row). Find which records the
  call acts on, then test the statement's condition on each of them.
- Report a violation only when the data shows it clearly. Keep reasons under 25 words and name the record.
- If this statement says nothing about this tool or this kind of action, answer applies=false.
Answer JSON only: {"applies": true, "code": "def check(call, data, today):\\n    ..."} or {"applies": false}."""

USER = """Policy statement: {statement}

Planned tool: {tool}({args})

Data read so far (record keys and samples per read output):
{schema}"""


def num(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(x or ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def date(x: Any) -> Optional[_dt.date]:
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    s = str(x or "").strip()
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        try:
            return _dt.date.fromisoformat(m.group(0))
        except ValueError:
            return None
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def mentions(text: Any, value: Any) -> bool:
    v = str(value or "").strip()
    if not v:
        return False
    return re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", str(text or ""), re.IGNORECASE) is not None


def records(data: List[Dict[str, Any]], tool: Optional[str] = None) -> List[Dict[str, Any]]:
    return [r for d in data if tool is None or d.get("tool") == tool for r in d.get("records", [])]


def flatten_records(result: Any) -> List[Dict[str, Any]]:
    """The records in a tool output: the dicts in its lists (rows, items), with nested ``cells`` merged in."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return []
    out: List[Dict[str, Any]] = []

    def flat(d: Dict[str, Any]) -> Dict[str, Any]:
        rec = {k: v for k, v in d.items() if not isinstance(v, (dict, list)) or k in ("tags", "labels")}
        for k in ("cells", "fields", "properties", "data"):
            if isinstance(d.get(k), dict):
                rec.update(d[k])
        return rec

    def walk(x: Any, top: bool) -> None:
        if isinstance(x, list):
            for i in x:
                if isinstance(i, dict):
                    out.append(flat(i))
                walk(i, False)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v, False)

    walk(result, True)
    if not out and isinstance(result, dict):
        out.append(flat(result))
    return out


def schema(data: List[Dict[str, Any]], chars: int = 6000) -> str:
    lines = []
    for i, d in enumerate(data):
        recs = d.get("records") or []
        keys = sorted({k for r in recs for k in r})[:30]
        sample = json.dumps(recs[:2], default=str)[:600]
        lines.append(
            f"[{i}] {d['tool']}({json.dumps(d.get('args'), default=str)[:150]}): {len(recs)} records; "
            f"keys {keys}; sample {sample}"
        )
    text = "\n".join(lines)
    return text[-chars:]


def is_safe(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax: {exc}"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if not funcs:
        return False, "no check() function"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and all(a.name in ALLOWED_MODULES for a in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module in ALLOWED_MODULES:
            continue
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.ClassDef,
            ),
        ):
            return False, f"not allowed: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"not allowed: attribute {node.attr}"
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            return False, f"not allowed: name {node.id}"
    return True, ""


class _StepLimit(Exception):
    pass


def run_check(fn: Callable, call: Dict[str, Any], data: List[Dict[str, Any]], today: str) -> List[str]:
    steps = [0]

    def tracer(frame, event, arg):  # stop runaway loops
        steps[0] += 1
        if steps[0] > MAX_STEPS:
            raise _StepLimit()
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn(call, data, today)
    finally:
        sys.settrace(old)
    return [str(x)[:300] for x in (out or []) if x] if isinstance(out, (list, tuple)) else []


def compile_check(code: str) -> Callable:
    ok, why = is_safe(code)
    if not ok:
        raise ValueError(why)
    env: Dict[str, Any] = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _import},
        "datetime": _dt,
        "re": re,
        "math": math,
        "num": num,
        "date": date,
        "norm": norm,
        "mentions": mentions,
        "records": records,
    }
    exec(compile(code, "<guard>", "exec"), env)  # noqa: S102 - checked by is_safe, restricted builtins
    return env["check"]


class GuardCode:
    """Compiles and runs guard checks. ``model(system, user) -> str`` is a small JSON model (e.g. gpt-oss-120b)."""

    def __init__(self, model: Callable[[str, str], str], pool: Optional[ThreadPoolExecutor] = None):
        self.model = model
        self.pool = pool or ThreadPoolExecutor(max_workers=6)
        self.cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        self.log: List[Dict[str, Any]] = []

    def _key(self, statement: str, tool: str) -> Tuple[str, str]:
        return hashlib.sha256(statement.encode()).hexdigest()[:16], tool

    def _compile_one(self, statement: str, tool: str, args: Dict[str, Any], data: List[Dict[str, Any]]):
        from .llm import parse_json

        raw = self.model(
            SYSTEM,
            USER.format(
                statement=statement, tool=tool, args=json.dumps(args, default=str)[:1500], schema=schema(data)
            ),
        )
        reply = parse_json(raw) or {}
        if not reply.get("applies") or not reply.get("code"):
            return None
        try:
            fn = compile_check(reply["code"])
        except Exception as exc:  # noqa: BLE001 - a bad check is dropped, the statement stays with the verifier
            self.log.append({"statement": statement[:120], "tool": tool, "error": f"compile: {exc}"[:300]})
            return None
        return {"fn": fn, "code": reply["code"]}

    def checks_for(
        self, statements: List[Dict[str, Any]], tool: str, args: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Compile (once per statement and tool, in parallel) and return {statement n: check} for those that apply."""
        todo = [s for s in statements if self._key(s["text"], tool) not in self.cache]
        futs = {s["n"]: self.pool.submit(self._compile_one, s["text"], tool, args, data) for s in todo}
        for s in todo:
            try:
                self.cache[self._key(s["text"], tool)] = futs[s["n"]].result(timeout=90)
            except Exception as exc:  # noqa: BLE001 - model or timeout errors: no check
                self.cache[self._key(s["text"], tool)] = None
                self.log.append({"statement": s["text"][:120], "tool": tool, "error": f"model: {exc}"[:300]})
        return {
            s["n"]: self.cache[self._key(s["text"], tool)]
            for s in statements
            if self.cache.get(self._key(s["text"], tool))
        }

    def evaluate(
        self, check: Dict[str, Any], tool: str, args: Dict[str, Any], data: List[Dict[str, Any]], today: str
    ) -> Optional[List[str]]:
        """Violations, or None when the check fails to run (no decision)."""
        try:
            return run_check(check["fn"], {"tool": tool, "args": args}, data, today)
        except Exception as exc:  # noqa: BLE001
            self.log.append({"tool": tool, "error": f"run: {type(exc).__name__}: {exc}"[:300]})
            return None
