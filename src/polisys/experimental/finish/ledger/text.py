"""Text as the ledger compares and shows it: folded text, the spans of evidence that quote the data, one-line and fenced
text, and a call in readable form with secrets removed."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from polisys.checks.rules import canon
from polisys.finish.requirements_tracker import _failed
from polisys.experimental.finish.ledger.constants import (
    MAX_VALUE_CHARS, MIN_EVIDENCE_CHARS, MIN_KEY_CHARS, REDACTED, SECRET_ARG, _QUOTES,
)


def fold(value: Any) -> str:
    """Text as it is compared: dashes and quotes folded, lower case, one space between words. No value is no text."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", canon(value).translate(_QUOTES).lower()).strip()


def _fold_part(text: str) -> str:
    """``fold`` for a part of a pattern: a space at its edge is kept, because the part continues."""
    return re.sub(r"\s+", " ", canon(text).translate(_QUOTES).lower())


def _as_text(value: Any) -> str:
    """A value of a call or of a package as text: booleans and numbers compare as text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    return str(value)


def _unescape(text: str) -> str:
    """Code and JSON text as a reader sees it: ``\\u2013`` is a dash and ``\\"`` a quote."""
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    return text.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ").replace("\\t", " ")


def _words(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", fold(_unescape(str(text or ""))))


def _hay(text: Any) -> str:
    return f" {' '.join(_words(text))} "


def quoted_spans(
    evidence: Any, data: str, policy: str = "", min_chars: int = MIN_EVIDENCE_CHARS
) -> List[str]:
    """The spans of the evidence that quote the data: the whole, or a part between quotes or ``...``, word by word. ``data``
    and ``policy`` are in the form of ``_hay``. A span that the policy text also holds is a rule, not evidence about the case
    (the agent may have read the SOP). Each span is given as its words."""
    text = str(evidence or "")
    spans = [text] + re.findall(r"[\"\u201c]([^\"\u201c\u201d]+)[\"\u201d]", text)
    spans += [p for p in re.split(r"\.\.\.|\u2026|[;|]", text) if p.strip()]
    out = []
    for span in spans:
        joined = " ".join(_words(span))
        if len(joined) >= min_chars and f" {joined} " in data and not (policy and f" {joined} " in policy):
            out.append(joined)
    return out


def is_quote(evidence: Any, data: str, policy: str = "", min_chars: int = MIN_EVIDENCE_CHARS) -> bool:
    return bool(quoted_spans(evidence, data, policy, min_chars))


def _one_line(text: Any, limit: int = MAX_VALUE_CHARS) -> str:
    """Text of a record as it may be shown to the agent: one short line, with no quote to close and no fence."""
    line = (
        re.sub(r"\s+", " ", str(text or "")).replace('"', "'").replace("<<<", "").replace(">>>", "").strip()
    )
    return line if len(line) <= limit else line[:limit] + "..."


def _fence(text: str) -> str:
    """Text of a record inside the prompt's fences: a record cannot close them."""
    return text.replace(">>>", "> > >").replace("<<<", "< < <")


def _names(keys: Iterable[str], text: str) -> bool:
    """Does folded text name one of these keys, as whole words?"""
    for key in keys:
        folded = fold(key)
        if len(folded) >= MIN_KEY_CHARS and re.search(rf"(?<![a-z0-9]){re.escape(folded)}(?![a-z0-9])", text):
            return True
    return False


def _call_text(args: Any) -> str:
    return fold(_unescape(_as_text(args)))


def _readable_call(args: Any) -> Tuple[str, Dict[str, Any]]:
    """The text of a call and the arguments that can be read. For a call that ``_call_text`` can read, its text and all its
    arguments. A value it cannot read (a dict with keys of two types, a dict that holds itself) keeps its name and its plain
    text in the text of the call, and is not among the arguments."""
    try:
        return _call_text(args), args
    except Exception:  # noqa: BLE001 - what is known about the call still counts
        pass
    parts: List[str] = []
    kept: Dict[str, Any] = {}
    for key, value in (args or {}).items():
        try:
            parts.append(_call_text({key: value}))
            kept[key] = value
        except Exception:  # noqa: BLE001
            try:
                parts.append(fold(_unescape(f"{key} {value}")))
            except Exception:  # noqa: BLE001
                parts.append(fold(str(key)))
    return " ".join(parts), kept


def _value_text(value: Any) -> str:
    try:
        return fold(_as_text(value))
    except Exception:  # noqa: BLE001 - a value that cannot be read equals no value that can
        return ""


def _without_secrets(args: Any) -> Any:
    """The arguments of a call as they may be written down: a value under a secret's name is gone, at any depth."""
    if isinstance(args, dict):
        return {k: REDACTED if SECRET_ARG.search(str(k)) else _without_secrets(v) for k, v in args.items()}
    if isinstance(args, (list, tuple)):
        return [_without_secrets(v) for v in args]
    return args


def _failed_call(result: Any) -> bool:
    """The call did not do its work. A code tool reports this by a non-zero return code, as a value and not as an error."""
    if _failed(result):
        return True
    value = result
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    if not isinstance(value, dict):
        return False
    code = value.get("returncode", value.get("exit_code"))
    return isinstance(code, int) and code != 0  # True is not 0
