"""Read every record at the start (``sources.listing``): the host lists all records of all apps through its read tools,
a model sorts out the records that state rules, and those go to the normal intake extraction.

Searching with policy words finds a rule only when its record uses such a word, and only in apps that have a search
tool. Listing reads everything, so the open question is only which records hold rules, and a model decides that.

Interface: ``RunContext.data_list()`` returns ``{source: [record, ...]}``. A table (a worksheet) is one record with a
``rows`` list; tables always go to the extraction, because one row can carry a rule (a hold note, a limit) while the
other rows are data.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .llm import parse_json

ID_KEYS = ("id", "Id", "ID", "ts", "message_id")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
SORT_BATCH_CHARS = 14000  # records per sorting call
RECORD_CHARS = 6000  # one record in a prompt; a longer table is split into parts of its rows

SORT_SYSTEM = "You sort business records for a policy check. Output JSON only."
SORT_USER = """Request: {request}

Records from the company's apps, numbered. For each record say:
- "rule": it states how work must be done or decided (policy, procedure, criteria, threshold, exception, approval,
  routing, deadline, required format), or it changes, replaces or cancels such a rule, or it gives handling
  instructions for itself (for example a note saying not to contact this customer).
- "data": anything else (records to work on, facts, questions, unrelated files).
- "unsure": you cannot tell.

Return {{"labels": [{{"n": <number>, "label": "rule" | "data" | "unsure"}}]}} with one entry per record.

Records:
{records}"""


def record_ids(x: Any) -> Iterator[str]:
    """Every ID value in a record, nested ones too (a table's row IDs)."""
    if isinstance(x, dict):
        for k in ID_KEYS:
            if isinstance(x.get(k), (str, int)) and x.get(k) != "":
                yield str(x[k])
        for v in x.values():
            yield from record_ids(v)
    elif isinstance(x, list):
        for v in x:
            yield from record_ids(v)


def is_table(record: Any) -> bool:
    return isinstance(record, dict) and isinstance(record.get("rows"), list)


def split_table(record: Dict[str, Any], limit: int = RECORD_CHARS) -> List[Dict[str, Any]]:
    """A table longer than one prompt record becomes parts of whole rows, each with the table's other fields."""
    head = {k: v for k, v in record.items() if k != "rows"}
    parts: List[Dict[str, Any]] = []
    rows: List[Any] = []
    size = len(json.dumps(head, default=str))
    for row in record["rows"]:
        n = len(json.dumps(row, default=str))
        if rows and size + n > limit:
            parts.append({**head, "rows": rows})
            rows, size = [], len(json.dumps(head, default=str))
        rows.append(row)
        size += n
    parts.append({**head, "rows": rows})
    return parts


def _first_date(record: Any) -> str:
    """The earliest ISO date in the record's own fields ("" when it has none): older records are read first, so a
    newer record sees the rule it replaces."""
    found = [
        m.group(0)
        for v in (record.values() if isinstance(record, dict) else [])
        if isinstance(v, str)
        for m in [ISO_DATE.search(v)]
        if m
    ]
    return min(found) if found else ""


def flatten(by_source: Dict[str, List[Any]]) -> List[Tuple[str, Any]]:
    """(source, record) pairs: tables split into parts, then per source in date order."""
    out: List[Tuple[str, Any]] = []
    for source, records in by_source.items():
        items: List[Any] = []
        for r in records or []:
            items.extend(split_table(r) if is_table(r) else [r])
        items.sort(key=_first_date)
        out.extend((source, r) for r in items)
    return out


def _text(source: str, record: Any) -> str:
    return f"({source}) {json.dumps(record, default=str)[:RECORD_CHARS]}"


def _batches(texts: Sequence[str], limit: int) -> Iterator[List[int]]:
    cur: List[int] = []
    size = 0
    for i, t in enumerate(texts):
        if cur and size + len(t) > limit:
            yield cur
            cur, size = [], 0
        cur.append(i)
        size += len(t)
    if cur:
        yield cur


def sort_records(
    model: Callable[[str, str], str], request: str, items: List[Tuple[str, Any]], workers: int = 8
) -> Tuple[List[int], Dict[str, int]]:
    """Indexes of the items that go to extraction: tables, and records the model calls "rule" or "unsure". A record
    the model gives no label (a failed call, a missing entry) goes too: unknown is kept, never dropped."""
    texts = [_text(s, r) for s, r in items]
    groups = [g for g in _batches(texts, SORT_BATCH_CHARS)]

    def one(group: List[int]) -> Dict[int, str]:
        body = "\n\n".join(f"[{n}] {texts[i]}" for n, i in enumerate(group, 1))
        try:
            data = parse_json(model(SORT_SYSTEM, SORT_USER.format(request=request[:3000], records=body)))
        except Exception:  # a failed call keeps the whole group
            return {}
        labels: Dict[int, str] = {}
        for x in data.get("labels") or []:
            try:
                n = int(x.get("n"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 1 <= n <= len(group):
                labels[group[n - 1]] = str(x.get("label", "unsure"))
        return labels

    labels: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for got in ex.map(one, groups):
            labels.update(got)
    keep = [i for i, (_, r) in enumerate(items) if is_table(r) or labels.get(i, "unsure") != "data"]
    counts = {
        "records": len(items),
        "sort_calls": len(groups),
        "kept": len(keep),
        "tables": sum(is_table(r) for _, r in items),
        "unlabelled": len(items) - len(labels),
    }
    return keep, counts


def extraction_groups(items: List[Tuple[str, Any]], keep: List[int], limit: int) -> List[Tuple[str, List[Any]]]:
    """Kept records grouped per source, each group within the extraction prompt limit."""
    out: List[Tuple[str, List[Any]]] = []
    by_source: Dict[str, List[Any]] = {}
    for i in keep:
        by_source.setdefault(items[i][0], []).append(items[i][1])
    for source, recs in by_source.items():
        texts = [json.dumps(r, default=str)[:RECORD_CHARS] for r in recs]
        for group in _batches(texts, limit):
            out.append((source, [recs[i] for i in group]))
    return out


def listed_ids(items: List[Tuple[str, Any]]) -> Set[str]:
    return {i for _, r in items for i in record_ids(r)}


def unlisted(records: List[Any], ids: Optional[Set[str]]) -> List[Any]:
    """Records the listing did not read (none of their IDs is known). A record without an ID counts as unlisted."""
    if not ids:
        return records
    return [r for r in records if not (set(record_ids(r)) & ids)]
