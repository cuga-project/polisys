"""Finding the records that state rules (part of ``OnlinePolicyIntake``)."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from polisys.hooks import HookResult
from polisys.experimental.checks.guard_code import flatten_records
from polisys.values import _walk
from polisys.sources.intake.prompts import (
    CHANGE_QUERIES, DISCOVERY_QUERIES, ID_KEYS, MAX_BATCH_CHARS, MAX_DETECT_RECORDS, MAX_RECORD_CHARS, MIN_TEXT_CHARS,
    OWN_TEXT_CHARS, POLICY_WORDS, REQUEST_POLICY, _dicts, _norm_text, _strings,
)


class _Discovery:
    """At the user message: the listing and the discovery searches. On each tool result: policy words (tier 0) and the
    detector (tier 1) pick the candidate records, and the extraction (tier 2) reads them."""

    def _candidates(self, tool: str, args: Dict[str, Any], result: Any) -> List[Any]:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                result = {"text": result}
        arg_hit = any(POLICY_WORDS.search(str(v)) for v in args.values()) or any(
            str(v) in self.flagged_ids for v in args.values()
        )
        hits = []
        for node in _walk(result):
            if isinstance(node, dict):
                if any(isinstance(v, str) and POLICY_WORDS.search(v) for v in node.values()):
                    hits.append(node)
                    self.flagged_ids.update(
                        str(v) for k, v in node.items() if isinstance(v, (str, int)) and ID_KEYS.search(k)
                    )
        texts = [json.dumps(h, default=str) for h in hits]
        hits = [h for h, t in zip(hits, texts) if not any(t != o and t in o for o in texts)]  # keep outermost
        if arg_hit and not hits:
            hits = [result]
        return hits

    def _text_records(self, result: Any, hits: List[Any]) -> List[Any]:
        """Tier 1 input: records with a long text field that tier 0 did not flag and that are not inside a hit."""
        hit_texts = [json.dumps(h, default=str) for h in hits]
        out = []
        for node in _dicts(result):
            strs = [v for v in node.values() if isinstance(v, str)]
            if not strs or max(map(len, strs)) < MIN_TEXT_CHARS or any(node is h for h in hits):
                continue
            text = json.dumps(node, default=str)
            if not any(text in t for t in hit_texts):
                out.append(node)
        return out

    def _detect_and_extract(self, tool: str, hits: List[Any], texts: List[Any]) -> Dict[str, Any]:
        import time as _time

        t0 = _time.monotonic()
        out = self._detect_and_extract_inner(tool, hits, texts)
        out["ms"] = round((_time.monotonic() - t0) * 1000)
        return out

    def _detect_and_extract_inner(self, tool: str, hits: List[Any], texts: List[Any]) -> Dict[str, Any]:
        found = []
        if texts:
            scores = self.detector(
                self.request, [json.dumps(t, default=str)[:MAX_RECORD_CHARS] for t in texts]
            )
            found = [t for t, p in zip(texts, scores) if p >= self.detect_threshold]
            if (
                self.decisions is not None
            ):  # a rule change without policy words: ask with the rules as context
                known = "\n".join(f"- [{k['source']}] {k['text']}" for k in self._current_rules()[:40])
                same = "\n".join(f"- {json.dumps(h, default=str)[:400]}" for h in hits[:6])
                context = "\n".join(x for x in (known, same) if x)
                if context:
                    for t, p in zip(texts, scores):
                        if p >= self.detect_threshold or p < 0.1:
                            continue
                        q = self.decisions.rule_changes(
                            context, json.dumps(t, default=str)[:MAX_RECORD_CHARS]
                        )
                        if self.decisions.yes(q):
                            found.append(t)
                            self.records.append(
                                {
                                    "hook": "after_tool_call",
                                    "outcome": "rule_change_detected",
                                    "tool": tool,
                                    "p": q,
                                }
                            )
            self.records.append(
                {
                    "hook": "after_tool_call",
                    "outcome": "detected",
                    "tool": tool,
                    "records": len(found),
                    "checked": len(texts),
                }
            )
        dumped = [json.dumps(r, default=str) for r in found]
        found = [
            r for r, t in zip(found, dumped) if not any(t != o and t in o for o in dumped)
        ]  # keep outermost
        candidates = hits + found
        if not candidates:
            return {"tool": tool, "n_records": 0, "policies": []}
        return self._extract(tool, candidates)

    def on_user_message(self, message: str, thread_id: Optional[str]) -> HookResult:
        self.request = message
        self.refs.add_from(message, "the request")
        if self.enforcer is not None:
            self.enforcer.on_user_message(message, thread_id)
        if REQUEST_POLICY.search(message or ""):  # a condition or limit in the request: read it as a policy
            fut = self.pool.submit(
                self._extract, "user_request", [{"from": "the user (request)", "text": message}]
            )
            fut.add_done_callback(self._done)
            self.pending.append(fut)
        if self.discover and not self._discovered:
            self._discovered = True
            self._run_discovery()
        if self.listing is not None and not self._listed:
            self._listed = True
            self._run_listing()
        return HookResult()

    def _run_listing(self) -> None:
        """Read every listed record; the model keeps those that state rules (tables always), and the kept records go
        through the normal extraction in the background. A failed listing leaves the online intake as it was."""
        from polisys.sources import listing as L

        t0 = time.monotonic()
        try:
            items = L.flatten(self.listing() or {})
        except Exception as exc:  # the agent runs on without it
            self.records.append({"hook": "on_user_message", "outcome": "listing_error", "reason": str(exc)[:200]})
            return
        self.listed = L.Listed(items)
        for _, r in items:  # the same records read later by the agent are not sent again
            self.seen.add(hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest())

        def job() -> None:
            data_out = [] if self.listing_data else None
            keep, counts = L.sort_records(self.model, self.request, items, data_out=data_out)
            if data_out:
                self.data_notes = [(label, src, json.dumps(rec, default=str)[:1500]) for label, src, rec in data_out]
                counts = {**counts, "corrections": sum(x[0] == "correction" for x in data_out),
                          "examples": sum(x[0] == "example" for x in data_out)}
            groups = L.extraction_groups(items, keep, MAX_BATCH_CHARS)
            with ThreadPoolExecutor(max_workers=4) as ex:
                for data in ex.map(lambda g: self._extract(g[0], g[1]), groups):
                    self._take(data)
            self.records.append(
                {"hook": "on_user_message", "outcome": "listing", **counts, "extract_calls": len(groups),
                 "ms": round((time.monotonic() - t0) * 1000)}
            )

        self.pending.append(self.pool.submit(job))

    def _run_discovery(self) -> None:
        """Search message and document sources for written rules; candidates go through the normal intake."""
        n_calls = n_records = 0
        queries = DISCOVERY_QUERIES + (CHANGE_QUERIES if self.decisions is not None else ())
        for name, search in self.discover:
            for q in queries:
                try:
                    result = search(q)
                except Exception as exc:  # a failed search must not stop the agent
                    self.records.append(
                        {
                            "hook": "on_user_message",
                            "outcome": "discovery_error",
                            "tool": name,
                            "reason": str(exc)[:200],
                        }
                    )
                    continue
                n_calls += 1
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except ValueError:
                        pass
                self._remember(name, result)
                hits = self._candidates(name, {"query": q}, result)
                texts = (
                    self._new(self._text_records(result, hits)[:MAX_DETECT_RECORDS]) if self.detector else []
                )
                new = self._new(hits)
                if new or texts:
                    n_records += len(new) + len(texts)
                    fut = self.pool.submit(self._detect_and_extract, name, new, texts)
                    fut.add_done_callback(self._done)
                    self.pending.append(fut)
        self.records.append(
            {"hook": "on_user_message", "outcome": "discovery", "searches": n_calls, "candidates": n_records}
        )

    def _new(self, records: List[Any]) -> List[Any]:
        out = []
        for rec in records:
            h = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
            if h not in self.seen:
                self.seen.add(h)
                out.append(rec)
        return out

    def _remember(self, tool: str, result: Any) -> None:
        self.store.add(tool, result)

    def after_tool_call(self, tool: str, args: Dict[str, Any], result: Any) -> HookResult:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        self._remember(tool, result)
        self.called.append(tool)
        self.refs.observe(tool, args, result, self.is_write(tool))
        if not self.is_write(tool):
            self.observations.append({"tool": tool, "args": args, "records": flatten_records(result)[:500]})
        if self.is_write(tool):
            self.writes.append(f"{tool}({json.dumps(args, default=str)[:400]})")
            self.own_texts += [t for t in (_norm_text(v) for v in _strings(args)) if len(t) >= OWN_TEXT_CHARS]
        if self.enforcer is not None:
            self.enforcer.after_tool_call(tool, args, result)
        if self.is_write(tool):  # the result of the agent's own write is its output, not a rule source
            return HookResult(message=self._tool_notes())
        hits = self._candidates(tool, args, result)
        texts = self._new(self._text_records(result, hits)[:MAX_DETECT_RECORDS]) if self.detector else []
        new = self._new(hits)
        if self.own_texts:  # a record that carries text the agent wrote in this run is its own output, not a rule
            new, texts = [r for r in new if not self._own(r)], [r for r in texts if not self._own(r)]
        if self.listed is not None:  # the listing read these records already
            from polisys.sources.listing import unlisted

            new, texts = unlisted(new, self.listed), unlisted(texts, self.listed)
        if not new and not texts:
            return HookResult(message=self._tool_notes())
        fut = self.pool.submit(self._detect_and_extract, tool, new, texts)
        fut.add_done_callback(self._done)
        self.pending.append(fut)
        return HookResult(
            message=self._tool_notes(),
            details=[
                {
                    "rule": "intake",
                    "outcome": "candidate",
                    "tool": tool,
                    "records": len(new),
                    "text_records": len(texts),
                }
            ],
        )

    def _own(self, record: Any) -> bool:
        text = _norm_text(json.dumps(record, default=str, ensure_ascii=False))
        return any(t in text for t in self.own_texts)

    def _tool_notes(self) -> Optional[str]:
        """Reminders queued for this call, handed to its tool result when notes go there (``note_channel``)."""
        if self.note_channel != "tool":
            return None
        lines = list(self.notes) + (self.enforcer.pop_notes() if self.enforcer is not None else [])
        self.notes = []
        return "\n".join(lines) or None
