"""Tier 1 detectors for online intake: a score per record, how likely it states a policy.

``systemone_detector`` asks a decision model (Kev-4B by default) one yes/no/unknown question per record through
OpenRouter's ``/api/v1/systemone`` endpoint. Offline probe on saved tool records (2026-10-31): Kev-4B 94.5% recall,
1.1% false alarms at threshold 0.47, 0.8 s p50. A record that cannot be scored gets 1.0, so it goes on to extraction
instead of being dropped.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence

SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"
POLICY_QUESTION = (
    "Does this record state company rules, a procedure or a policy that an assistant must follow when it "
    "acts, including a record that adds to, changes or replaces such rules? Plain business data, reports, "
    "tickets and ordinary messages are not policies."
)
REQUEST_CHARS = 1500


def systemone_detector(
    model: str = "jaredpalmer/kev-4b",
    api_key: Optional[str] = None,
    url: str = SYSTEMONE_URL,
    workers: int = 8,
    timeout: float = 60.0,
    retries: int = 3,
) -> Callable[[str, Sequence[str]], List[float]]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    usage = {"calls": 0, "errors": 0, "cost": 0.0}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(request: str, text: str) -> float:
        body = {
            "model": model,
            "state": f"User request: {request[:REQUEST_CHARS]}\n\nRecord (returned by a business tool):\n{text}",
            "questions": {
                "is_policy": {
                    "type": "noul",
                    "instructions": POLICY_QUESTION,
                    "criteria": {"true": "yes", "false": "no"},
                }
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "polisys intake",
            },
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                p = float(data.get("answers", {}).get("is_policy", {}).get("noul", 1.0))
                with lock:
                    usage["calls"] += 1
                    usage["cost"] += float((data.get("usage") or {}).get("cost") or 0.0)
                return p
            except (urllib.error.URLError, OSError, ValueError, TypeError):
                time.sleep(2 + 3 * attempt)
        with lock:
            usage["errors"] += 1
        return 1.0

    def detect(request: str, texts: Sequence[str]) -> List[float]:
        return list(pool.map(lambda t: one(request, t), texts))

    detect.model_name = model  # type: ignore[attr-defined]
    detect.usage = usage  # type: ignore[attr-defined]
    return detect


def systemone_questions(
    model: str = "jaredpalmer/kev-4b",
    api_key: Optional[str] = None,
    url: str = SYSTEMONE_URL,
    timeout: float = 60.0,
    retries: int = 3,
) -> Callable[[str, dict], dict]:
    """Typed yes/no questions to a decision model: ``ask(state, {name: instructions}) -> {name: p_yes}``.

    All questions share one state and one call. A question that cannot be answered gets ``None`` (unknown), so the
    caller keeps its default instead of acting on a guess.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    usage = {"calls": 0, "errors": 0, "cost": 0.0}
    lock = threading.Lock()

    def ask(state: str, questions: dict) -> dict:
        body = {
            "model": model,
            "state": state,
            "questions": {
                n: {"type": "noul", "instructions": q, "criteria": {"true": "yes", "false": "no"}}
                for n, q in questions.items()
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "polisys decisions",
            },
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                answers = data.get("answers", {})
                with lock:
                    usage["calls"] += 1
                    usage["cost"] += float((data.get("usage") or {}).get("cost") or 0.0)
                out = {}
                for n in questions:
                    try:
                        out[n] = float(answers[n]["noul"])
                    except (KeyError, TypeError, ValueError):
                        out[n] = None
                return out
            except (urllib.error.URLError, OSError, ValueError, TypeError):
                time.sleep(2 + 3 * attempt)
        with lock:
            usage["errors"] += 1
        return {n: None for n in questions}

    ask.model_name = model  # type: ignore[attr-defined]
    ask.usage = usage  # type: ignore[attr-defined]
    return ask
