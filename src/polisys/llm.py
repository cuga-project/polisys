"""Small-model clients for the policy layer. Each client is ``call(system, user) -> str`` and counts its token use
in ``call.usage`` (for the cost ledger) and names its model in ``call.model_name``.

- ``openai_json_model``: any OpenAI-compatible gateway; default LiteLLM (``OPENAI_BASE_URL``, ``OPENAI_API_KEY``).
- ``openrouter_json_model``: OpenRouter (``OPENROUTER_API_KEY``), for models that LiteLLM does not serve
  (gpt-oss-20b, gpt-oss-safeguard-20b).
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Callable, Dict, Optional

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def openai_json_model(
    model: str,
    timeout: float = 90.0,
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
) -> Callable[[str, str], str]:
    """Chat call through an OpenAI-compatible gateway. Returns the raw text of the answer."""
    from openai import OpenAI

    if not model:
        raise ValueError("openai_json_model needs a model name: the policy config or the caller names it")

    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get(api_key_env),
        timeout=timeout,
        max_retries=max_retries,
    )
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    lock = threading.Lock()

    def call(system: str, user: str) -> str:
        extra = {"max_tokens": max_tokens} if max_tokens else {}  # long answers (a checklist per record)
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            **extra,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        with lock:
            usage["calls"] += 1
            usage["input_tokens"] += getattr(r.usage, "prompt_tokens", 0) or 0
            usage["output_tokens"] += getattr(r.usage, "completion_tokens", 0) or 0
        return r.choices[0].message.content or ""

    call.model_name = model  # type: ignore[attr-defined]
    call.usage = usage  # type: ignore[attr-defined]
    return call


def openrouter_json_model(model: str, timeout: float = 90.0) -> Callable[[str, str], str]:
    return openai_json_model(
        model, timeout=timeout, base_url=OPENROUTER_BASE, api_key_env="OPENROUTER_API_KEY"
    )


def today_line(today: str) -> str:
    """The date a check must judge dates against (a task's world time in benchmarks). Without it a model guesses
    today, and "older than 60 days" goes wrong."""
    return f"Today's date: {today}\n\n" if today else ""


def parse_json(text: str) -> Dict[str, Any]:
    """The first JSON object in a model answer, or {} when there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}
