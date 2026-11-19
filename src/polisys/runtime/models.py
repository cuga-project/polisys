"""The model factory of a real run. ``Assembly`` asks it for every model a component needs."""

from __future__ import annotations

import os
from typing import Any, Callable


class Models:
    """The real model factory. A dry run gives ``dry_run.RecordingModels`` instead."""

    def json_model(self, model: str, **kwargs: Any) -> Callable[[str, str], str]:
        from polisys.llm import openai_json_model

        return openai_json_model(model, **kwargs)

    def openrouter_json_model(self, model: str, **kwargs: Any) -> Callable[[str, str], str]:
        from polisys.llm import openrouter_json_model

        return openrouter_json_model(model, **kwargs)

    @staticmethod
    def _need_key(model: str) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):  # with no key, every answer would be "unknown"
            raise ValueError(f"{model} needs OPENROUTER_API_KEY, and it is not set")

    def questions(self, model: str) -> Callable[..., Any]:
        from polisys.sources.detectors import systemone_questions

        self._need_key(model)
        return systemone_questions(model)

    def detector(self, model: str) -> Callable[..., Any]:
        from polisys.sources.detectors import systemone_detector

        self._need_key(model)
        return systemone_detector(model)
