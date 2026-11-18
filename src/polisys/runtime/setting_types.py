"""The types that settings share. A component's ``Settings`` uses them, and ``runtime/profile.py`` reads them when it
loads a file. This module imports nothing else from polisys."""

from __future__ import annotations

from typing import Annotated, Literal

Mode = Literal["auto", "on", "off"]


class _AtLeast:
    """Marker for ``Annotated``: the whole number may not be below ``floor``."""

    def __init__(self, floor: int):
        self.floor = floor


Count = Annotated[int, _AtLeast(0)]  # a number of blocks or follow-ups: 0 or more
