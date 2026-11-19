"""Run time: what one run is made of.

- ``profile``: the profile file and its settings. ``setting_types``: the types settings share.
- ``context``: what a task gives (``RunContext``), what the runner gets (``RuntimeParts``), what a component adds (``Piece``).
- ``registry``: the list of components, in hook order.
- ``assembly``: ``build_runtime`` and what the components of one run share (``Assembly``).
- ``settings_check``: settings that cannot work together.
- ``levels``: control levels. ``trace``: the audit. ``dry_run``: the wiring description without a model.

    from polisys.runtime import RunContext, build_runtime
"""

from polisys._aliases import names_of

# The names of ``assembly`` and ``settings_check``, read on first use: ``polisys.runtime`` was one module that held both
# before the folders existed.
__getattr__ = names_of(__name__, "polisys.runtime.assembly", "polisys.runtime.settings_check")
