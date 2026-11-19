"""Checks that a built rule is well-formed and matches the tool schemas and the policy text.

``check_rule`` is used on every rule a model or a person writes (pipeline and skill). It returns the reason a rule
is rejected, or None. ``validate_build`` checks a whole build. ``validate_package`` checks a whole policy package
(design sections 9 and 10): rule slots and types, coverage, rules with no trigger tool, conflicts between rules.
It only reports; ``resolve_conflicts`` is the one function that changes a package, and only a new one. Nothing here
knows any benchmark: it only compares rules with tool schemas, quotes and each other.

The parts, in the order they build on each other:

- ``rule``: one rule against the policy text and the tool schemas (``check_rule``, ``quote_ok``).
- ``build``: a whole build (``validate_build``, ``diff_builds``).
- ``slots``: the slots of each rule type (``slot_problem``), ``PackageError``, outcomes and warning codes.
- ``fields``: the fields of a package and their form.
- ``obligations``: obligations and their value bindings.
- ``invented``: text that is not in the policy.
- ``coverage``: every statement has an outcome.
- ``conflicts``: rules that cannot both hold (``resolve_conflicts``).
- ``whole_package``: ``validate_package``.

    from polisys.compile.validate import check_rule, validate_package
"""

from polisys._aliases import names_of

_PARTS = ("rule", "build", "slots", "fields", "obligations", "invented", "coverage", "conflicts", "whole_package")

# A name of one of the parts, read on first use: ``polisys.compile.validate`` was one module before 2026-11.
__getattr__ = names_of(__name__, *(f"{__name__}.{part}" for part in _PARTS))
