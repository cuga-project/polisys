"""The rules of the folder layout, checked on the import statements of the source (also imports inside functions).

1. ``package/`` imports nothing from the rest of polisys.
2. Only ``hosts/`` imports an agent framework.
3. Core code does not import ``experimental/``. Two files may: ``runtime/registry.py`` lists every component, and
   ``runtime/profile.py`` gives every component's settings a place in the profile.
4. Components do not import each other. A component is a module, or a folder of parts, under ``sources/``, ``checks/``,
   ``guidance/``, ``finish/`` or ``experimental/<role>/``. The parts of one component may import each other. The ``build``
   function of a component imports what it wires together (the rules build their verifier, the gate its router), so
   those imports are in the list too.

The code does not meet rules 1 to 4 everywhere yet. Each list below holds the imports that break a rule today. A list may
only get shorter: a new import that breaks a rule fails the test, and an import that is gone must leave its list. The one
case that adds an entry is a split of a file that held core and experimental code together: the import was inside the file
before, and is visible now (``guidance.work_method`` builds the experimental work-method hooks).
"""

import ast
from pathlib import Path

import polisys

ROOT = Path(polisys.__file__).parent
FRAMEWORKS = ("langchain_core", "langchain", "langgraph", "cuga")
COMPONENT_FOLDERS = ("sources", "checks", "guidance", "finish", "experimental")
WIRING = ("runtime.registry", "runtime.profile")


def _is_module(name: str) -> bool:
    """A file, or a folder with an ``__init__``."""
    base = ROOT / name.replace(".", "/")
    return base.with_suffix(".py").exists() or (base / "__init__.py").exists()


def component(module: str) -> str:
    """The component a module belongs to: ``checks.rules``, or ``experimental.finish.ledger`` for each of its parts."""
    parts = module.split(".")
    return ".".join(parts[: 3 if parts[0] == "experimental" else 2])


def imports():
    """(importing module, imported module) for polisys modules, and (importing module, framework) for frameworks. Names
    are relative to the package: ``checks.rules``."""
    inside, outside = set(), set()
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        here = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]  # from a folder: its modules
                targets = [
                    name if name.startswith("polisys.") and _is_module(name[len("polisys.") :]) else node.module
                    for name in names
                ]
            else:
                assert not isinstance(node, ast.ImportFrom) or node.level == 0, f"{path}: a relative import"
                continue
            for target in targets:
                if target.split(".")[0] in FRAMEWORKS:
                    outside.add((here, target.split(".")[0]))
                elif target.startswith("polisys.") and _is_module(target[len("polisys.") :]):
                    inside.add((here, target[len("polisys.") :]))
    return inside, outside


INSIDE, OUTSIDE = imports()

PACKAGE_IMPORTS_THE_REST = {
    ("package.files", "compile.handlers.prose"),
}
FRAMEWORK_OUTSIDE_HOSTS = {
    ("experimental.guidance.procedure_store", "langchain_core"),
    ("experimental.organizers.sop_navigator", "langchain_core"),
    ("hooks", "langchain_core"),
}
CORE_IMPORTS_EXPERIMENTAL = {
    ("guidance.work_method", "experimental.guidance.work_method_notes"),
    ("runtime.assembly", "experimental.guidance.procedure_store"),
    ("runtime.levels", "experimental.checks.gate"),
    ("runtime.trace", "experimental.checks.gate"),
    ("runtime.trace", "experimental.organizers.sop_navigator"),
    ("sources.intake", "experimental.checks.guard_code"),
}
COMPONENT_IMPORTS_COMPONENT = {
    ("checks.router", "checks.retrieval"),
    ("checks.rules", "checks.cedar"),
    ("checks.rules", "checks.facts"),
    ("checks.rules", "checks.verifier"),
    ("experimental.checks.default_values", "experimental.guidance.tool_notes"),
    ("experimental.checks.default_values", "experimental.guidance.work_method_notes"),
    ("experimental.checks.gate", "checks.router"),
    ("experimental.finish.ledger", "checks.rules"),
    ("experimental.finish.ledger", "experimental.finish.request_checklist"),
    ("experimental.finish.ledger", "experimental.guidance.procedure_store"),
    ("experimental.finish.ledger", "finish.requirements_tracker"),
    ("experimental.finish.request_checklist", "experimental.finish.ledger"),
    ("experimental.guidance.policy_lines", "experimental.guidance.tool_notes"),
    ("experimental.guidance.procedure_push", "checks.router"),
    ("experimental.guidance.procedure_push", "experimental.checks.gate"),
    ("experimental.organizers.sop_loop", "checks.retrieval"),
    ("experimental.organizers.sop_loop", "experimental.guidance.procedure_guide"),
    ("finish.requirements", "finish.followups"),
    ("finish.requirements_tracker", "finish.followups"),
    ("finish.requirements_tracker", "finish.requirements"),
    ("guidance.work_method", "experimental.guidance.work_method_notes"),
    ("sources.intake", "checks.decisions"),
    ("sources.intake", "checks.facts"),
    ("sources.intake", "checks.rules"),
    ("sources.intake", "experimental.checks.guard_code"),
    ("sources.intake", "finish.followups"),
    ("sources.intake", "sources.listing"),
    ("sources.intake", "sources.references"),
}


def test_the_scan_sees_the_source():
    assert ("checks.rules.rule_set_hooks", "hooks") in INSIDE and ("hosts.cuga", "cuga") in OUTSIDE


def test_package_imports_nothing_from_the_rest():
    found = {(a, b) for a, b in INSIDE if a.startswith("package.") and not b.startswith("package.")}
    assert found == PACKAGE_IMPORTS_THE_REST


def test_only_hosts_imports_an_agent_framework():
    found = {(a, b) for a, b in OUTSIDE if not a.startswith("hosts")}
    assert found == FRAMEWORK_OUTSIDE_HOSTS


def test_core_code_does_not_import_experimental_code():
    found = {
        (component(a), b)  # the parts of one component count once
        for a, b in INSIDE
        if not a.startswith("experimental.") and b.startswith("experimental.") and a not in WIRING
    }
    assert found == CORE_IMPORTS_EXPERIMENTAL


def test_components_do_not_import_each_other():
    found = {
        (component(a), component(b))
        for a, b in INSIDE
        if a.split(".")[0] in COMPONENT_FOLDERS and b.split(".")[0] in COMPONENT_FOLDERS and component(a) != component(b)
    }
    assert found == COMPONENT_IMPORTS_COMPONENT
