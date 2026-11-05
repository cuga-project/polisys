"""Enforce a policy package at tool-call time, with no agent and no model.

    python examples/quickstart.py

It loads the Facilities Desk package (made up, see examples/facilities_desk/policy.md), builds the runtime of a
profile, and shows what the hooks say about three calls an agent might make.
"""

import json
from pathlib import Path

from polisys.hooks import HookRegistry
from polisys.package import load_package
from polisys.policy_config import load_config
from polisys.runtime import RunContext, build_runtime

HERE = Path(__file__).parent

package = load_package(HERE / "facilities_desk" / "package.json")
tool_specs = json.loads((HERE / "facilities_desk" / "tools.json").read_text())

for profile in ("baseline", "rules_only"):
    config = load_config(HERE / "profiles" / f"{profile}.yaml")
    # A profile that says "package: none" must get no package: the runtime refuses the mismatch instead of guessing.
    opened = None if profile == "baseline" else package
    parts = build_runtime(config, opened, RunContext(task_id="demo", tool_specs=tool_specs))
    registry = HookRegistry(parts.hooks)
    print(f"\n== profile {profile}: {len(parts.hooks)} hook(s)")
    calls = [
        ("create_ticket", {"requester": "ann", "priority": "normal"}),  # fine
        ("create_ticket", {"priority": "normal"}),  # no requester: the policy says every ticket needs one
        ("create_ticket", {"requester": "ann", "priority": "whenever"}),  # not one of low, normal, urgent
    ]
    for tool, args in calls:
        message = registry.before_tool_call(tool, args)
        print(f"  {tool}({args})\n    -> {'BLOCKED: ' + message if message else 'allowed'}")
