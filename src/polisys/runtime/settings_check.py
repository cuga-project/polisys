"""Settings that cannot work. ``build_runtime`` runs ``check_settings`` before it builds anything.

Most checks here involve more than one component: a setting that only one component reads, with that component off; two
components that would do the same thing; a host that lacks what a component needs. A check that needs only the component's
own settings and the task is in the component's ``build``. The order of the checks is fixed: a profile with two problems
reports the first one in this order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from polisys.runtime.profile import PolicyConfig, ledger_on, rules_on_package

if TYPE_CHECKING:
    from polisys.runtime.assembly import Assembly

# Nothing reads these yet (a later task does). Any other value would be ignored with no word about it.
NOT_READ_YET: Dict[str, Any] = {
    "authority.company_domains": [],
    "authority.may_change_rules": [],
}
PRODUCERS_BUILT = {"request", "package", "duty"}


def _value(cfg: PolicyConfig, path: str) -> Any:
    node: Any = cfg
    for key in path.split("."):
        node = getattr(node, key)
    return node


def check_settings(a: Assembly) -> None:
    for check in CHECKS:
        check(a)


def _host_has_what_the_ledger_needs(a: Assembly) -> None:
    cfg = a.cfg
    features = a.ctx.host_features
    if features is not None:
        if ledger_on(cfg) and "tool_blocked" not in features:
            raise a.fail(
                "obligations",
                "needs a host whose hook registry tells every hook about a stopped call (tool_blocked), and RunContext."
                "host_features says this host does not: a follow-up could ask for work that another hook blocked",
            )
        if "failure_report" not in features:
            logger.warning(
                "polisys: the host's hook registry does not report a hook that raises, so a failed check leaves no audit line"
            )


def _no_mode_is_still_auto(a: Assembly) -> None:
    cfg, checkers, organizers = a.cfg, a.cfg.checkers, a.cfg.organizers
    modes = {
        "checkers.rules": checkers.rules,
        "checkers.gate.mode": checkers.gate.mode,
        "checkers.verifier.mode": checkers.verifier.mode,
        "decisions.mode": cfg.decisions.mode,
        "organizers.procedure_tools": organizers.procedure_tools,
        "organizers.guide.mode": organizers.guide.mode,
        "organizers.loop.mode": organizers.loop.mode,
        "organizers.sop_tracker.mode": organizers.sop_tracker.mode,
        "organizers.navigator.mode": organizers.navigator.mode,
        "organizers.requirements_tracker.mode": organizers.requirements_tracker.mode,
    }
    for path, mode in {**modes, "checkers.write_checks": checkers.write_checks}.items():
        if mode == "auto":  # levels.resolve replaces every auto before this runs
            raise a.fail(
                path,
                "is still auto: build_runtime resolves it, and a component built alone must not skip that",
            )


def _package_is_the_one_the_profile_names(a: Assembly) -> None:
    cfg = a.cfg
    if cfg.package and "{policy}" in cfg.package and not a.ctx.policy_version:
        raise a.fail(
            "package", f"the profile names {cfg.package}, and the task gives no policy_version to fill in"
        )
    if cfg.package and a.pkg is None:
        raise a.fail("package", f"the profile names the package {cfg.package}, and none is available")
    if not cfg.package and a.pkg is not None:
        raise a.fail("package", "the profile says no package, and one was given")


def _no_setting_that_nothing_reads_yet(a: Assembly) -> None:
    cfg = a.cfg
    for path, default in NOT_READ_YET.items():
        if _value(cfg, path) != default:
            raise a.fail(path, "is set, and nothing reads it yet")


def _router_names_and_options(a: Assembly) -> None:
    checkers = a.cfg.checkers
    from polisys.checks.router import UnknownMethod, check_names, expander_options, router_names, router_options

    try:
        check_names(checkers.router.kind, checkers.router.expand)
    except UnknownMethod as exc:
        raise a.fail(
            "checkers.router.kind"
            if checkers.router.kind not in router_names()
            else "checkers.router.expand",
            str(exc),
        ) from exc
    r = checkers.router
    read = set(router_options(r.kind)) | set(expander_options(r.expand))
    for name in ("model", "top_k", "hops"):  # a setting the chosen methods do not read would change nothing
        if getattr(r, name) != getattr(type(r)(), name) and name not in read:
            raise a.fail(
                f"checkers.router.{name}",
                f"is set, and the router {r.kind} with the expander {r.expand} does not read it",
            )


def _producers_exist(a: Assembly) -> None:
    cfg = a.cfg
    unbuilt = set(cfg.obligations.producers) - PRODUCERS_BUILT
    if unbuilt:
        raise a.fail("obligations.producers", f"{sorted(unbuilt)}: no such producer exists yet")


def _write_checks_and_risk_have_a_check(a: Assembly) -> None:
    cfg, checkers = a.cfg, a.cfg.checkers
    # The verifier of the online intake keeps its measured path (it takes every statement that did not compile, whatever the
    # scope says); the scope sets what the verifier of the package's rules takes.
    write = checkers.write_checks == "on"
    if write and not (rules_on_package(cfg) or checkers.gate.mode == "on"):
        raise a.fail(
            "checkers.write_checks",
            "is on, and neither the package's rules nor the gate is on, so it has no check to run",
        )
    if a.resolution.risk_active and not write:
        raise a.fail(
            "control.risk.level3_tools",
            "is set, and separate hooks cannot give one tool more checks: it needs checkers.write_checks, and no check is "
            "on that it could hold",
        )
    if a.resolution.risk_active and not a.resolution.auto_stages:
        raise a.fail(
            "control.risk.level3_tools",
            "is set, and it does nothing: a check set to on or off runs the same for every tool, so one tool can get more "
            "checks only from a check left to auto: set rules, gate or verifier to auto, or remove the setting",
        )


def _control_settings_are_read(a: Assembly) -> None:
    cfg, checkers = a.cfg, a.cfg.checkers
    # The control settings change something only where something reads the level: a component left to ``auto``,
    # ``write_checks``, a ledger with no ``item_level``, level 0 (shadow), and the gate's comparison of strength.
    control, res = cfg.control, a.resolution
    gate_compares = checkers.gate.mode == "on" and checkers.gate.min_strength == "agent"
    unread = []
    if control.level != "by_model" and not res.level_read:
        unread.append("control.level")
    if control.by_model and not (res.level_read or gate_compares):
        unread.append("control.by_model")
    if control.default_level != 1 and not (control.level == "by_model" and res.level_read):
        unread.append("control.default_level")
    # A declared rule list is read by the rule hooks (the brief shows each rule with its exceptions) and by the procedure monitor
    # (it builds none), whatever else the level decides.
    rule_list_read = "rules" in cfg.shapes and rules_on_package(cfg)
    if cfg.shapes and not (res.had_auto or rule_list_read):
        unread.append("shapes")
    if unread:
        raise a.fail(
            unread[0],
            "is set, and nothing reads it: the profile lists every component, so no level decides anything (a component "
            "left to auto, checkers.write_checks, a ledger with no item_level or level 0 would read it)",
        )


def _one_sender_of_follow_ups(a: Assembly) -> None:
    cfg, organizers = a.cfg, a.cfg.organizers
    ledger = ledger_on(cfg)
    if ledger and organizers.navigator.mode == "on":
        raise a.fail("obligations", "the navigator checks the same hand-offs: turn one of them off")
    if ledger and organizers.requirements_tracker.mode == "on":
        raise a.fail(
            "obligations",
            "the requirements tracker sends its own follow-up, and with a producer only the ledger sends them: turn one of "
            "them off",
        )


def _every_setting_has_a_reader(a: Assembly) -> None:
    cfg, checkers, organizers = a.cfg, a.cfg.checkers, a.cfg.organizers
    # A setting that only one path reads, with that path off.
    data = cfg.sources.data
    rules = checkers.rules == "on"
    loop = organizers.loop.mode == "on"
    ledger = ledger_on(cfg)
    later = (
        a.resolution.pending
    )  # components that auto left off: the level decides, and a higher level would read these
    router = checkers.router
    default_router = type(router)()
    changed = [
        f
        for f in ("kind", "expand", "model", "top_k", "hops")
        if getattr(router, f) != getattr(default_router, f)
    ]
    gate_reads = (
        checkers.gate.mode == "on" and checkers.gate.scope == "active_procedures"
    ) or "gate" in later
    readers = [
        (
            f"checkers.router.{changed[0] if changed else 'kind'}",
            bool(changed),
            gate_reads,
            "checkers.gate with scope active_procedures: only the gate reads the scope",
        ),
        ("priority_classes", bool(cfg.priority_classes), bool(cfg.package), "a package"),
        (
            "notes.brief_guidance",
            cfg.notes.brief_guidance,
            cfg.notes.brief != "off" and (rules_on_package(cfg) or "rules" in later),
            "the brief (notes.brief is not off) and checkers.rules on a package",
        ),
        ("obligations.max_follow_ups", cfg.obligations.max_follow_ups != 1, ledger, "a producer"),
        (
            "obligations.wrong_values",
            cfg.obligations.wrong_values != "known_origin_only",
            ledger,
            "a producer",
        ),
        ("obligations.item_level", cfg.obligations.item_level is not None, ledger, "a producer"),
        ("sources.discovery", cfg.sources.discovery, data, "sources.data"),
        ("sources.listing", cfg.sources.listing, data, "sources.data"),
        ("authority.supersedes", cfg.authority.supersedes, data, "sources.data"),
        ("checkers.compile", bool(checkers.compile), data, "sources.data"),
        ("checkers.detector", bool(checkers.detector), data, "sources.data"),
        ("checkers.guard_code", checkers.guard_code != "off", data, "sources.data"),
        ("checkers.read_before_write", checkers.read_before_write != "off", data, "sources.data"),
        ("decisions.mode", cfg.decisions.mode == "on", data, "sources.data"),
        (
            "decisions.require_evidence",
            cfg.decisions.require_evidence,
            loop or "loop" in later,
            "organizers.loop",
        ),
        (
            "checkers.gate.seed_request",
            checkers.gate.seed_request,
            checkers.gate.mode == "on" or "gate" in later,
            "checkers.gate",
        ),
        ("checkers.rules", rules, data or bool(cfg.package), "a package or sources.data"),
        (
            "checkers.verifier.mode",
            checkers.verifier.mode == "on",
            rules or data or "rules" in later,
            "checkers.rules or sources.data",
        ),
        (
            "checkers.method_choice",
            bool(checkers.method_choice),
            rules or data or "rules" in later,
            "checkers.rules or sources.data",
        ),
        (
            "checkers.narrow_questions",
            bool(checkers.narrow_questions),
            cfg.decisions.mode == "on"
            or (router.kind == "bm25_kev" and gate_reads)
            or organizers.navigator.mode == "on"
            or (loop and organizers.loop.style == "discovery")
            or bool(later & {"decisions", "navigator", "loop"}),
            "decisions, the navigator, the discovery loop or the router bm25_kev",
        ),
    ]
    for path, is_set, is_read, reader in readers:
        if is_set and not is_read:
            raise a.fail(path, f"is set, and nothing reads it: it needs {reader}")


def _priority_classes_can_rank(a: Assembly) -> None:
    cfg = a.cfg
    if cfg.priority_classes and a.pkg is not None:
        # The ranks themselves decide a conflict when the package is validated or its conflicts are settled
        # (``validate_package`` and ``resolve_conflicts`` take the list); a run reports the documents it cannot rank.
        from polisys.compile.validate import priority_findings

        for found in priority_findings(a.pkg, cfg.priority_classes):
            a.notes.append(("priority_classes", found.message))


def _request_source_matches_its_readers(a: Assembly) -> None:
    cfg, checkers, organizers = a.cfg, a.cfg.checkers, a.cfg.organizers
    ledger = ledger_on(cfg)
    later = a.resolution.pending
    reads_request = (
        organizers.requirements_tracker.mode == "on"
        or "request" in cfg.obligations.producers
        or ledger
        or (checkers.gate.mode == "on" and checkers.gate.seed_request)
        or (cfg.sources.request and "requirements_tracker" in later)  # the level decides whether it reads it
    )
    if cfg.sources.request != reads_request:
        raise a.fail(
            "sources.request",
            f"is {cfg.sources.request}, and the components that read the request are "
            f"{'on' if reads_request else 'off'} (the requirements tracker, the producers request and package, a gate "
            "with seed_request)",
        )


def _brief_placement_fits_the_host(a: Assembly) -> None:
    cfg = a.cfg
    if cfg.notes.brief == "message":
        if a.ctx.host_note_placement != "message":
            raise a.fail("notes.brief", "is message, and the host's note placement (RunContext.host_note_placement) is not")


# The order is the order of the errors: a profile with two problems reports the first.
CHECKS = (
    _host_has_what_the_ledger_needs,
    _no_mode_is_still_auto,
    _package_is_the_one_the_profile_names,
    _no_setting_that_nothing_reads_yet,
    _router_names_and_options,
    _producers_exist,
    _write_checks_and_risk_have_a_check,
    _control_settings_are_read,
    _one_sender_of_follow_ups,
    _every_setting_has_a_reader,
    _priority_classes_can_rank,
    _request_source_matches_its_readers,
    _brief_placement_fits_the_host,
)
