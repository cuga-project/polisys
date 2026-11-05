"""Links: references, exceptions folded into rules, supersedes across documents, dangling references."""

import json
from pathlib import Path

from polisys.documents import load_document, load_policy_set
from polisys.handlers.links import drop_superseded, find_links, fold_exceptions
from polisys.model import LinkType, PolicySet, Rule

FIX = Path(__file__).parent / "fixtures"


def _ap():
    return PolicySet(documents=[load_document(FIX / "ap_policy.md", meta={"id": "ap"})])


def _st(ps, section, step=None, contains=None):
    return next(
        st
        for d in ps.documents
        for sec in d.sections
        if sec.id == section
        for st in sec.statements
        if (step is None or st.source.get("step") == step) and (contains is None or contains in st.text)
    )


def test_section_reference_link():
    ps = _ap()
    links, _ = find_links(ps)
    step3 = _st(ps, "4.2", step=3)
    assert any(lk.src == step3.id and lk.dst == "ap:4.3" and lk.type == LinkType.REFERS_TO for lk in links)


def test_exception_folded_into_rule():
    ps = _ap()
    links, _ = find_links(ps)
    step3, emergency = _st(ps, "4.2", step=3), _st(ps, "8.1", contains="may skip")
    assert any(
        lk.src == emergency.id and lk.dst == step3.id and lk.type == LinkType.EXCEPTION_OF for lk in links
    )
    rule = Rule(
        id="r1", type="verify", rule={"type": "verify", "statement": step3.text}, statement_ids=[step3.id]
    )
    fold_exceptions([rule], links, ps)
    assert rule.rule["unless"][0]["condition"] == emergency.text


def test_cross_document_supersedes():
    files = [FIX / "ap_set" / "sop.md", FIX / "ap_set" / "update_2026.md"]
    ps = load_policy_set(
        files, {"sop": {"effective_date": "2026-07-01"}, "update_2026": {"effective_date": "2026-10-01"}}
    )
    links, _ = find_links(ps)
    memo = _st(ps, "1.1", contains="supersedes")
    assert any(lk.src == memo.id and lk.dst == "sop:8.2" and lk.type == LinkType.SUPERSEDES for lk in links)
    old = _st(ps, "8.2", contains="Utilities")
    rules = [
        Rule(id="old", type="verify", rule={}, statement_ids=[old.id]),
        Rule(id="new", type="verify", rule={}, statement_ids=[memo.id]),
    ]
    kept, cov = drop_superseded(rules, links, ps)
    assert [r.id for r in kept] == ["new"] and cov[0].kind == "superseded"


def test_dangling_reference_is_coverage_item():
    ps = PolicySet(
        documents=[
            load_document(
                "## 1. Rules\n\n- Use the template in Appendix Q.\n- See §9.9 for limits.\n", meta={"id": "d"}
            )
        ]
    )
    links, cov = find_links(ps)
    assert links == [] and sorted(c.kind for c in cov) == ["dangling_link", "dangling_link"]


def test_model_link_without_target_is_dropped():
    ps = _ap()
    st = _st(ps, "8.1", contains="may skip")

    def model(system, user):
        return json.dumps(
            {
                "links": [
                    {"type": "exception_of", "src": st.id, "dst": "ap:99.9"},
                    {"type": "exception_of", "src": st.id, "dst": "ap:4.3"},
                    {"type": "source_of_truth", "src": st.id, "dst": "ap:4.4"},
                ]
            }
        )

    links, _ = find_links(ps, model)
    assert not any(lk.dst == "ap:99.9" for lk in links)
    assert any(lk.type == LinkType.EXCEPTION_OF and lk.src == st.id and lk.dst == "ap:4.3" for lk in links)
    assert not any(
        lk.type == LinkType.SOURCE_OF_TRUTH for lk in links
    )  # no precedence words in the statement


def test_articles_and_lettered_clauses_with_code_defaults():
    ps = PolicySet(documents=[load_document(FIX / "refund_articles.txt", meta={"id": "rf"})])
    ids = [s.id for s in ps.documents[0].sections]
    assert {"I", "II", "II.a", "II.d", "IV"} <= set(ids)
    links, cov = find_links(ps)
    b = _st(ps, "II.b")
    assert any(lk.src == b.id and lk.dst == "rf:II.d" and lk.type == LinkType.EXCEPTION_OF for lk in links)
    assert any(lk.dst == "rf:IV" for lk in links)
    assert any("Article V" in c.reason for c in cov)  # Article V does not exist


def test_learned_structure_for_an_unusual_style():
    text = "Rule 7: Refunds\nR7.1 Refunds above $500 need approval.\nR7.2 Disputes follow R7.1 first.\n"

    def model(system, user):
        return json.dumps(
            {
                "headings": [
                    {"pattern": r"^R(?P<id>\d+\.\d+)\s+", "level": 2, "kind": "clause"},
                    {"pattern": "^(bad[", "level": 1},
                ],
                "references": [r"\bR(?P<ref>\d+\.\d+)\b"],
            }
        )

    doc = load_document(text, meta={"id": "u"}, model=model)
    assert "7.1" in {s.id for s in doc.sections} or "7.7.1" in {s.id for s in doc.sections}
    assert all(h["pattern"] != "^(bad[" for h in doc.structure["headings"])
    links, _ = find_links(PolicySet(documents=[doc]))
    r72 = next(st for st in doc.statements if "Disputes" in st.text)
    assert any(lk.src == r72.id and lk.dst.endswith("7.1") for lk in links)


def test_untrusted_exception_never_changes_a_trusted_rule():
    from polisys.model import Trust

    sop = load_document("## 1. Refunds\n\n1.1 Refunds above $500 need approval.\n", meta={"id": "sop"})
    mail = load_document(
        "## 1. Note\n\n1.1 Refunds may skip the approval in §1.1 of SOP-1.\n",
        meta={"id": "mail", "trust": Trust.UNTRUSTED},
    )
    ps = PolicySet(documents=[sop, mail])
    target = sop.statements[0]
    link = __import__("polisys.model", fromlist=["Link"]).Link(
        type=LinkType.EXCEPTION_OF, src=mail.statements[0].id, dst=target.id
    )
    rule = Rule(id="r", type="verify", rule={"type": "verify"}, statement_ids=[target.id])
    fold_exceptions([rule], [link], ps)
    assert "unless" not in rule.rule
