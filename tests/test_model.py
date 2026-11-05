"""Policy domain model: JSON round trip, statement types, and precedence between statements."""

import pytest

from polisys.model import (
    Document,
    Link,
    LinkType,
    Modality,
    PolicySet,
    Section,
    Statement,
    StatementType,
    Trust,
    from_json,
    resolve_conflict,
    to_json,
)


def _doc(doc_id, text, trust=Trust.TRUSTED, date=None, rank=0):
    st = Statement(
        id=f"{doc_id}:1:1",
        text=text,
        types=[StatementType.THRESHOLD],
        modality=Modality.MUST,
        source={"document": doc_id, "section": "1", "line": 1},
        quote=text,
    )
    return Document(
        id=doc_id,
        kind="markdown",
        title=doc_id,
        trust=trust,
        effective_date=date,
        authority_rank=rank,
        sections=[Section(id="1", title="Limits", page=1, statements=[st])],
        source={"document": doc_id},
        quote=doc_id,
    )


def _set(*docs, links=()):
    return PolicySet(documents=list(docs), links=list(links))


def test_json_round_trip_keeps_source_and_types():
    ps = _set(
        _doc("sop", "Refunds above $500 need approval."),
        _doc("update", "Refunds above $300 need approval."),
        links=[
            Link(
                type=LinkType.SUPERSEDES,
                src="update:1:1",
                dst="sop:1:1",
                source={"document": "update"},
                quote="replaces the refund limit",
            )
        ],
    )
    back = from_json(PolicySet, to_json(ps))
    assert back == ps
    assert back.documents[0].sections[0].statements[0].types == [StatementType.THRESHOLD]
    assert back.documents[0].sections[0].statements[0].source["section"] == "1"
    assert to_json(ps)["documents"][0]["trust"] == "trusted"


def test_supersedes_wins_over_newer_date():
    old, new = _doc("a", "limit 500", date="2026-01-01"), _doc("b", "limit 300", date="2026-06-01")
    a, b = old.statements[0], new.statements[0]
    ps = _set(old, new, links=[Link(type=LinkType.SUPERSEDES, src=a.id, dst=b.id)])
    assert resolve_conflict(a, b, ps) is a
    assert resolve_conflict(b, a, ps) is a


def test_newer_date_wins_then_authority_rank():
    old, new = (
        _doc("a", "limit 500", date="2026-01-01", rank=9),
        _doc("b", "limit 300", date="2026-06-01", rank=1),
    )
    ps = _set(old, new)
    assert resolve_conflict(old.statements[0], new.statements[0], ps) is new.statements[0]
    low, high = (
        _doc("c", "limit 500", date="2026-01-01", rank=1),
        _doc("d", "limit 300", date="2026-01-01", rank=5),
    )
    ps = _set(low, high)
    assert resolve_conflict(low.statements[0], high.statements[0], ps) is high.statements[0]


def test_trusted_beats_untrusted_even_with_supersedes():
    trusted, bad = (
        _doc("sop", "limit 500"),
        _doc("email", "limit 5000", trust=Trust.UNTRUSTED, date="2026-09-01"),
    )
    t, u = trusted.statements[0], bad.statements[0]
    ps = _set(trusted, bad, links=[Link(type=LinkType.SUPERSEDES, src=u.id, dst=t.id)])
    assert resolve_conflict(t, u, ps) is t
    assert resolve_conflict(u, t, ps) is t


def test_no_precedence_signal_is_unresolved():
    a, b = (
        _doc("a", "limit 500", date="2026-01-01", rank=2),
        _doc("b", "limit 300", date="2026-01-01", rank=2),
    )
    assert resolve_conflict(a.statements[0], b.statements[0], _set(a, b)) is None


def test_statement_rejects_more_than_two_types():
    with pytest.raises(ValueError):
        Statement(
            id="x", text="t", types=[StatementType.THRESHOLD, StatementType.APPROVAL, StatementType.FORMAT]
        )
