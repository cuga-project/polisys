"""Documents and policy sets: Markdown and PDF reading, sections, statements, set metadata."""

import pytest
import re
import shutil
from pathlib import Path


from polisys.documents import document_from_text, load_document, load_policy_set
from polisys.model import Trust

FIX = Path(__file__).parent / "fixtures"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9$§%]+", " ", text.lower()).strip()


def test_markdown_sections_keep_numbered_ids():
    doc = load_document(FIX / "ap_policy.md")
    ids = {s.id for s in doc.sections}
    assert {"preamble", "4", "4.2", "8.1", "8.2", "Appendix B"} <= ids
    sec = next(s for s in doc.sections if s.id == "4.2")
    assert sec.parent == "4"
    step3 = [st for st in sec.statements if st.source.get("step") == 3]
    assert step3 and "three-way match" in step3[0].text
    assert doc.title.startswith("Northwind Corp")


def test_pdf_and_markdown_give_the_same_statements():
    pytest.importorskip("docling")
    md = load_document(FIX / "ap_policy.md")
    pdf = load_document(FIX / "ap_policy.pdf")
    md_texts = {_norm(s.text) for s in md.statements}
    pdf_texts = {_norm(s.text) for s in pdf.statements}
    found = sum(t in pdf_texts for t in md_texts)
    assert found / len(md_texts) >= 0.9
    assert all((s.source.get("page") or 0) >= 1 for s in pdf.statements)
    assert {"4.2", "8.1"} <= {s.id for s in pdf.sections}


def test_table_rows_are_statements():
    doc = load_document(FIX / "ap_policy.md")
    defs = next(s for s in doc.sections if s.id == "2")
    texts = [st.text for st in defs.statements]
    assert any(t.startswith("Emergency payment |") for t in texts)
    assert not any(set(t) <= set("|-: ") for t in texts)  # no separator rows


def test_policy_set_keeps_meta_and_hash(tmp_path):
    for f in (FIX / "ap_set").iterdir():
        shutil.copy(f, tmp_path / f.name)
    files = [tmp_path / "sop.md", tmp_path / "appendix.md", tmp_path / "update_2026.md"]
    meta = {
        "update_2026": {"effective_date": "2026-10-01", "authority_rank": 1},
        "sop": {"effective_date": "2026-07-01", "trust": "trusted"},
    }
    ps = load_policy_set(files, meta)
    assert [d.id for d in ps.documents] == ["sop", "appendix", "update_2026"]
    assert ps.documents[2].effective_date == "2026-10-01" and ps.documents[2].authority_rank == 1
    assert ps.documents[0].trust == Trust.TRUSTED
    before = ps.version
    (tmp_path / "update_2026.md").write_text(
        (tmp_path / "update_2026.md").read_text().replace("$15,000", "$12,000")
    )
    assert load_policy_set(files, meta).version != before


def test_document_from_text_for_online_records():
    doc = document_from_text(
        "Do not schedule units whose Notes contain an FSR- code.",
        kind="sheet_row",
        source={"record_id": "ws_hold:1"},
        trust=Trust.UNTRUSTED,
    )
    assert doc.trust == Trust.UNTRUSTED and len(doc.statements) == 1
    assert doc.statements[0].source["record_id"] == "ws_hold:1"


def test_split_policy_still_importable_from_builder():
    from polisys.builder import Section, split_policy

    secs = split_policy("## 1. Scope\nAll incidents.\n## 2. Rules\n- Never assume urgency.")
    assert [s.sid for s in secs] == ["1", "2"] and isinstance(secs[0], Section)


def test_a_learned_pattern_that_matches_restarting_list_numbers_is_rejected():
    from polisys.structure import _valid_heading

    lines = [
        "## 5.1 Intake",
        "1. Review the claim.",
        "2. Check coverage.",
        "3. Notify the adjuster.",
        "## 5.2 Tax",
        "1. Download the schedule.",
        "2. Determine surtaxes.",
        "## 5.3 Close",
        "1. Archive.",
    ]
    assert (
        _valid_heading({"pattern": r"^(?P<id>\d+(?:\.\d+)*)\.\s+.+$", "level": 1, "kind": "clause"}, lines)
        is None
    )
    assert (
        _valid_heading({"pattern": r"^##\s+(?P<id>\d+\.\d+)\s+(?P<title>.+)$", "level": 2}, lines) is not None
    )


def test_a_clause_pattern_colliding_with_heading_numbers_is_dropped():
    from polisys.structure import _drop_colliding_clauses

    lines = [
        "# 1 Purpose",
        "# 2 Scope",
        "## 5.2 Tax",
        "1. Review the application.",
        "2. Determine surtaxes.",
        "3. Request payroll.",
    ]
    heads = [
        {"pattern": r"^#\s+(?P<id>\d+)\s+(?P<title>.+)$", "level": 1, "kind": "heading"},
        {"pattern": r"^(?P<id>\d+(?:\.\d+)*)\.\s+.+$", "level": 1, "kind": "clause"},
    ]
    assert [h["kind"] for h in _drop_colliding_clauses(heads, lines)] == ["heading"]
    contract = ["1. The tenant shall pay rent.", "2. The landlord shall repair.", "3. Either party may end."]
    assert _drop_colliding_clauses(heads[1:], contract) == heads[1:]  # numbered clauses with no headings stay


def test_a_title_before_a_name_does_not_end_a_sentence():
    from polisys.documents import sentences

    text = "Send the notice to Dr. Rivera and to Mr. Hall. The notice names Prof. Lee. Ms. Cho signs it."
    assert sentences(text) == [
        "Send the notice to Dr. Rivera and to Mr. Hall.",
        "The notice names Prof. Lee.",
        "Ms. Cho signs it.",
    ]


def test_dashes_of_an_html_document_stay_and_a_leading_dash_is_still_a_bullet(tmp_path):
    pytest.importorskip("docling")
    """docling folds en and em dashes to a hyphen. They are kept, except one that starts a line, which stays the hyphen it
    always became (a list bullet), so no statement or id moves."""
    from polisys.documents import _hold_dashes, _restore_dashes

    held = _hold_dashes("Mon &ndash; Fri — open &#8211; late")
    assert "–" not in held and "&ndash;" not in held and "—" not in held
    assert _restore_dashes(" item\nopenlate  x") == "- item\nopen–late — x"
    page = tmp_path / "sop.html"
    page.write_text(
        "<html><body><h2>1. Hours</h2><p>Open Monday \u2013 Friday, 8am \u2014 5pm.</p>"
        "<p>\u2013 Lock the door.</p><ul><li>Call the lead \u2013 now.</li></ul></body></html>",
        encoding="utf-8",
    )
    doc = load_document(page)
    texts = [st.text for st in doc.statements]
    assert "Open Monday \u2013 Friday, 8am \u2014 5pm." in texts
    assert (
        "Lock the door." in texts and "Call the lead \u2013 now." in texts
    )  # a leading dash is the bullet it was


def test_html_that_is_not_utf8_is_read_as_it_is():
    from polisys.documents import _held_html

    assert _held_html("<p>a – b</p>".encode("cp1252")) is None  # not UTF-8
    assert _held_html('<meta charset="windows-1252"><p>a &ndash; b</p>'.encode()) is None  # another charset
    assert _held_html(b"<p>no dash</p>") is None
    held = _held_html("<p>a – b</p>".encode())
    assert held.startswith(b'<meta charset="utf-8">') and "–".encode() not in held
