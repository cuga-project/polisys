"""Statement type classifier: tables by code, prose by one model call per section, safe fallbacks."""

import json

from polisys.classify import classify
from polisys.documents import load_document
from polisys.model import Modality, StatementType

MATRIX = "## 3. Priority\n\n| Impact | Urgency | Priority |\n|---|---|---|\n| 1 | 1 | 1 - Critical |\n| 1 | 2 | 2 - High |\n"
PROSE = (
    "## 5. Refunds\n\n- Refunds above $500 need approval by the AP Manager.\n- Never email card numbers.\n"
)


class Fake:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def __call__(self, system, user):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def test_tables_are_typed_without_a_model_call():
    doc, model = load_document(MATRIX), Fake({"statements": []})
    assert classify(doc, model) == []
    assert model.calls == 0
    assert all(st.types == [StatementType.LOOKUP_MATRIX] for st in doc.statements)


def test_prose_types_and_modality_from_model():
    doc = load_document(PROSE)
    model = Fake(
        {
            "statements": [
                {"n": 1, "types": ["threshold", "approval"], "modality": "must"},
                {"n": 2, "types": ["prohibition"], "modality": "must_not"},
            ]
        }
    )
    assert classify(doc, model) == []
    a, b = doc.statements
    assert a.types == [StatementType.THRESHOLD, StatementType.APPROVAL] and a.modality == Modality.MUST
    assert b.types == [StatementType.PROHIBITION] and b.modality == Modality.MUST_NOT


def test_unknown_type_from_model_becomes_open_judgment_with_note():
    doc = load_document(PROSE)
    notes = classify(
        doc, Fake({"statements": [{"n": 1, "types": ["vibes"]}, {"n": 2, "types": ["prohibition"]}]})
    )
    assert doc.statements[0].types == [StatementType.OPEN_JUDGMENT]
    assert [n.kind for n in notes] == ["classify_fallback"]


def test_model_error_or_bad_json_falls_back_to_open_judgment():
    for model in (Fake(error=TimeoutError("slow")), Fake("not json at all")):
        doc = load_document(PROSE)
        notes = classify(doc, model)
        assert all(st.types == [StatementType.OPEN_JUDGMENT] for st in doc.statements)
        assert len(notes) == 2 and all(n.kind == "classify_fallback" for n in notes)
