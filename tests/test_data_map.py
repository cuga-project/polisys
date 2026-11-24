"""sources.data_map: the brief says where records exist, built by code from the listing."""

from polisys.sources.listing import world_map


def test_world_map_lists_tables_records_and_empty_sources():
    lines = world_map({
        "sheets": [{"spreadsheet_id": "ss_1", "title": "Queue", "rows": [{"cells": {"a": 1, "b": 2}}, {"cells": {"a": 3, "b": 4}}]}],
        "mail": [{"id": "m1", "subject": "Hello"}, {"id": "m2", "subject": "Rules"}],
        "tracker": [],
    })
    assert lines[0] == "- mail: 2 records: 'Hello' (m1); 'Rules' (m2)"
    assert lines[1] == "- sheets: table spreadsheet_id=ss_1, title=Queue: 2 rows; columns: a, b"
    assert lines[2] == "- tracker: no records"


def test_world_map_is_capped():
    lines = world_map({f"app{i:02d}": [{"id": str(i), "subject": "x" * 50}] for i in range(80)})
    assert len(lines) <= 40 and sum(len(x) + 1 for x in lines) <= 3000
