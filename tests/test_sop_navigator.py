"""SopNavigator: the walk follows reference links, and a hand-off gets a note only for open action statements."""

from polisys.sop_navigator import SopNavigator

POLICY = """# Messages
## Fix messages
Use check_messages() to check the app. Ensure data works; refer to the Data section.
## Message permissions
Use grant_permission() to grant the missing permission.
# Data
## Roaming
If the user is abroad, guide them to use toggle_roaming().
Check that the line is roaming enabled.
## Saver
Use toggle_saver() to turn the data saver off.
"""


def _st(sec, n, text):
    return {"id": f"doc:{sec}:{n}", "text": text, "source": {"document": "doc", "section": sec}}


BUILD = {
    "policy_set": {
        "documents": [
            {
                "id": "doc",
                "sections": [
                    {"id": "messages", "title": "Messages", "statements": []},
                    {
                        "id": "fix-messages",
                        "title": "Fix messages",
                        "statements": [
                            _st("fix-messages", 1, "Use check_messages() to check the app."),
                            _st("fix-messages", 2, "Ensure data works; refer to the Data section."),
                        ],
                    },
                    {
                        "id": "message-permissions",
                        "title": "Message permissions",
                        "statements": [_st("message-permissions", 1, "Use grant_permission() to grant it.")],
                    },
                    {"id": "data", "title": "Data", "statements": []},
                    {
                        "id": "roaming",
                        "title": "Roaming",
                        "statements": [
                            _st("roaming", 1, "If the user is abroad, guide them to use toggle_roaming()."),
                            _st("roaming", 2, "Check that the line is roaming enabled."),
                        ],
                    },
                    {
                        "id": "saver",
                        "title": "Saver",
                        "statements": [_st("saver", 1, "Use toggle_saver() to turn it off.")],
                    },
                ],
            }
        ]
    },
    "build": {
        "links": [{"type": "refers_to", "src": "doc:fix-messages:2", "dst": "doc:data"}],
        "procedures": [
            {
                "name": "p",
                "escalations": [
                    {"when": "if the permission cannot be granted", "from_section": "message-permissions"}
                ],
            }
        ],
    },
}


def fake_ask(solved=0.0, exit_p=0.0):
    def ask(state, questions):
        out = {}
        for name, q in questions.items():
            if name == "solved":
                out[name] = solved
            elif name.startswith("e"):
                out[name] = exit_p
            else:
                out[name] = 0.9 if "Messages" in q else 0.1
        return out

    return ask


TURNS = [
    {"role": "customer", "text": "My picture messages do not send."},
    {"role": "agent", "text": "Please run check_messages() and grant_permission()."},
    {"role": "customer", "text": "Done, still failing."},
]


def test_walk_follows_reference_links():
    nav = SopNavigator(
        POLICY,
        BUILD,
        min_chapter_actions=1,
        agent_tools=["enable_roaming", "transfer_to_human_agents"],
        ask=fake_ask(),
    )
    root = nav.choose_chapter("My picture messages do not send.")
    titles = [nav.secs[i]["title"] for i in nav.walk(root)]
    assert titles[:3] == ["Messages", "Fix messages", "Message permissions"]
    assert {"Data", "Roaming", "Saver"} <= set(titles)
    assert [
        nav.secs[i]["title"]
        for i in SopNavigator(
            POLICY, BUILD, min_chapter_actions=1, agent_tools=[], ask=fake_ask(), depth=0
        ).walk(root)
    ] == titles[:3]


def test_handoff_lists_open_actions_from_linked_sections():
    nav = SopNavigator(POLICY, BUILD, min_chapter_actions=1, agent_tools=["enable_roaming"], ask=fake_ask())
    note = nav.check_handoff(TURNS, {"summary": "tried everything"})
    assert note and "toggle_roaming()" in note and "roaming enabled" in note and "toggle_saver()" in note
    assert "check_messages()" not in note  # named by the agent: handled
    assert nav.records[-1]["pending"] == ["doc:roaming:1", "doc:roaming:2", "doc:saver:1"]


def test_no_note_when_solved_or_exit_holds_or_all_done():
    assert (
        SopNavigator(
            POLICY, BUILD, min_chapter_actions=1, agent_tools=["enable_roaming"], ask=fake_ask(solved=0.9)
        ).check_handoff(TURNS)
        is None
    )
    assert (
        SopNavigator(
            POLICY, BUILD, min_chapter_actions=1, agent_tools=["enable_roaming"], ask=fake_ask(exit_p=0.8)
        ).check_handoff(TURNS)
        is None
    )
    done = TURNS + [
        {"role": "agent", "text": "Use toggle_roaming() and toggle_saver()."},
        {"role": "tool", "name": "enable_roaming", "result": "ok"},
    ]
    assert (
        SopNavigator(
            POLICY, BUILD, min_chapter_actions=1, agent_tools=["enable_roaming"], ask=fake_ask()
        ).check_handoff(done)
        is None
    )
