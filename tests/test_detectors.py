"""Tier 1 detector client (decision model over OpenRouter systemone), without network."""

import io
import json
import urllib.error

from polisys import detectors


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_systemone_detector_asks_one_noul_question_per_record(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        sent.append(body)
        p = 0.9 if "10k+" in body["state"] else 0.05
        return _Resp(
            json.dumps({"answers": {"is_policy": {"noul": p}}, "usage": {"cost": 0.000002}}).encode()
        )

    monkeypatch.setattr(detectors.urllib.request, "urlopen", fake_urlopen)
    detect = detectors.systemone_detector(model="jaredpalmer/kev-4b", api_key="k")
    scores = detect("answer the mentions", ['{"body": "10k+ followers: reply"}', '{"body": "Pizza"}'])
    assert scores == [0.9, 0.05]
    assert {b["model"] for b in sent} == {"jaredpalmer/kev-4b"}
    assert all(b["questions"]["is_policy"]["type"] == "noul" for b in sent)
    assert detect.usage["calls"] == 2


def test_systemone_detector_fails_toward_extraction(monkeypatch):
    def broken(req, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(detectors.urllib.request, "urlopen", broken)
    monkeypatch.setattr(detectors.time, "sleep", lambda s: None)
    detect = detectors.systemone_detector(api_key="k")
    assert detect("r", ['{"body": "x"}']) == [1.0]
    assert detect.usage["errors"] == 1
