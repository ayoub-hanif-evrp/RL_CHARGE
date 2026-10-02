"""V3-HPPO paper protocol and frozen-result guardrails."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "v3_hppo"


def test_protocol_exists_and_excludes_dppo():
    protocol = json.loads((V3 / "PAPER_PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["written_before_test_evaluation"] is True
    assert "DiscretePPO" in protocol["excluded_from_v3_test"]
    assert "HybridPPO" in protocol["methods_on_test"]
    assert "DiscretePPO" not in protocol["methods_on_test"]
    assert protocol["confirmatory_test"]["generator_seed_start"] == 400000
    assert protocol["confirmatory_test"]["n_routes_target"] == 180


def test_v2_final_untouched_marker_files():
    final = ROOT / "results" / "v2" / "final"
    for name in (
        "FINAL_PROTOCOL.json",
        "METHOD_FREEZE.json",
        "CHECKPOINT_FREEZE.json",
        "TEST_LOCK.json",
        "EVALUATION_CONSUMED.json",
    ):
        assert (final / name).is_file(), name


def test_paper_claims_file():
    text = (ROOT / "paper" / "CLAIMS.md").read_text(encoding="utf-8")
    assert "MUST NOT" in text
    assert "frvcpy" in text.lower()
    assert "customer order" in text.lower() or "customer ordering" in text.lower()


def test_consumed_guard_if_present():
    consumed = V3 / "EVALUATION_CONSUMED.json"
    if not consumed.is_file():
        return
    payload = json.loads(consumed.read_text(encoding="utf-8"))
    assert payload.get("consumed") is True
    assert "DiscretePPO" not in payload.get("methods", [])
    raw = ROOT / payload["raw_file"]
    assert raw.is_file()
    freeze = json.loads((V3 / "CHECKPOINT_FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["n_checkpoints"] == 5
    assert all(item["method"] == "HybridPPO" for item in freeze["checkpoints"])
