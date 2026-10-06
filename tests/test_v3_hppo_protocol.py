"""V3-HPPO paper protocol and frozen-result guardrails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repo_paths import resolve_repo_path  # noqa: E402

V3 = ROOT / "results" / "v3"


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
    text = (ROOT / "results" / "v3" / "paper" / "claims" / "CLAIMS.md").read_text(encoding="utf-8")
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
    assert payload.get("no_final_fahppo_checkpoint_retrain_retune_or_reselect_after_test") is True
    raw = resolve_repo_path(payload["raw_file"])
    assert raw.is_file()
    freeze = json.loads((V3 / "CHECKPOINT_FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["n_checkpoints"] == 5
    assert all(item["method"] == "HybridPPO" for item in freeze["checkpoints"])


def test_wording_and_claims_discipline():
    wording = (V3 / "WORDING_CLARIFICATIONS.md").read_text(encoding="utf-8")
    assert "independently generated" in wording
    assert "external domain" in wording.lower() or "external generalization" in wording.lower()
    claims = (ROOT / "results" / "v3" / "paper" / "claims" / "CLAIMS.md").read_text(encoding="utf-8")
    assert "FA-HPPO-Max" in claims
    assert "DiscretePPO" in claims
    assert "feasibility-aware" in claims.lower()


def test_results_paper_png_only_if_present():
    fig = ROOT / "results" / "v3" / "paper" / "figures"
    if not fig.is_dir():
        return
    pngs = list(fig.glob("*.png"))
    assert len(pngs) == 20
    assert not list(p for p in fig.rglob("*.png") if p.parent != fig)
    for path in fig.glob("*.png"):
        assert path.suffix.lower() == ".png", path
    assert (fig / "fig01_usecase_route.png").is_file()
    assert (fig / "fig02_method_schematic.png").is_file()
    assert (fig / "fig05_main_feasibility.png").is_file()
    assert (fig / "fig06_main_completion.png").is_file()
    assert (fig / "fig13_ablation_bars.png").is_file()
    assert (fig / "fig14_amount_policies.png").is_file()
    assert (ROOT / "results" / "v3" / "paper" / "case_study" / "illustrative_val_episode.json").is_file()
    assert (ROOT / "results" / "v3" / "paper" / "CAPTIONS.md").is_file()
    assert (ROOT / "results" / "v3" / "paper" / "MANIFEST.json").is_file()
    assert (ROOT / "results" / "v3" / "paper" / "tables" / "table02_main_results.md").is_file()
    assert (ROOT / "results" / "v3" / "paper" / "tables" / "tableA03_failure_routes.md").is_file()
