"""code_git_dirty ignores approved experiment output namespaces."""

from __future__ import annotations

from experiments.provenance import _is_approved_output, code_dirty_paths, run_manifest


def test_approved_output_prefixes():
    assert _is_approved_output("results/v4_reward/final_authoritative/x.json")
    assert _is_approved_output("checkpoints_v4/final_authoritative/V4/seed_42/best.pt")
    assert _is_approved_output("results_v4/figures/fig.png")
    assert not _is_approved_output("src/rl/ppo.py")
    assert not _is_approved_output("scripts/v4_reward/run_final_clean.py")


def test_run_manifest_exposes_code_git_dirty():
    payload = run_manifest(method="unit_test")
    assert "code_git_dirty" in payload
    assert "training_git_sha" in payload
    assert payload["git_dirty"] == payload["code_git_dirty"]
    assert isinstance(code_dirty_paths(), list)
