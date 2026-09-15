"""Parent-level splits: 20/6/6, seed 42, Large repair, no leakage."""

from experiments.split import assign_splits, assert_no_leakage, load_parents, write_splits
from routing.serialize import read_jsonl
from data.paths import ROUTES_DIR


def test_split_counts_and_large_repair(raw_root):
    parents = load_parents(raw_root)
    assignment = assign_splits(parents, seed=42)
    assert len(assignment.train) == 20
    assert len(assignment.validation) == 6
    assert len(assignment.test) == 6
    n_inst = lambda names: sum(assignment.parents[p].n_instances for p in names)
    assert n_inst(assignment.train) == 73
    assert n_inst(assignment.validation) == 30
    assert n_inst(assignment.test) == 21
    large_train = {p for p in assignment.train if assignment.parents[p].has_large}
    large_val = {p for p in assignment.validation if assignment.parents[p].has_large}
    large_test = {p for p in assignment.test if assignment.parents[p].has_large}
    assert large_train == {"c102", "r107"}
    assert large_val == {"r102"}
    assert large_test == {"c101"}
    assert_no_leakage(assignment)


def test_zero_leakage_including_routes(raw_root, tmp_path):
    parents = load_parents(raw_root)
    assignment = assign_splits(parents, seed=42)
    routes = None
    corpus = ROUTES_DIR / "corpus.jsonl"
    if corpus.is_file():
        routes = read_jsonl(corpus)
    assert_no_leakage(assignment, routes=routes)
    write_splits(assignment, tmp_path)
    assert (tmp_path / "train.json").is_file()
    assert (tmp_path / "validation.json").is_file()
    assert (tmp_path / "test.json").is_file()
    assert (tmp_path / "split_manifest.csv").is_file()
    assert (tmp_path / "split_metadata.json").is_file()
