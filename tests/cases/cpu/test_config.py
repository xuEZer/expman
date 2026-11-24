import os
import subprocess
import sys
import warnings
from datetime import date

import pytest

from expman import ConfigError, MissingConfigWarning
from expman.config.loader import _load_configs


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def experiment(root):
    return root / "experiment.yaml"


@pytest.fixture
def load(root, experiment):
    def _load(content, files=None):
        for name, body in (files or {}).items():
            write(root, name, body)
        write(root, "experiment.yaml", content)
        return _load_configs(experiment)

    return _load


def test_plain_config_and_lists_produce_one_run(load):
    assert load("seed: 42\nhidden_sizes: [128, 64, 32]\noptional: null\n") == [
        {"seed": 42, "hidden_sizes": [128, 64, 32], "optional": None}
    ]
    assert load("{}") == [{}]


def test_date_and_string_variants_remain_distinct(load):
    runs = load("day: x\ngrid:\n  day: [2026-01-01, '2026-01-01', 2026-01-01]\n")
    assert [run["day"] for run in runs] == [date(2026, 1, 1), "2026-01-01"]


def test_set_dependency_digest_is_stable_across_worker_hash_seeds():
    script = (
        "from expman.core.dependencies import observation\n"
        "print(observation({'tags': set('abcdefgh')}, ('tags',)))"
    )
    digests = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ("1", "2")
    ]
    assert digests[0] == digests[1]


def test_grid_multiplies_axes_in_declaration_order_and_replaces_subtrees(load):
    runs = load("""
model: {name: old, width: 32}
seed: 7
grid:
  seed: [0, 1]
  model: [{name: a}, {name: b}]
""")
    assert runs == [
        {"seed": 0, "model": {"name": "a"}},
        {"seed": 0, "model": {"name": "b"}},
        {"seed": 1, "model": {"name": "a"}},
        {"seed": 1, "model": {"name": "b"}},
    ]


def test_a_leaf_can_hold_lists_and_null(load):
    assert load("""
widths: [128]
grid:
  widths: [[128, 64], [32], null]
""") == [
        {"widths": [128, 64]},
        {"widths": [32]},
        {"widths": None},
    ]


def test_subtree_choice_replaces_inherited_defaults_before_scanning(load):
    runs = load(
        "defaults: original\ngrid: {model: [{defaults: chosen}]}\nscan: {model: {scale: [1, 2]}}",
        {
            "configs/original.yaml": "model: {old: true}",
            "configs/model/chosen.yaml": "width: 8",
        },
    )
    assert runs == [{"model": {"width": 8, "scale": value}} for value in (1, 2)]


def test_axis_patch_changes_only_the_selected_path_of_a_yaml_alias(load):
    assert load("left: &item {x: 0}\nright: *item\ngrid: {left: {x: [1]}}") == [
        {"left": {"x": 1}, "right": {"x": 0}}
    ]


def test_scan_changes_one_axis_at_a_time_in_declaration_order(load):
    runs = load("""
fixed: 7
model: {name: old, width: 32}
scan:
  optimizer: {lr: [0.01, 0.1, 0.1]}
  model: [{name: a}, {name: b}]
""")
    assert runs == [
        {"fixed": 7, "optimizer": {"lr": 0.01}, "model": {"name": "a"}},
        {"fixed": 7, "optimizer": {"lr": 0.1}, "model": {"name": "a"}},
        {"fixed": 7, "optimizer": {"lr": 0.01}, "model": {"name": "b"}},
    ]


def test_combines_merges_distinct_paired_choices_over_the_base(load):
    runs = load("""
dataset: {name: A, rows: 10}
model: {name: a}
combines:
  - {}
  - {dataset: {name: B}, model: {name: b}}
  - {}
""")
    assert runs == [
        {"dataset": {"name": "A", "rows": 10}, "model": {"name": "a"}},
        {"dataset": {"name": "B", "rows": 10}, "model": {"name": "b"}},
    ]


def test_all_three_layers_nest_from_the_outside_in(load):
    runs = load("""
stage: B1
seed: 42
combines:
  - {stage: A}
  - {stage: B}
grid:
  seed: [0, 1]
scan:
  rank: [2, 4]
""")
    assert [(run["stage"], run["seed"], run["rank"]) for run in runs] == [
        ("A", 0, 2),
        ("A", 0, 4),
        ("A", 1, 2),
        ("A", 1, 4),
        ("B", 0, 2),
        ("B", 0, 4),
        ("B", 1, 2),
        ("B", 1, 4),
    ]


def test_a_run_produced_by_two_layers_is_not_repeated(load):
    runs = load("""
a: 0
combines:
  - {a: 1}
  - {a: 2}
grid:
  a: [1, 0]
""")
    assert [run["a"] for run in runs] == [1, 0]


def test_deduplication_happens_after_defaults_are_resolved(load):
    runs = load(
        """
combines:
  - {model: {defaults: a}}
  - {model: {kind: saits, lr: 0.001}}
""",
        {"configs/model/a.yaml": "kind: saits\nlr: 0.001\n"},
    )
    assert runs == [{"model": {"kind": "saits", "lr": 0.001}}]


def test_a_grid_axis_must_exist_in_the_document(load):
    with pytest.raises(ConfigError, match=r"grid\.tip\.missing"):
        load("tip: {rank: 1}\ngrid:\n  tip: {missing: [1]}\n")


def test_a_grid_axis_can_target_a_field_the_combines_introduced(load):
    runs = load("""
tip: {rank: 1}
combines:
  - {tip: {rank: 1, epsilon: 0.5}}
grid:
  tip: {epsilon: [0.5, 0.25]}
""")
    assert [run["tip"]["epsilon"] for run in runs] == [0.5, 0.25]


def test_an_axis_can_target_a_field_supplied_by_defaults(load):
    runs = load(
        "model: {defaults: a}\ngrid:\n  model: {depth: [2, 8]}\n",
        {"configs/model/a.yaml": "name: a\nwidth: 64\ndepth: 2\n"},
    )
    assert [(run["model"]["width"], run["model"]["depth"]) for run in runs] == [
        (64, 2),
        (64, 8),
    ]


def test_a_scan_axis_overrides_a_field_supplied_by_defaults(load):
    runs = load(
        "model: {defaults: a}\nscan:\n  model: {depth: [2, 8]}\n",
        {"configs/model/a.yaml": "name: a\nwidth: 64\ndepth: 32\n"},
    )
    assert [
        (run["model"]["name"], run["model"]["width"], run["model"]["depth"])
        for run in runs
    ] == [("a", 64, 2), ("a", 64, 8)]


def test_only_the_root_control_keys_are_control_keys(load):
    assert load("tip: {grid: 1, scan: 2, combines: 3}\n") == [
        {"tip": {"grid": 1, "scan": 2, "combines": 3}}
    ]


@pytest.mark.parametrize(
    "content",
    [
        "grid: null",
        "scan: [1]",
        "combines: {a: [1]}",
        "grid: {}",
        "a: 0\ngrid: {a: []}",
        "a: 0\ngrid: {a: {nested: [1]}}",
        "scan: {a: 1}",
        "combines: []",
        "combines: [{a: 1}, 2]",
        "device: [0]\ngrid: {device: [[0], [1]]}",
        "device: [0]\nscan: {device: [[0], [1]]}",
    ],
)
def test_malformed_control_trees_are_rejected(load, content):
    with pytest.raises(ConfigError):
        load(content)


@pytest.mark.parametrize("key", ["combines", "grid", "scan"])
def test_control_keys_are_rejected_in_defaults_files(root, load, key):
    path = write(root, "configs/model/a.yaml", f"{key}: {{x: [1]}}")
    with pytest.raises(ConfigError) as raised:
        load("model: {defaults: a}\nx: 0")
    assert str(path) in str(raised.value)


def test_defaults_merge_recursively_and_never_leak_into_the_run(root, load):
    defaults = write(
        root,
        "configs/model/a.yaml",
        """
optimizer:
  lr: 0.001
  weight_decay: 0.01
layers: [128, 64]
optional: {enabled: true}
replace_mapping: {value: 1}
replace_scalar: 1
""",
    )
    original = defaults.read_bytes()
    runs = load("""
model:
  defaults: a
  optimizer:
    lr: 0.0005
  layers: [32]
  optional: null
  replace_mapping: false
  replace_scalar: {value: 2}
""")
    assert runs[0]["model"] == {
        "optimizer": {"lr": 0.0005, "weight_decay": 0.01},
        "layers": [32],
        "optional": None,
        "replace_mapping": False,
        "replace_scalar": {"value": 2},
    }
    assert defaults.read_bytes() == original


def test_nested_defaults_are_resolved(load):
    runs = load(
        "model: {defaults: a}",
        {
            "configs/model/a.yaml": "optimizer: {defaults: adam}\nname: a\n",
            "configs/model/optimizer/adam.yaml": "lr: 0.001",
        },
    )
    assert runs[0]["model"] == {
        "name": "a",
        "optimizer": {"lr": 0.001},
    }


def test_root_defaults_follow_the_same_convention(load):
    assert load(
        "defaults: demo\nvalue: 3",
        {
            "configs/demo.yaml": "defaults: base\nvalue: 2",
            "configs/base.yaml": "value: 1\nother: 4",
        },
    ) == [{"value": 3, "other": 4}]


def test_same_node_defaults_cycle_is_rejected(load):
    with pytest.raises(ConfigError, match="cyclic defaults"):
        load(
            "defaults: a",
            {"configs/a.yaml": "defaults: b", "configs/b.yaml": "defaults: a"},
        )


def test_device_validation_uses_the_resolved_configurations(load):
    with pytest.raises(ConfigError, match="same for every experiment"):
        load(
            "combines: [{defaults: first}, {defaults: second}]",
            {"configs/first.yaml": "device: [1]", "configs/second.yaml": "device: [2]"},
        )


def test_defaults_use_working_directory_and_field_keys(root):
    path = write(root, "nested/experiment.yaml", "items: [{defaults: a}]\n")
    write(root, "nested/pyproject.toml", "")
    write(root, "nested/configs/items/a.yaml", "value: 99")
    with pytest.warns(MissingConfigWarning):
        assert _load_configs(path) == [{"items": [{}]}]
    write(root, "configs/items/a.yaml", "value: 42")
    assert _load_configs(str(path)) == [{"items": [{"value": 42}]}]


def test_missing_defaults_warn_once_per_file_and_preserve_explicit(root, load):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = load("""
seed: 0
model: {defaults: missing, width: 32}
grid:
  seed: [0, 1, 2]
""")
    assert len(caught) == 1
    assert caught[0].category is MissingConfigWarning
    assert str(root / "configs/model/missing.yaml") in str(caught[0].message)
    assert "$.model.defaults" in str(caught[0].message)
    assert all(run["model"] == {"width": 32} for run in runs)


def test_runs_and_aliases_are_independent(load):
    runs = load(
        "seed: 0\nleft: &shared {items: [1, 2]}\nright: *shared\n"
        "model: {defaults: a}\ngrid:\n  seed: [0, 1]\n",
        {"configs/model/a.yaml": "layers: [128, 64]\n"},
    )
    runs[0]["left"]["items"].append(3)
    runs[0]["model"]["layers"].append(32)
    assert runs[0]["right"]["items"] == [1, 2]
    assert runs[1]["left"]["items"] == [1, 2]
    assert runs[1]["model"]["layers"] == [128, 64]


def test_the_defaults_cache_is_local_to_each_load(root, load, experiment):
    path = write(root, "configs/model/a.yaml", "value: 1")
    assert load("model: {defaults: a}")[0]["model"]["value"] == 1
    path.write_text("value: 2")
    assert _load_configs(experiment)[0]["model"]["value"] == 2


def test_name_field_does_not_trigger_defaults_lookup(load):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = load(
            "model: {name: a}\ndataset: {name: missing}",
            {"configs/model/a.yaml": "width: 64"},
        )
    assert runs == [{"model": {"name": "a"}, "dataset": {"name": "missing"}}]
    assert caught == []


@pytest.mark.parametrize(
    "content",
    [
        "",
        "[1, 2]",
        "x: [",
        "x: {a: 1, a: 2}",
        "1: value",
        "x: !unknown [1]",
        "x: &a {nested: *a}",
        "x: {<<: {a: 1}}",
        "x: !!python/object:builtins.object {}",
        "x: 1\n---\nx: 2",
    ],
)
def test_invalid_documents_report_source(load, experiment, content):
    with pytest.raises(ConfigError) as raised:
        load(content)
    assert str(experiment) in str(raised.value)


def test_malformed_defaults_are_errors_not_missing_warnings(root, load):
    path = write(root, "configs/model/a.yaml", "x: [")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigError) as raised:
            load("model: {defaults: a}")
    assert str(path) in str(raised.value)
    assert caught == []


def test_missing_experiment_is_an_error(experiment):
    with pytest.raises(ConfigError) as raised:
        _load_configs(experiment)
    assert str(experiment) in str(raised.value)


@pytest.mark.parametrize(
    "name",
    ["42", "''", "'../outside'", "'a\\b'", "'.'"],
)
def test_invalid_defaults_names_are_rejected(load, name):
    with pytest.raises(ConfigError):
        load(f"model: {{defaults: {name}}}")


def test_a_traversal_key_is_rejected(load):
    with pytest.raises(ConfigError):
        load("'../model': {defaults: a}")


def test_a_directory_instead_of_a_defaults_file_is_an_error(root, load):
    path = root / "configs/model/a.yaml"
    path.mkdir(parents=True)
    with pytest.raises(ConfigError) as raised:
        load("model: {defaults: a}")
    assert str(path) in str(raised.value)


def test_a_defaults_symlink_cannot_escape_the_config_root(root, load):
    outside = write(root, "outside.yaml", "secret: 1")
    link = root / "configs/model/a.yaml"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ConfigError, match="outside"):
        load("model: {defaults: a}")


def test_recursive_defaults_through_a_directory_alias_are_rejected(root, load):
    path = write(root, "configs/model/a.yaml", "child: {defaults: a}")
    try:
        (path.parent / "child").symlink_to(path.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ConfigError, match="cyclic defaults"):
        load("model: {defaults: a}")
