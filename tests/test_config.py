import warnings

import pytest

from expman import ConfigError, MissingConfigWarning, load_configs


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pyproject.toml").touch()
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
        return load_configs(experiment)

    return _load


def test_plain_config_and_lists_produce_one_run(load):
    assert load("seed: 42\nhidden_sizes: [128, 64, 32]\noptional: null\n") == [
        {"seed": 42, "hidden_sizes": [128, 64, 32], "optional": None}
    ]
    assert load("{}") == [{}]


def test_a_grid_multiplies_its_axes(load):
    runs = load("""
a: 9
b: 9
grid:
  a: [1, 2]
  b: [3, 4]
""")
    assert [(run["a"], run["b"]) for run in runs] == [
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]
    assert all("grid" not in run for run in runs)


def test_grid_order_follows_the_tree_not_the_data_field_order(load):
    tree = "grid:\n  x: [10, 50]\n  y: [20, 60]\n"
    first = load("x: 1\ny: 2\n" + tree)
    second = load("y: 2\nx: 1\n" + tree)
    assert [(run["x"], run["y"]) for run in first] == [
        (10, 20),
        (10, 60),
        (50, 20),
        (50, 60),
    ]
    assert first == second


def test_a_grid_replaces_the_documented_value(load):
    runs = load("lr: 0.5\ngrid:\n  lr: [0.001, 0.01]\n")
    assert [run["lr"] for run in runs] == [0.001, 0.01]


def test_repeated_alternatives_do_not_run_twice(load):
    runs = load("a: 0\nb: 0\ngrid:\n  a: [1, 1]\n  b: [2, 2]\n")
    assert [(run["a"], run["b"]) for run in runs] == [(1, 2)]


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


def test_a_subtree_is_swapped_as_a_whole(load):
    runs = load("""
models:
  imputation: {kind: saits, lr: 0.001}
grid:
  models:
    imputation:
      - {kind: saits, lr: 0.0001}
      - {kind: brits}
""")
    assert [run["models"]["imputation"] for run in runs] == [
        {"kind": "saits", "lr": 0.0001},
        {"kind": "brits"},
    ]


def test_a_subtree_axis_multiplies_with_a_leaf_axis(load):
    runs = load("""
seed: 0
model: {kind: saits}
grid:
  seed: [0, 1]
  model:
    - {kind: saits}
    - {kind: brits}
""")
    assert [(run["seed"], run["model"]["kind"]) for run in runs] == [
        (0, "saits"),
        (0, "brits"),
        (1, "saits"),
        (1, "brits"),
    ]


def test_a_scan_starts_at_the_first_value_of_every_axis(load):
    runs = load("""
tip:
  cone: 0.03
scan:
  tip:
    rank: [2, 4]
    epsilon: [0.00001, 0.001]
""")
    assert [
        (run["tip"]["rank"], run["tip"]["epsilon"], run["tip"]["cone"]) for run in runs
    ] == [
        (2, 0.00001, 0.03),
        (4, 0.00001, 0.03),
        (2, 0.001, 0.03),
    ]
    assert all("scan" not in run for run in runs)


def test_a_scan_axis_takes_its_first_value_over_the_file(load):
    runs = load("seed: 7\nhidden: 64\nscan:\n  seed: [0, 1]\n")
    assert [(run["seed"], run["hidden"]) for run in runs] == [(0, 64), (1, 64)]


def test_a_scan_declares_parameters_absent_from_the_file(load):
    runs = load("""
device: [0]
model: {kind: saits}
scan:
  optimizer: {lr: [0.001, 0.01], weight_decay: [0, 0.01]}
""")
    assert [
        (run["optimizer"]["lr"], run["optimizer"]["weight_decay"]) for run in runs
    ] == [
        (0.001, 0),
        (0.01, 0),
        (0.001, 0.01),
    ]
    assert all(run["model"] == {"kind": "saits"} for run in runs)


def test_a_scan_axis_can_swap_a_whole_subtree(load):
    runs = load("""
seed: 0
model: {kind: saits}
scan:
  model:
    - {kind: brits}
    - {kind: saits, lr: 0.001}
""")
    assert [run["model"] for run in runs] == [
        {"kind": "brits"},
        {"kind": "saits", "lr": 0.001},
    ]
    assert all(run["seed"] == 0 for run in runs)


def test_repeated_scan_values_do_not_run_twice(load):
    assert load("scan:\n  seed: [1, 1]\n") == [{"seed": 1}]


def test_a_scan_follows_the_tree_not_the_data_field_order(load):
    tree = "scan:\n  x: [10, 50]\n  y: [20, 60]\n"
    first = load(tree)
    second = load("y: 0\nx: 0\n" + tree)
    assert [(run["x"], run["y"]) for run in first] == [
        (10, 20),
        (50, 20),
        (10, 60),
    ]
    assert first == second


def test_combines_pairs_fields_that_move_together(load):
    runs = load("""
stage: B2
combines:
  - {dataset: {name: A}, model: {name: a}}
  - {dataset: {name: B}, model: {name: b}}
""")
    assert [
        (run["stage"], run["dataset"]["name"], run["model"]["name"]) for run in runs
    ] == [("B2", "A", "a"), ("B2", "B", "b")]
    assert all("combines" not in run for run in runs)


def test_a_combine_candidate_is_merged_over_the_file(load):
    runs = load("""
stage: B1
dataset: {name: A, rows: 10}
combines:
  - {dataset: {name: B}}
""")
    assert runs == [{"stage": "B1", "dataset": {"name": "B", "rows": 10}}]


def test_a_combine_candidate_that_matches_the_file_still_runs(load):
    runs = load("""
stage: B1
variant: A
combines:
  - {stage: B1, variant: A}
  - {stage: B1, variant: B}
""")
    assert [run["variant"] for run in runs] == ["A", "B"]


def test_repeated_candidates_do_not_run_twice(load):
    runs = load("""
stage: B1
combines:
  - {stage: B1}
  - {stage: B1, extra: true}
  - {stage: B1}
""")
    assert runs == [{"stage": "B1"}, {"stage": "B1", "extra": True}]


def test_combines_and_scan_multiply(load):
    runs = load("""
stage: B1
seed: 42
combines:
  - {stage: A}
  - {stage: B}
scan:
  seed: [0, 1]
""")
    assert [(run["stage"], run["seed"]) for run in runs] == [
        ("A", 0),
        ("A", 1),
        ("B", 0),
        ("B", 1),
    ]


def test_combines_and_grid_multiply(load):
    runs = load("""
a: 0
b: 0
combines:
  - {a: 1}
  - {a: 2}
grid:
  b: [1, 2]
""")
    assert [(run["a"], run["b"]) for run in runs] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_grid_and_scan_multiply(load):
    runs = load("""
seed: 0
tip: {rank: 3, epsilon: 0.5}
grid:
  seed: [0, 1]
scan:
  tip:
    rank: [2, 4]
""")
    assert [
        (run["seed"], run["tip"]["rank"], run["tip"]["epsilon"]) for run in runs
    ] == [
        (0, 2, 0.5),
        (0, 4, 0.5),
        (1, 2, 0.5),
        (1, 4, 0.5),
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
  - {a: 1}
grid:
  a: [1, 0]
""")
    assert [run["a"] for run in runs] == [1, 0]


def test_deduplication_happens_after_defaults_are_resolved(load):
    runs = load(
        """
model: {defaults: a}
combines:
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


def test_an_axis_can_target_a_field_supplied_by_defaults(root, load):
    runs = load(
        "model: {defaults: a}\ngrid:\n  model: {depth: [2, 8]}\n",
        {"configs/model/a.yaml": "name: a\nwidth: 64\ndepth: 2\n"},
    )
    assert [(run["model"]["width"], run["model"]["depth"]) for run in runs] == [
        (64, 2),
        (64, 8),
    ]


def test_a_scan_axis_overrides_a_field_supplied_by_defaults(root, load):
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
        "a: 0\ngrid: []\n",
        "a: 0\ngrid: 5\n",
        "a: 0\ngrid: {}\n",
        "a: 0\ngrid: {a: []}\n",
        "a: 0\ngrid: {a: 1}\n",
        "a: 0\ngrid: {missing: [1]}\n",
        "a: 0\ngrid: {a: {nested: 1}}\n",
        "a: 0\nscan: 5\n",
        "a: 0\nscan: {}\n",
        "a: 0\nscan: {a: []}\n",
        "a: 0\nscan: {a: 1}\n",
        "a: 0\nscan: {a: {nested: 1}}\n",
        "a: 0\nscan: []\n",
        "a: 0\nscan: [1]\n",
        "a: 0\nscan: [{a: [1]}, 2]\n",
        "a: 0\ncombines: 5\n",
        "a: 0\ncombines: {a: [1]}\n",
        "a: 0\ncombines: []\n",
        "a: 0\ncombines: [1]\n",
        "a: 0\ncombines: [{a: [1]}, 2]\n",
        "device: [0]\ngrid: {device: [[0], [1]]}\n",
        "device: [0]\nscan: {device: [[0], [1]]}\n",
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


def test_nested_defaults_are_resolved(root, load):
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


def test_root_defaults_follow_the_same_convention(root, load):
    assert load("defaults: demo", {"configs/demo.yaml": "value: 1\n"}) == [{"value": 1}]


def test_defaults_paths_follow_the_project_root_and_field_keys(root):
    path = write(root, "nested/experiment.yaml", "items: [{defaults: a}]\n")
    write(root, "configs/items/a.yaml", "value: 42")
    assert load_configs(str(path)) == [{"items": [{"value": 42}]}]


@pytest.mark.parametrize(
    "location",
    ["configs/run.yaml", "configs/experiments/run.yaml", "experiments/run.yaml"],
)
def test_data_defaults_use_the_project_root_from_configs_and_nested_yaml(
    root, location
):
    write(root, "configs/data/demo.yaml", "width: 32")
    write(root, "configs/configs/data/demo.yaml", "width: 99")
    path = write(root, location, "data: {defaults: demo}")
    assert load_configs(path)[0]["data"]["width"] == 32
    path.write_text("data: {defaults: demo, width: 64}")
    assert load_configs(path)[0]["data"]["width"] == 64


def test_a_git_file_identifies_the_project_root(root):
    (root / "pyproject.toml").unlink()
    write(root, ".git", "gitdir: somewhere")
    write(root, "configs/data/demo.yaml", "value: 42")
    path = write(root, "experiments/run.yaml", "data: {defaults: demo}")
    assert load_configs(path)[0]["data"]["value"] == 42


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


def test_runs_and_aliases_are_independent(root, load):
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
    assert load_configs(experiment)[0]["model"]["value"] == 2


def test_name_is_ordinary_data_when_no_defaults_file_matches(root, load):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = load("optimizer: {name: AdamW, lr: 0.01}\ndataset: {name: A}\n")
    assert caught == []
    assert runs == [
        {"optimizer": {"name": "AdamW", "lr": 0.01}, "dataset": {"name": "A"}}
    ]


def test_name_is_ordinary_data_even_when_a_file_matches(root, load):
    assert load("model: {name: a}", {"configs/model/a.yaml": "width: 64\n"}) == [
        {"model": {"name": "a"}}
    ]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "null",
        "[1, 2]",
        "x: [",
        "x: 1\nx: 2",
        "x: {a: 1, a: 2}",
        "1: value",
        "x: !unknown [1]",
        "x: !choice [1]",
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
        load_configs(experiment)
    assert str(experiment) in str(raised.value)


@pytest.mark.parametrize(
    "name",
    ["null", "42", "[]", "''", "'../outside'", "'a/b'", "'a\\b'", "'.'"],
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
