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
    def _load(content):
        write(root, "experiment.yaml", content)
        return load_configs(experiment)

    return _load


def test_plain_config_and_lists_produce_one_run(load):
    assert load("seed: 42\nhidden_sizes: [128, 64, 32]\noptional: null\n") == [
        {"seed": 42, "hidden_sizes": [128, 64, 32], "optional": None}
    ]
    assert load("{}") == [{}]


def test_independent_choices_have_stable_product_order(load):
    runs = load("seed: !choice [0, 1]\ntrain:\n  lr: !choice [0.1, 0.2]\n")
    assert [(run["seed"], run["train"]["lr"]) for run in runs] == [
        (0, 0.1),
        (0, 0.2),
        (1, 0.1),
        (1, 0.2),
    ]


def test_nested_choices_only_expand_selected_branch(load):
    runs = load("""
seed: !choice [0, 1]
model: !choice
  - kind: saits
    lr: !choice [0.001, 0.0001]
  - kind: brits
""")
    assert len(runs) == 6
    assert [run["model"] for run in runs[:3]] == [
        {"kind": "saits", "lr": 0.001},
        {"kind": "saits", "lr": 0.0001},
        {"kind": "brits"},
    ]


def test_choices_can_contain_lists_and_null(load):
    assert load("widths: !choice [[128, 64], [32], null]") == [
        {"widths": [128, 64]},
        {"widths": [32]},
        {"widths": None},
    ]
    assert load("items: [fixed, !choice [a, b]]") == [
        {"items": ["fixed", "a"]},
        {"items": ["fixed", "b"]},
    ]


def test_sweep_ofat_varies_one_axis_at_a_time_from_the_baseline(load):
    runs = load("""
stage: B2
tip:
  rank: 3
  epsilon: 0.0001
  cone: 0.03
sweep:
  axes:
    tip.rank: [2, 4]
    tip.epsilon: [0.00001, 0.001]
    tip.cone: [0.01]
""")
    assert [
        (run["tip"]["rank"], run["tip"]["epsilon"], run["tip"]["cone"]) for run in runs
    ] == [
        (3, 0.0001, 0.03),
        (2, 0.0001, 0.03),
        (4, 0.0001, 0.03),
        (3, 0.00001, 0.03),
        (3, 0.001, 0.03),
        (3, 0.0001, 0.01),
    ]
    assert all("sweep" not in run for run in runs)


def test_sweep_can_omit_the_baseline(load):
    runs = load(
        "a: 0\nb: 0\nc: 0\nsweep:\n  include_baseline: false\n  axes:\n    a: [1, 2]\n"
    )
    assert [(run["a"], run["b"], run["c"]) for run in runs] == [(1, 0, 0), (2, 0, 0)]


def test_sweep_grid_mode_takes_the_axis_product(load):
    runs = load(
        "a: 0\nb: 0\n"
        "sweep:\n  mode: grid\n  include_baseline: false\n"
        "  axes:\n    a: [1, 2]\n    b: [3, 4]\n"
    )
    assert [(run["a"], run["b"]) for run in runs] == [
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]


def test_sweep_combines_with_choice_expansion(load):
    runs = load("seed: !choice [0, 1]\na: 0\nsweep:\n  axes:\n    a: [1]\n")
    assert [(run["seed"], run["a"]) for run in runs] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]


def test_sweep_axis_uses_dotted_paths(load):
    runs = load("a:\n  b:\n    c: 0\nsweep:\n  axes:\n    a.b.c: [1, 2]\n")
    assert [run["a"]["b"]["c"] for run in runs] == [0, 1, 2]


def test_sweep_axis_overrides_the_default_value(root, load):
    write(root, "configs/model/a.yaml", "width: 64\ndepth: 2")
    runs = load(
        "model: {name: a, width: 128}\nsweep:\n  axes:\n    model.width: [32]\n"
    )
    assert [run["model"]["width"] for run in runs] == [128, 32]
    assert all(run["model"]["depth"] == 2 for run in runs)


@pytest.mark.parametrize(
    "content",
    [
        "x: 1\nsweep: []\n",
        "x: 1\nsweep: {}\n",
        "x: 1\nsweep: {axes: {}}\n",
        "x: 1\nsweep: {axes: {x: []}}\n",
        "x: 1\nsweep: {axes: {x: !choice [1, 2]}}\n",
        "x: 1\nsweep: {axes: {x: [2]}, mode: full}\n",
        "x: 1\nsweep: {axes: {x: [2]}, extra: 1}\n",
        "x: 1\nsweep: {axes: {x: [2]}, include_baseline: 1}\n",
    ],
)
def test_malformed_sweep_specs_are_rejected(load, content):
    with pytest.raises(ConfigError):
        load(content)


def test_sweep_is_rejected_in_defaults_files(root, load):
    path = write(root, "configs/model/a.yaml", "sweep: {axes: {x: [1]}}")
    with pytest.raises(ConfigError) as raised:
        load("model: {name: a}\nx: 0")
    assert str(path) in str(raised.value)


def test_root_choice_pairs_sibling_fields(load):
    runs = load("""
!choice
- stage: B1
  dataset: {name: A}
  model: {name: a}
- stage: B1
  dataset: {name: B}
  model: {name: b}
""")
    assert [(run["dataset"]["name"], run["model"]["name"]) for run in runs] == [
        ("A", "a"),
        ("B", "b"),
    ]


def test_root_choice_candidates_expand_inner_choices(load):
    runs = load("""
!choice
- dataset: A
  model: a
  seed: !choice [0, 1]
- dataset: B
  model: b
""")
    assert [(run["dataset"], run["model"], run.get("seed")) for run in runs] == [
        ("A", "a", 0),
        ("A", "a", 1),
        ("B", "b", None),
    ]


def test_root_choice_resolves_named_defaults(root, load):
    write(root, "configs/model/a.yaml", "width: 64")
    write(root, "configs/model/b.yaml", "width: 32")
    runs = load("""
!choice
- dataset: A
  model: {name: a}
- dataset: B
  model: {name: b}
""")
    assert [(run["model"]["name"], run["model"]["width"]) for run in runs] == [
        ("a", 64),
        ("b", 32),
    ]


@pytest.mark.parametrize("content", ["!choice [1, 2]\n", "!choice [[a], [b]]\n"])
def test_root_choice_requires_mapping_candidates(load, content):
    with pytest.raises(ConfigError):
        load(content)


def test_root_choice_rejects_sweep_inside_a_candidate(load):
    with pytest.raises(ConfigError, match="sweep"):
        load("""
!choice
- dataset: A
  sweep: {axes: {dataset: [B]}}
- dataset: B
""")


def test_root_choice_rejects_device_choices(load):
    with pytest.raises(ConfigError):
        load("""
!choice
- device: !choice [[0], [1]]
  seed: 0
- device: [0]
  seed: 0
""")


def test_two_model_roles_load_after_selection_without_parameter_leaks(root, load):
    write(root, "configs/models/imputation/saits.yaml", "width: 64\nonly_saits: true")
    write(root, "configs/models/imputation/brits.yaml", "width: 32")
    write(
        root, "configs/models/forecasting/patchtst.yaml", "patch_len: 32\nd_model: 128"
    )
    runs = load("""
seed: !choice [0, 1, 2]
models:
  imputation: !choice
    - name: saits
      width: 128
    - name: brits
  forecasting:
    name: patchtst
    patch_len: !choice [8, 16]
""")
    assert len(runs) == 12
    for run in runs:
        model = run["models"]["imputation"]
        if model["name"] == "saits":
            assert model == {"name": "saits", "width": 128, "only_saits": True}
        else:
            assert model == {"name": "brits", "width": 32}
        assert run["models"]["forecasting"]["d_model"] == 128


def test_recursive_merge_replaces_lists_scalars_and_null(root, load):
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
  name: a
  optimizer:
    lr: 0.0005
  layers: [32]
  optional: null
  replace_mapping: false
  replace_scalar: {value: 2}
""")
    assert runs[0]["model"] == {
        "name": "a",
        "optimizer": {"lr": 0.0005, "weight_decay": 0.01},
        "layers": [32],
        "optional": None,
        "replace_mapping": False,
        "replace_scalar": {"value": 2},
    }
    assert defaults.read_bytes() == original


def test_runs_and_aliases_are_independent(root, load):
    write(root, "configs/model/a.yaml", "layers: [128, 64]")
    runs = load("""
seed: !choice [0, 1]
left: &shared {items: [1, 2]}
right: *shared
model: {name: a}
""")
    runs[0]["left"]["items"].append(3)
    runs[0]["model"]["layers"].append(32)
    assert runs[0]["right"]["items"] == [1, 2]
    assert runs[1]["left"]["items"] == [1, 2]
    assert runs[1]["model"]["layers"] == [128, 64]


def test_missing_defaults_warn_once_per_file_and_preserve_explicit(root, load):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = load("seed: !choice [0, 1, 2]\nmodel: {name: missing, width: 32}")
    assert len(caught) == 1
    assert caught[0].category is MissingConfigWarning
    assert str(root / "configs/model/missing.yaml") in str(caught[0].message)
    assert "$.model.name" in str(caught[0].message)
    assert all(run["model"] == {"name": "missing", "width": 32} for run in runs)


def test_name_itself_can_be_a_choice(root, load):
    write(root, "configs/model/a.yaml", "width: 1")
    write(root, "configs/model/b.yaml", "width: 2")
    assert load("model: {name: !choice [a, b]}") == [
        {"model": {"width": 1, "name": "a"}},
        {"model": {"width": 2, "name": "b"}},
    ]


def test_defaults_paths_follow_project_root_and_field_keys(root):
    path = write(root, "nested/experiment.yaml", "items: [{name: a}]\n")
    write(root, "configs/items/a.yaml", "value: 42")
    assert load_configs(str(path)) == [{"items": [{"name": "a", "value": 42}]}]


@pytest.mark.parametrize(
    "location",
    ["configs/run.yaml", "configs/experiments/run.yaml", "experiments/run.yaml"],
)
def test_data_defaults_use_project_root_from_configs_and_nested_yaml(root, location):
    write(root, "configs/data/demo.yaml", "width: 32")
    write(root, "configs/configs/data/demo.yaml", "width: 99")
    path = write(root, location, "data: {name: demo}")
    assert load_configs(path)[0]["data"]["width"] == 32
    path.write_text("data: {name: demo, width: 64}")
    assert load_configs(path)[0]["data"]["width"] == 64


def test_git_file_identifies_project_root(root):
    (root / "pyproject.toml").unlink()
    write(root, ".git", "gitdir: somewhere")
    write(root, "configs/data/demo.yaml", "value: 42")
    path = write(root, "experiments/run.yaml", "data: {name: demo}")
    assert load_configs(path)[0]["data"]["value"] == 42


def test_nested_name_from_defaults_is_resolved(root, load):
    write(root, "configs/model/a.yaml", "optimizer: {name: adam}")
    write(root, "configs/model/optimizer/adam.yaml", "lr: 0.001")
    assert load("model: {name: a}")[0]["model"] == {
        "name": "a",
        "optimizer": {"name": "adam", "lr": 0.001},
    }


def test_root_name_follows_same_convention(root, load):
    write(root, "configs/demo.yaml", "value: 1")
    assert load("name: demo") == [{"name": "demo", "value": 1}]


def test_defaults_cache_is_local_to_each_load(root, load, experiment):
    path = write(root, "configs/model/a.yaml", "value: 1")
    assert load("model: {name: a}")[0]["model"]["value"] == 1
    path.write_text("value: 2")
    assert load_configs(experiment)[0]["model"]["value"] == 2


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
        "x: !choice []",
        "x: !choice scalar",
        "x: !choice {a: 1}",
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


def test_default_choice_is_rejected_even_when_explicitly_overridden(root, load):
    path = write(root, "configs/model/a.yaml", "width: !choice [32, 64]")
    with pytest.raises(ConfigError) as raised:
        load("model: {name: a, width: 128}")
    assert str(path) in str(raised.value)
    assert "!choice" in str(raised.value)


@pytest.mark.parametrize("content", ["x: [", "[1, 2]", "x: 1\nx: 2"])
def test_malformed_defaults_are_errors_not_missing_warnings(root, load, content):
    path = write(root, "configs/model/a.yaml", content)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigError) as raised:
            load("model: {name: a}")
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
        load(f"model: {{name: {name}}}")


def test_traversal_key_is_rejected(load):
    with pytest.raises(ConfigError):
        load("'../model': {name: a}")


def test_directory_instead_of_defaults_file_is_an_error(root, load):
    path = root / "configs/model/a.yaml"
    path.mkdir(parents=True)
    with pytest.raises(ConfigError) as raised:
        load("model: {name: a}")
    assert str(path) in str(raised.value)


def test_defaults_symlink_cannot_escape_config_root(root, load):
    outside = write(root, "outside.yaml", "secret: 1")
    link = root / "configs/model/a.yaml"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ConfigError, match="outside"):
        load("model: {name: a}")


def test_recursive_defaults_through_directory_alias_are_rejected(root, load):
    path = write(root, "configs/model/a.yaml", "child: {name: a}")
    try:
        (path.parent / "child").symlink_to(path.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ConfigError, match="cyclic defaults"):
        load("model: {name: a}")
