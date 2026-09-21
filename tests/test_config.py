import tempfile
import unittest
import warnings
from pathlib import Path

from expman import ConfigError, MissingConfigWarning, load_configs


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "pyproject.toml").touch()
        self.experiment = self.root / "experiment.yaml"

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def load(self, content):
        self.write("experiment.yaml", content)
        return load_configs(self.experiment)

    def test_plain_config_and_lists_produce_one_run(self):
        self.assertEqual(
            self.load("seed: 42\nhidden_sizes: [128, 64, 32]\noptional: null\n"),
            [{"seed": 42, "hidden_sizes": [128, 64, 32], "optional": None}],
        )
        self.assertEqual(self.load("{}"), [{}])

    def test_independent_choices_have_stable_product_order(self):
        runs = self.load("seed: !choice [0, 1]\ntrain:\n  lr: !choice [0.1, 0.2]\n")
        self.assertEqual(
            [(run["seed"], run["train"]["lr"]) for run in runs],
            [(0, 0.1), (0, 0.2), (1, 0.1), (1, 0.2)],
        )

    def test_nested_choices_only_expand_selected_branch(self):
        runs = self.load("""
seed: !choice [0, 1]
model: !choice
  - kind: saits
    lr: !choice [0.001, 0.0001]
  - kind: brits
""")
        self.assertEqual(len(runs), 6)
        self.assertEqual(
            [run["model"] for run in runs[:3]],
            [
                {"kind": "saits", "lr": 0.001},
                {"kind": "saits", "lr": 0.0001},
                {"kind": "brits"},
            ],
        )

    def test_choices_can_contain_lists_and_null(self):
        self.assertEqual(
            self.load("widths: !choice [[128, 64], [32], null]"),
            [{"widths": [128, 64]}, {"widths": [32]}, {"widths": None}],
        )
        self.assertEqual(
            self.load("items: [fixed, !choice [a, b]]"),
            [{"items": ["fixed", "a"]}, {"items": ["fixed", "b"]}],
        )

    def test_sweep_ofat_varies_one_axis_at_a_time_from_the_baseline(self):
        runs = self.load("""
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
        self.assertEqual(
            [
                (run["tip"]["rank"], run["tip"]["epsilon"], run["tip"]["cone"])
                for run in runs
            ],
            [
                (3, 0.0001, 0.03),
                (2, 0.0001, 0.03),
                (4, 0.0001, 0.03),
                (3, 0.00001, 0.03),
                (3, 0.001, 0.03),
                (3, 0.0001, 0.01),
            ],
        )
        self.assertTrue(all("sweep" not in run for run in runs))

    def test_sweep_can_omit_the_baseline(self):
        runs = self.load(
            "a: 0\nb: 0\nc: 0\n"
            "sweep:\n  include_baseline: false\n  axes:\n    a: [1, 2]\n"
        )
        self.assertEqual(
            [(run["a"], run["b"], run["c"]) for run in runs], [(1, 0, 0), (2, 0, 0)]
        )

    def test_sweep_grid_mode_takes_the_axis_product(self):
        runs = self.load(
            "a: 0\nb: 0\n"
            "sweep:\n  mode: grid\n  include_baseline: false\n"
            "  axes:\n    a: [1, 2]\n    b: [3, 4]\n"
        )
        self.assertEqual(
            [(run["a"], run["b"]) for run in runs],
            [(1, 3), (1, 4), (2, 3), (2, 4)],
        )

    def test_sweep_combines_with_choice_expansion(self):
        runs = self.load("seed: !choice [0, 1]\na: 0\nsweep:\n  axes:\n    a: [1]\n")
        self.assertEqual(
            [(run["seed"], run["a"]) for run in runs],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
        )

    def test_sweep_axis_uses_dotted_paths(self):
        runs = self.load("a:\n  b:\n    c: 0\nsweep:\n  axes:\n    a.b.c: [1, 2]\n")
        self.assertEqual([run["a"]["b"]["c"] for run in runs], [0, 1, 2])

    def test_sweep_axis_overrides_the_default_value(self):
        self.write("configs/model/a.yaml", "width: 64\ndepth: 2")
        runs = self.load(
            "model: {name: a, width: 128}\nsweep:\n  axes:\n    model.width: [32]\n"
        )
        self.assertEqual([run["model"]["width"] for run in runs], [128, 32])
        self.assertTrue(all(run["model"]["depth"] == 2 for run in runs))

    def test_sweep_axis_must_exist_and_not_cross_a_choice(self):
        with self.assertRaisesRegex(ConfigError, "not present"):
            self.load("a: 0\nsweep:\n  axes:\n    missing: [1]\n")
        with self.assertRaisesRegex(ConfigError, "!choice"):
            self.load("a: !choice [0, 1]\nsweep:\n  axes:\n    a: [2]\n")
        with self.assertRaisesRegex(ConfigError, "!choice"):
            self.load("a: {b: !choice [0, 1]}\nsweep:\n  axes:\n    a.b: [2]\n")

    def test_malformed_sweep_specs_are_rejected(self):
        cases = [
            "x: 1\nsweep: []\n",
            "x: 1\nsweep: {}\n",
            "x: 1\nsweep: {axes: {}}\n",
            "x: 1\nsweep: {axes: {x: []}}\n",
            "x: 1\nsweep: {axes: {x: !choice [1, 2]}}\n",
            "x: 1\nsweep: {axes: {x: [2]}, mode: full}\n",
            "x: 1\nsweep: {axes: {x: [2]}, extra: 1}\n",
            "x: 1\nsweep: {axes: {x: [2]}, include_baseline: 1}\n",
        ]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ConfigError):
                self.load(content)

    def test_sweep_is_rejected_in_defaults_files(self):
        path = self.write("configs/model/a.yaml", "sweep: {axes: {x: [1]}}")
        with self.assertRaises(ConfigError) as raised:
            self.load("model: {name: a}\nx: 0")
        self.assertIn(str(path), str(raised.exception))

    def test_root_choice_pairs_sibling_fields(self):
        runs = self.load("""
!choice
- stage: B1
  dataset: {name: A}
  model: {name: a}
- stage: B1
  dataset: {name: B}
  model: {name: b}
""")
        self.assertEqual(
            [(run["dataset"]["name"], run["model"]["name"]) for run in runs],
            [("A", "a"), ("B", "b")],
        )

    def test_root_choice_candidates_expand_inner_choices(self):
        runs = self.load("""
!choice
- dataset: A
  model: a
  seed: !choice [0, 1]
- dataset: B
  model: b
""")
        self.assertEqual(
            [(run["dataset"], run["model"], run.get("seed")) for run in runs],
            [("A", "a", 0), ("A", "a", 1), ("B", "b", None)],
        )

    def test_root_choice_resolves_named_defaults(self):
        self.write("configs/model/a.yaml", "width: 64")
        self.write("configs/model/b.yaml", "width: 32")
        runs = self.load("""
!choice
- dataset: A
  model: {name: a}
- dataset: B
  model: {name: b}
""")
        self.assertEqual(
            [(run["model"]["name"], run["model"]["width"]) for run in runs],
            [("a", 64), ("b", 32)],
        )

    def test_root_choice_requires_mapping_candidates(self):
        for content in ("!choice [1, 2]\n", "!choice [[a], [b]]\n"):
            with self.subTest(content=content), self.assertRaises(ConfigError):
                self.load(content)

    def test_root_choice_rejects_sweep_inside_a_candidate(self):
        with self.assertRaisesRegex(ConfigError, "sweep"):
            self.load("""
!choice
- dataset: A
  sweep: {axes: {dataset: [B]}}
- dataset: B
""")

    def test_root_choice_rejects_device_choices(self):
        with self.assertRaises(ConfigError):
            self.load("""
!choice
- device: !choice [[0], [1]]
  seed: 0
- device: [0]
  seed: 0
""")

    def test_two_model_roles_load_after_selection_without_parameter_leaks(self):
        self.write(
            "configs/models/imputation/saits.yaml", "width: 64\nonly_saits: true"
        )
        self.write("configs/models/imputation/brits.yaml", "width: 32")
        self.write(
            "configs/models/forecasting/patchtst.yaml", "patch_len: 32\nd_model: 128"
        )
        runs = self.load("""
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
        self.assertEqual(len(runs), 12)
        for run in runs:
            model = run["models"]["imputation"]
            if model["name"] == "saits":
                self.assertEqual(
                    model, {"name": "saits", "width": 128, "only_saits": True}
                )
            else:
                self.assertEqual(model, {"name": "brits", "width": 32})
            self.assertEqual(run["models"]["forecasting"]["d_model"], 128)

    def test_recursive_merge_replaces_lists_scalars_and_null(self):
        defaults = self.write(
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
        runs = self.load("""
model:
  name: a
  optimizer:
    lr: 0.0005
  layers: [32]
  optional: null
  replace_mapping: false
  replace_scalar: {value: 2}
""")
        self.assertEqual(
            runs[0]["model"],
            {
                "name": "a",
                "optimizer": {"lr": 0.0005, "weight_decay": 0.01},
                "layers": [32],
                "optional": None,
                "replace_mapping": False,
                "replace_scalar": {"value": 2},
            },
        )
        self.assertEqual(defaults.read_bytes(), original)

    def test_runs_and_aliases_are_independent(self):
        self.write("configs/model/a.yaml", "layers: [128, 64]")
        runs = self.load("""
seed: !choice [0, 1]
left: &shared {items: [1, 2]}
right: *shared
model: {name: a}
""")
        runs[0]["left"]["items"].append(3)
        runs[0]["model"]["layers"].append(32)
        self.assertEqual(runs[0]["right"]["items"], [1, 2])
        self.assertEqual(runs[1]["left"]["items"], [1, 2])
        self.assertEqual(runs[1]["model"]["layers"], [128, 64])

    def test_missing_defaults_warn_once_per_file_and_preserve_explicit(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runs = self.load(
                "seed: !choice [0, 1, 2]\nmodel: {name: missing, width: 32}"
            )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, MissingConfigWarning)
        self.assertIn(
            str(self.root / "configs/model/missing.yaml"), str(caught[0].message)
        )
        self.assertIn("$.model.name", str(caught[0].message))
        self.assertTrue(
            all(run["model"] == {"name": "missing", "width": 32} for run in runs)
        )

    def test_name_itself_can_be_a_choice(self):
        self.write("configs/model/a.yaml", "width: 1")
        self.write("configs/model/b.yaml", "width: 2")
        self.assertEqual(
            self.load("model: {name: !choice [a, b]}"),
            [
                {"model": {"width": 1, "name": "a"}},
                {"model": {"width": 2, "name": "b"}},
            ],
        )

    def test_defaults_paths_follow_project_root_and_field_keys(self):
        path = self.write("nested/experiment.yaml", "items: [{name: a}]\n")
        self.write("configs/items/a.yaml", "value: 42")
        self.assertEqual(
            load_configs(str(path)), [{"items": [{"name": "a", "value": 42}]}]
        )

    def test_data_defaults_use_project_root_from_configs_and_nested_yaml(self):
        self.write("configs/data/demo.yaml", "width: 32")
        self.write("configs/configs/data/demo.yaml", "width: 99")
        for location in (
            "configs/run.yaml",
            "configs/experiments/run.yaml",
            "experiments/run.yaml",
        ):
            with self.subTest(location=location):
                path = self.write(location, "data: {name: demo}")
                self.assertEqual(load_configs(path)[0]["data"]["width"], 32)
                path.write_text("data: {name: demo, width: 64}")
                self.assertEqual(load_configs(path)[0]["data"]["width"], 64)

    def test_git_file_identifies_project_root(self):
        (self.root / "pyproject.toml").unlink()
        self.write(".git", "gitdir: somewhere")
        self.write("configs/data/demo.yaml", "value: 42")
        path = self.write("experiments/run.yaml", "data: {name: demo}")
        self.assertEqual(load_configs(path)[0]["data"]["value"], 42)

    def test_nested_name_from_defaults_is_resolved(self):
        self.write("configs/model/a.yaml", "optimizer: {name: adam}")
        self.write("configs/model/optimizer/adam.yaml", "lr: 0.001")
        self.assertEqual(
            self.load("model: {name: a}")[0]["model"],
            {
                "name": "a",
                "optimizer": {"name": "adam", "lr": 0.001},
            },
        )

    def test_root_name_follows_same_convention(self):
        self.write("configs/demo.yaml", "value: 1")
        self.assertEqual(self.load("name: demo"), [{"name": "demo", "value": 1}])

    def test_defaults_cache_is_local_to_each_load(self):
        path = self.write("configs/model/a.yaml", "value: 1")
        self.assertEqual(self.load("model: {name: a}")[0]["model"]["value"], 1)
        path.write_text("value: 2")
        self.assertEqual(load_configs(self.experiment)[0]["model"]["value"], 2)

    def test_invalid_documents_report_source(self):
        cases = [
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
        ]
        for content in cases:
            with (
                self.subTest(content=content),
                self.assertRaises(ConfigError) as raised,
            ):
                self.load(content)
            self.assertIn(str(self.experiment), str(raised.exception))

    def test_default_choice_is_rejected_even_when_explicitly_overridden(self):
        path = self.write("configs/model/a.yaml", "width: !choice [32, 64]")
        with self.assertRaises(ConfigError) as raised:
            self.load("model: {name: a, width: 128}")
        self.assertIn(str(path), str(raised.exception))
        self.assertIn("!choice", str(raised.exception))

    def test_malformed_defaults_are_errors_not_missing_warnings(self):
        for content in ("x: [", "[1, 2]", "x: 1\nx: 2"):
            path = self.write("configs/model/a.yaml", content)
            with (
                self.subTest(content=content),
                warnings.catch_warnings(record=True) as caught,
            ):
                with self.assertRaises(ConfigError) as raised:
                    self.load("model: {name: a}")
                self.assertIn(str(path), str(raised.exception))
                self.assertEqual(caught, [])

    def test_missing_experiment_is_an_error(self):
        with self.assertRaises(ConfigError) as raised:
            load_configs(self.experiment)
        self.assertIn(str(self.experiment), str(raised.exception))

    def test_invalid_names_and_traversal_are_rejected(self):
        for name in (
            "null",
            "42",
            "[]",
            "''",
            "'../outside'",
            "'a/b'",
            "'a\\b'",
            "'.'",
        ):
            with self.subTest(name=name), self.assertRaises(ConfigError):
                self.load(f"model: {{name: {name}}}")
        with self.assertRaises(ConfigError):
            self.load("'../model': {name: a}")

    def test_directory_instead_of_defaults_file_is_an_error(self):
        path = self.root / "configs/model/a.yaml"
        path.mkdir(parents=True)
        with self.assertRaises(ConfigError) as raised:
            self.load("model: {name: a}")
        self.assertIn(str(path), str(raised.exception))

    def test_defaults_symlink_cannot_escape_config_root(self):
        outside = self.write("outside.yaml", "secret: 1")
        link = self.root / "configs/model/a.yaml"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(ConfigError, "outside"):
            self.load("model: {name: a}")

    def test_recursive_defaults_through_directory_alias_are_rejected(self):
        path = self.write("configs/model/a.yaml", "child: {name: a}")
        try:
            (path.parent / "child").symlink_to(path.parent, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(ConfigError, "cyclic defaults"):
            self.load("model: {name: a}")


if __name__ == "__main__":
    unittest.main()
