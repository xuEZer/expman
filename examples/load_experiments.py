"""Inspect the expanded configurations: python examples/load_experiments.py."""

from pathlib import Path

from expman import load_configs


def main() -> None:
    configs = load_configs(Path(__file__).with_name("experiment.yaml"))
    for index, cfg in enumerate(configs, start=1):
        models = cfg["models"]
        print(
            f"run={index} seed={cfg['seed']} "
            f"imputation={models['imputation']['name']} "
            f"forecasting={models['forecasting']['name']} "
            f"patch_len={models['forecasting']['patch_len']}"
        )


if __name__ == "__main__":
    main()
