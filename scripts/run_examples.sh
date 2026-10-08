#!/usr/bin/env bash
# Run every example script the pipeline ships, the same set CI runs, so a
# broken example fails at commit time instead of in the pipeline.
set -euo pipefail

cd "$(dirname "$0")/.."

scripts=(
    examples/basic_pipeline.py
    examples/load_experiments.py
    examples/batch_pipeline.py
    examples/checkpoint_pipeline.py
    examples/estimate_time.py
    examples/multi_gpu.py
    examples/random_state.py
    examples/shared_prefix.py
)

for script in "${scripts[@]}"; do
    echo "--- ${script}"
    uv run python "${script}"
done
