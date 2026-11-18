"""Pytest integration for the real CUDA/MPS suite."""

import pytest
from gpu_allocation import reserve_card


@pytest.fixture(scope="session")
def gpu_device(request):
    with reserve_card() as card:
        reporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"CUDA/MPS tests selected physical GPU {card.index} ({card.uuid}), "
                f"{card.utilization}% utilization, {card.free_mib} MiB free"
            )
        yield card.index
