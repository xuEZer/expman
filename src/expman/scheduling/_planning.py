"""Pure helper functions for the scheduler packing plan."""

from dataclasses import dataclass

from expman.estimation.duration_model import StageDuration
from expman.settings import settings


@dataclass(frozen=True)
class Candidate:
    """One ready Stage with its resource reservations for this scheduling tick."""

    run_id: str
    stage_index: int
    host_reserve_kb: float
    gpu_reserve_kb: float
    gpu_cap_kb: float
    expected: StageDuration | None


@dataclass(frozen=True)
class Plan:
    """A feasible placement of ready Stage candidates on the selected cards."""

    host_left_kb: float
    gpu_left_kb: tuple[float, ...]
    placements: tuple[tuple[Candidate, int], ...] = ()


def _packing_cost(plan, host_capacity, gpu_totals):
    """Sum the squared fractions of unused host and GPU memory.

    The host fraction uses the available budget at the start of this tick.
    Each GPU fraction uses the card's physical capacity so its existing load
    counts alongside the new placements.
    """
    unused = (plan.host_left_kb / host_capacity) ** 2
    unused += sum(
        (left / capacity) ** 2
        for left, capacity in zip(plan.gpu_left_kb, gpu_totals, strict=True)
        if capacity > 0
    )
    return unused


def pack_candidates(candidates, host_capacity, devices, gpu_capacities, gpu_totals):
    """Choose placements for the current headroom, balancing load against card totals."""
    plans = [Plan(host_capacity, gpu_capacities)]
    for candidate in candidates:
        expanded = list(plans)
        for plan in plans:
            if candidate.host_reserve_kb > plan.host_left_kb:
                continue
            for index, gpu_left in enumerate(plan.gpu_left_kb):
                if candidate.gpu_reserve_kb > gpu_left:
                    continue
                left = list(plan.gpu_left_kb)
                left[index] -= candidate.gpu_reserve_kb
                expanded.append(
                    Plan(
                        plan.host_left_kb - candidate.host_reserve_kb,
                        tuple(left),
                        (*plan.placements, (candidate, devices[index])),
                    )
                )
        plans = sorted(
            expanded,
            key=lambda plan: _packing_cost(plan, host_capacity, gpu_totals),
        )[: settings.plan_beam_width]
    return plans[0].placements
