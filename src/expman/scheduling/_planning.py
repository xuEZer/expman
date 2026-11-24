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
    mps_slots_left: tuple[int, ...]
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


def pack_candidates(
    candidates, host_capacity, devices, gpu_capacities, gpu_totals, mps_slots
):
    """Fit candidates within memory and client slots, balancing memory across cards."""
    plans = [Plan(host_capacity, gpu_capacities, mps_slots)]
    for candidate in candidates:
        expanded = list(plans)
        for plan in plans:
            if candidate.host_reserve_kb > plan.host_left_kb:
                continue
            for index, gpu_left in enumerate(plan.gpu_left_kb):
                if (
                    plan.mps_slots_left[index] == 0
                    or candidate.gpu_reserve_kb > gpu_left
                ):
                    continue
                left = list(plan.gpu_left_kb)
                left[index] -= candidate.gpu_reserve_kb
                slots_left = list(plan.mps_slots_left)
                slots_left[index] -= 1
                expanded.append(
                    Plan(
                        plan.host_left_kb - candidate.host_reserve_kb,
                        tuple(left),
                        tuple(slots_left),
                        (*plan.placements, (candidate, devices[index])),
                    )
                )
        plans = sorted(
            expanded,
            key=lambda plan: _packing_cost(plan, host_capacity, gpu_totals),
        )[: settings.plan_beam_width]
    return plans[0].placements
