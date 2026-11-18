"""Pure helper functions for the scheduler packing plan."""

from dataclasses import dataclass

from expman.estimation.duration_model import StageDuration

PLAN_BEAM_WIDTH = 128


@dataclass(frozen=True)
class Candidate:
    """One ready Stage with its resource reservations for this scheduling tick."""

    run_id: str
    stage_index: int
    host_reserve_kb: float
    gpu_reserve_kb: float
    gpu_cap_kb: float
    expected: StageDuration | None
    novelty: float


@dataclass(frozen=True)
class Plan:
    """A feasible placement of ready Stage candidates on the selected cards."""

    host_left_kb: float
    gpu_left_kb: tuple[float, ...]
    placements: tuple[tuple[Candidate, int], ...] = ()
    novelty: float = 0.0


def _plan_score(plan, host_capacity, gpu_totals):
    """Fill feasible capacity while balancing load against each card's total.

    Normalizing by per-tick free space would reward a job for using the busiest
    card that still fits it. Fixed totals retain both existing and planned load.
    """
    unused = (plan.host_left_kb / host_capacity) ** 2
    unused += sum(
        (left / capacity) ** 2
        for left, capacity in zip(plan.gpu_left_kb, gpu_totals, strict=True)
        if capacity > 0
    )
    # Parameter diversity only breaks materially equivalent packing choices.
    return -unused + 1e-6 * plan.novelty


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
                        plan.novelty + candidate.novelty,
                    )
                )
        plans = sorted(
            expanded,
            key=lambda plan: _plan_score(plan, host_capacity, gpu_totals),
            reverse=True,
        )[:PLAN_BEAM_WIDTH]
    return plans[0].placements
